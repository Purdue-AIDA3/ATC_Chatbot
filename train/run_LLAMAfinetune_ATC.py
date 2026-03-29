from utils_models_llama2 import *
from utils_libs import *
from utils_dataset_llama import *
from utils_methods import *
from huggingface_hub import login, hf_hub_download
import json
import sentencepiece as spm

from utils_downloads import *
from pathlib import Path


###########################################################################################
# Configure LLAMA-2 and Load Pre-trained Weights for the Instruct Model from Hugginface Hub
###########################################################################################
LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}

model = Llama2Model(LLAMA2_CONFIG_7B)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# Send model to device (GPU is available - require 26.17GB for bfloat16)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)

# login to huggngface to access model Llama tokenizer
with open("config.json", "r") as config_file:
    config = json.load(config_file)
    access_token = config["HF_ACCESS_TOKEN"]

login(token=access_token)

# Get Llama tokenizer
tokenizer_file = hf_hub_download(
    repo_id="meta-llama/Llama-2-7b",
    filename="tokenizer.model",
    local_dir="Llama-2-7b"
)

# define a class to easily access tokenizer
class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.encode(text, out_type=int)

    def decode(self, ids):
        return self.tokenizer.decode(ids)


tokenizer = LlamaTokenizer(tokenizer_file)


# Load pre-trained weights for LLama2 from huggingface hub
# Some helper functions to load weight is first defined
def assign(left, right, tensor_name="unknown"):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
    
    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right)
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    return left 


def permute(w: torch.Tensor, n_heads, out_dim, in_dim):
    return (w.view(n_heads, out_dim // n_heads // 2, 2, in_dim)
             .transpose(1, 2)          # put axis 2 next to heads
             .reshape(out_dim, in_dim))


def load_weights_into_llama(model, param_config, params):

    cfg = LLAMA2_CONFIG_7B
    
    model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"])

    for l in range(param_config["n_layers"]):

        # The original Meta/Llama checkpoints store Q and K so that the two numbers 
        # that form one complex RoPE pair sit next to each other inside the head dimension ("sliced" layout).
        # Our RoPE implementation, similar to the one in Hugging Face, expects an interleaved layout
        # For example, with n_heads=2 and head_dim = 8
        #                         ┌── pair 0 ──┐      ┌── pair 1 ──┐
        # Meta (sliced):    [ h0:  r0 r1 r2 r3,   h1:  r0 r1 r2 r3  ]
        # Ours & HF (interleaved):  [ h0: r0 r0 r1 r1 r2 r2 r3 r3  , h1: ... ]
        # For more information, please see the discussion in the PR: https://github.com/rasbt/LLMs-from-scratch/pull/747 
        
        # So, below, for q_raw and k_raw, we must re‑order the checkpoint weights using the slices_to_interleave helper

        q_raw = params[f"layers.{l}.attention.wq.weight"]
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            permute(q_raw, cfg["n_heads"], cfg["emb_dim"], cfg["emb_dim"])
        )
        k_raw = params[f"layers.{l}.attention.wk.weight"]
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            permute(k_raw, cfg["n_heads"], cfg["emb_dim"], cfg["emb_dim"])
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"layers.{l}.attention.wv.weight"]
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"layers.{l}.attention.wo.weight"]
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"layers.{l}.attention_norm.weight"]
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"layers.{l}.feed_forward.w1.weight"]
        )
        # For some reason w2 and w3 are provided in the wrong order in the weights file
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"layers.{l}.feed_forward.w3.weight"]
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"layers.{l}.feed_forward.w2.weight"]
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"layers.{l}.ffn_norm.weight"]
        )

    # Load output layer weights
    model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"])
    model.out_head.weight = assign(model.out_head.weight, params["output.weight"])


# Now load weights for llama-2 instruct model
# Loads the instruction finetuned model - "meta-llama/Llama-2-7b-chat"
weights_file = hf_hub_download(
   repo_id="meta-llama/Llama-2-7b-chat",
   filename="consolidated.00.pth",
   local_dir="Llama-2-7b-chat"
)

weights = torch.load(weights_file, weights_only=True)
model = Llama2Model(LLAMA2_CONFIG_7B)
load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)
model.to(device)


##########################################
# Prepare Trainning Dataset and DataLoader
##########################################

def main(test_mode=False):
    #######################################
    # Print package versions
    #######################################
    print()
    pkgs = [
        "matplotlib",  # Plotting library
        "tiktoken",    # Tokenizer
        "torch",       # Deep learning library
        "tqdm",        # Progress bar
        "tensorflow",  # For OpenAI's pretrained weights
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    print(50*"-")


    #######################################
    # Load Tokenizer
    #######################################

    # login to huggngface to access model Llama tokenizer
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        access_token = config["HF_ACCESS_TOKEN"]

    login(token=access_token)

    # Get Llama tokenizer
    tokenizer_file = hf_hub_download(
        repo_id="meta-llama/Llama-2-7b",
        filename="tokenizer.model",
        local_dir="Llama-2-7b"
    )

    # define a class to easily access tokenizer
    class LlamaTokenizer:
        def __init__(self, tokenizer_file):
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_file)
            self.tokenizer = sp

        def encode(self, text):
            return self.tokenizer.encode(text, out_type=int)

        def decode(self, ids):
            return self.tokenizer.decode(ids)

    tokenizer = LlamaTokenizer(tokenizer_file)



    #######################################
    # Load ATC Dataset from file
    #######################################
    combined_file_path = "atc-communication-data.json"
    combined_atc_data = []

    with open(combined_file_path, "r", encoding="utf-8") as f:
        combined_atc_data = json.load(f)

    data = combined_atc_data

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

   
    # Use very small subset for testing purposes
    if test_mode:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50*"-")

    #tokenizer = tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = AtcDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = AtcDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # Load pretrained model
    #######################################

    # Small LLAMA model for testing purposes
    if args.test_mode:
        LLAMA2_CONFIG_7B = {
            "vocab_size": 32000,     # Vocabulary size
            "context_length": 4096,  # Context length
            "emb_dim": 1024,         # Embedding dimension
            "n_heads": 16,           # Number of attention heads
            "n_layers": 12,          # Number of layers
            "hidden_dim": 4096,     # NEW: Size of the intermediate dimension in FeedForward
            "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
        }

        model = Llama2Model(LLAMA2_CONFIG_7B)
        model.eval()
        device = "cpu"
        CHOOSE_MODEL = "Small test model"

    # Main code
    else:
        LLAMA2_CONFIG_7B = {
            "vocab_size": 32000,     # Vocabulary size
            "context_length": 4096,  # Context length
            "emb_dim": 4096,         # Embedding dimension
            "n_heads": 32,           # Number of attention heads
            "n_layers": 32,          # Number of layers
            "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
            "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
        }

        model_configs = {
            "Llama-2-7b (7B)": {"emb_dim": 4096, "n_layers": 32, "n_heads": 32},
            "Llama2-7b-chat (7B)": {"emb_dim": 4096, "n_layers": 32, "n_heads": 32},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        CHOOSE_MODEL = "Llama-2-7b (7B)"
        #CHOOSE_MODEL = "Llama2-7b-chat (7B)"
        #CHOOSE_MODEL = "gpt2-large (774M)"
        #CHOOSE_MODEL = "gpt2-xl (1558M)"

        LLAMA2_CONFIG_7B.update(model_configs[CHOOSE_MODEL])

        weights_file = hf_hub_download(
          repo_id="meta-llama/Llama-2-7b-chat",
          filename="consolidated.00.pth",
          local_dir="Llama-2-7b-chat"
        )

        weights = torch.load(weights_file, weights_only=True)
        model = Llama2Model(LLAMA2_CONFIG_7B)
        # Enable checkpointing to reduce memory footprint
        load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)
        model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    ########################################################################
    # Before Finetuning the causal model, save checkpoint for original model
    ########################################################################
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-atc-llama2-clm-original-model.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Original Model before finetuning saved as {file_name}")

    #######################################
    # Finetuning the model
    #######################################
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 5 #2

    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_atc(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    model_size = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }"
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, model_size + "causal-loss")
    
    
    # Save Losses data in file
    loss_data_path = f"loss-data-llama-clm-loss{re.sub(r'[ ()]', '', CHOOSE_MODEL) }.json"
    loss_data = num_epochs, tokens_seen, train_losses, val_losses, model_size + "clm-loss", execution_time_minutes
    with open(loss_data_path, "w") as file:
        json.dump(loss_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Loss data saved as {loss_data_path}")
    
    print(50*"-")

    #######################################
    # Saving results
    #######################################
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_atc(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=LLAMA2_CONFIG_7B["context_length"],
            eos_id=2,
            top_k=1,
            temperature=0.
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### ATC:", "").replace("### UAV Pilot:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = f"atc-test-data-with-response-Llama-clm-loss{re.sub(r'[ ()]', '', CHOOSE_MODEL) }.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Responses saved as {test_data_path}")

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-atc-Llama-clm-loss.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")
    print(model)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Finetune a LLAMA model for atc dialogue"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode for internal testing purposes. "
              "Otherwise, it runs the model as it is used in the chapter (recommended).")
    )
    #args = parser.parse_args()

    # parse_known_args returns a tuple (namespace, unknown_args)
    args, unknown = parser.parse_known_args()

    main(args.test_mode)

    
