from utils_models_qwen import *
from utils_libs import *
from utils_dataset_qwen import *
from utils_methods import *
from huggingface_hub import login, hf_hub_download, snapshot_download
import json
import sentencepiece as spm

from utils_downloads import *
from pathlib import Path
import torch

import os
from safetensors.torch import load_file

import re
from tokenizers import Tokenizer
from transformers import AutoTokenizer


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


    ###########################################################################################
    # Configure QWEN-3 and Load Pre-trained Weights for the Instruct Model from Hugginface Hub
    ###########################################################################################
    # Select which model to use via the following flag; only one can be True

    USE_BASE_MODEL = False
    USE_REASONING_MODEL = False #True 
    USE_INSTRUCT_MODEL = True #False

    if (USE_BASE_MODEL + USE_REASONING_MODEL
        + USE_INSTRUCT_MODEL) != 1:
        raise AttributeError("Only one of the options above can be True.")


    # Initialize Model
    CHOOSE_MODEL = "8B"

    if CHOOSE_MODEL == "0.6B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,           # Vocabulary size
            "context_length": 40_960,        # Context length that was used to train the model
            "emb_dim": 1024,                 # Embedding dimension
            "n_heads": 16,                   # Number of attention heads
            "n_layers": 28,                  # Number of layers
            "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
            "head_dim": 128,                 # Size of the heads in GQA
            "qk_norm": True,                 # Whether to normalize queries and keys in GQA
            "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
            "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
            "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
        }

    elif CHOOSE_MODEL == "1.7B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 2048,                 # 2x larger than above
            "n_heads": 16,
            "n_layers": 28,
            "hidden_dim": 6144,              # 2x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }   

    elif CHOOSE_MODEL == "4B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 2560,                 # 25% larger than above
            "n_heads": 32,                   # 2x larger than above
            "n_layers": 36,                  # 29% larger than above
            "hidden_dim": 9728,              # ~3x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }  

    elif CHOOSE_MODEL == "8B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 4096,                 # 60% larger than above
            "n_heads": 32,
            "n_layers": 36,                  # 26% larger than above
            "hidden_dim": 12288,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        } 

    elif CHOOSE_MODEL == "14B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 5120,                 # 25% larger than above
            "n_heads": 40,                   # 25% larger than above
            "n_layers": 40,                  # 11% larger than above
            "hidden_dim": 17408,             # 42% larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        } 

    elif CHOOSE_MODEL == "32B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 5120,                
            "n_heads": 64,                   # 60% larger than above
            "n_layers": 64,                  # 60% larger than above
            "hidden_dim": 25600,             # 47% larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        } 

    else:
        raise ValueError(f"{CHOOSE_MODEL} is not supported.")


    torch.manual_seed(123)
    model = Qwen3Model(QWEN3_CONFIG)

    # Move model to device - GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)


    # Load pre-trained weights
    def load_weights_into_qwen(model, param_config, params):
        def assign(left, right, tensor_name="unknown"):
            if left.shape != right.shape:
                raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
            
            with torch.no_grad():
                if isinstance(right, torch.Tensor):
                    left.copy_(right)
                else:
                    left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))
        
            return left 

        model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

        for l in range(param_config["n_layers"]):
            block = model.trf_blocks[l]
            att = block.att

            # Q, K, V projections
            att.W_query.weight = assign(
                att.W_query.weight,
                params[f"model.layers.{l}.self_attn.q_proj.weight"],
                f"model.layers.{l}.self_attn.q_proj.weight"
            )
            att.W_key.weight = assign(
                att.W_key.weight,
                params[f"model.layers.{l}.self_attn.k_proj.weight"],
                f"model.layers.{l}.self_attn.k_proj.weight"
            )
            att.W_value.weight = assign(
                att.W_value.weight,
                params[f"model.layers.{l}.self_attn.v_proj.weight"],
                f"model.layers.{l}.self_attn.v_proj.weight"
            )

            # Output projection
            att.out_proj.weight = assign(
                att.out_proj.weight,
                params[f"model.layers.{l}.self_attn.o_proj.weight"],
                f"model.layers.{l}.self_attn.o_proj.weight"
            )

            # QK norms
            if hasattr(att, "q_norm") and att.q_norm is not None:
                att.q_norm.scale = assign(
                    att.q_norm.scale,
                    params[f"model.layers.{l}.self_attn.q_norm.weight"],
                    f"model.layers.{l}.self_attn.q_norm.weight"
                )
            if hasattr(att, "k_norm") and att.k_norm is not None:
                att.k_norm.scale = assign(
                    att.k_norm.scale,
                    params[f"model.layers.{l}.self_attn.k_norm.weight"],
                    f"model.layers.{l}.self_attn.k_norm.weight"
                )

            # Attention layernorm
            block.norm1.scale = assign(
                block.norm1.scale,
                params[f"model.layers.{l}.input_layernorm.weight"],
                f"model.layers.{l}.input_layernorm.weight"
            )

            # Feedforward weights
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight"
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight"
            )
            block.norm2.scale = assign(
                block.norm2.scale,
                params[f"model.layers.{l}.post_attention_layernorm.weight"],
                f"model.layers.{l}.post_attention_layernorm.weight"
            )

        # Final normalization and output head
        model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

        if "lm_head.weight" in params:
            model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
        else:
            model.out_head.weight = model.tok_emb.weight
            print("Model uses weight tying.")


    # login to huggngface to access model Llama tokenizer
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        access_token = config["HF_ACCESS_TOKEN"]

    login(token=access_token)


    if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
        repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}"
    else:
        repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}-Base"

    local_dir = Path(repo_id).parts[-1]

    if CHOOSE_MODEL == "0.6B":
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights_dict = load_file(weights_file)
    elif CHOOSE_MODEL == "XB":
        shard_files = [
          f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)
        ]
        full_state_weights_dict = {}
        for shard in shard_files:
            print(f"Downloading and loading {shard}...")
            # 1. Download shard from HF Hub
            local_shard_path = hf_hub_download(repo_id=repo_id, filename=shard, local_dir=local_dir)
            
            # 2. Load the safetensors file
            shard_weights_dict = load_file(local_shard_path)
            
            # 3. Merge into the master state_dict
            full_state_weights_dict.update(shard_weights_dict)

        print(f"Successfully loaded {len(full_state_weight_dict)} tensors.")
        weights_dict = full_state_weight_dict
    else:
        repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        index_path = os.path.join(repo_dir, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)

    load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
    model.to(device)
    del weights_dict


    #######################################
    # Load Tokenizer
    #######################################
    class Qwen3Tokenizer:
        _SPECIALS = [
            "<|endoftext|>",
            "<|im_start|>", "<|im_end|>",
            "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>",
            "<|quad_start|>", "<|quad_end|>",
            "<|vision_start|>", "<|vision_end|>",
            "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
            "<think>", "</think>"
        ]
        _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")

        def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None,
                    apply_chat_template=True, add_generation_prompt=False, add_thinking=False):

            self.apply_chat_template = apply_chat_template
            self.add_generation_prompt = add_generation_prompt
            self.add_thinking = add_thinking

            tok_file = Path(tokenizer_file_path)
            if not tok_file.exists():
                raise FileNotFoundError(f"Could not find {tok_file} after download.")
            self._tok = Tokenizer.from_file(str(tok_file))
            self._special_to_id = {}
            for t in self._SPECIALS:
                tid = self._tok.token_to_id(t)
                if tid is not None:
                    self._special_to_id[t] = tid

            self.pad_token_id = self._special_to_id["<|endoftext|>"]
            self.eos_token_id = self.pad_token_id

            if repo_id and "Base" not in repo_id:
                eos_token = "<|im_end|>"
            else:
                eos_token = "<|endoftext|>"
            if eos_token in self._special_to_id:
                self.eos_token_id = self._special_to_id[eos_token]

        def encode(self, text, chat_wrapped=None):
            if chat_wrapped is None:
                chat_wrapped = self.apply_chat_template

            stripped = text.strip()
            if stripped in self._special_to_id and "\n" not in stripped:
                return [self._special_to_id[stripped]]

            if chat_wrapped:
                text = self._wrap_chat(text)

            ids = []
            for part in filter(None, self._SPLIT_RE.split(text)):
                if part in self._special_to_id:
                    ids.append(self._special_to_id[part])
                else:
                    ids.extend(self._tok.encode(part).ids)
            return ids

        def decode(self, ids):
            return self._tok.decode(ids, skip_special_tokens=False)

        def _wrap_chat(self, user_msg):
            s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            if self.add_generation_prompt:
                s += "<|im_start|>assistant"
                if self.add_thinking:
                    s += "\n"
                else:
                    s += "\n<think>\n\n</think>\n\n"
            return s


    if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
        tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}/tokenizer.json"
    else:
        tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}-Base/tokenizer.json"

    hf_hub_download(
        repo_id=repo_id,
        filename="tokenizer.json",
        local_dir=local_dir,
    )

    if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_file_path,
            repo_id=repo_id,
            apply_chat_template= False, #True
            add_generation_prompt=True,
            add_thinking= False #USE_REASONING_MODEL
        )

    else:
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_file_path,
            repo_id=repo_id,
            apply_chat_template=False,
            add_generation_prompt=False,
            add_thinking=False
        )
    

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

    customized_collate_fn_grammar = partial(custom_collate_fn_grammar, pad_token_id=tokenizer.eos_token_id, device=device, allowed_max_length=1024)
    #customized_collate_fn = partial(custom_collate_fn, pad_token_id=tokenizer.eos_token_id, device=device, allowed_max_length=1024)

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = AtcDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn_grammar,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = AtcDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn_grammar,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # Load pretrained model
    #######################################

    # Small LLAMA model for testing purposes
    if args.test_mode:
        QWEN3_CONFIG = {
            "vocab_size": 32000,     # Vocabulary size
            "context_length": 4096,  # Context length
            "emb_dim": 1024,         # Embedding dimension
            "n_heads": 16,           # Number of attention heads
            "n_layers": 12,          # Number of layers
            "hidden_dim": 4096,     # NEW: Size of the intermediate dimension in FeedForward
            "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
        }

        model = Qwen3Model(QWEN3_CONFIG)
        model.eval()
        device = "cpu"
        CHOOSE_MODEL = "Small test model"

    # Main code
    else:
        '''LLAMA2_CONFIG_7B = {
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
        }'''

        CHOOSE_MODEL = "Qwen-2-8b (8B)"
        #CHOOSE_MODEL = "Llama2-7b-chat (7B)"
        #CHOOSE_MODEL = "gpt2-large (774M)"
        #CHOOSE_MODEL = "gpt2-xl (1558M)"'''

        #QWEN3_CONFIG.update(model_configs[CHOOSE_MODEL])

        '''weights_file = hf_hub_download(
          repo_id="meta-llama/Llama-2-7b-chat",
          filename="consolidated.00.pth",
          local_dir="Llama-2-7b-chat"
        )

        weights = torch.load(weights_file, weights_only=True)
        model = Llama2Model(LLAMA2_CONFIG_7B)
        # Enable checkpointing to reduce memory footprint
        load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)
        model.to(device)'''
        model = model

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")



    ######################################################
    # Finetuning the llama model with gramar-informed loss
    ######################################################
    # Load the saved tensor of valid ATC vocabulary (token ids)
    V_ATC_ids = torch.load("V_ATC_ids_qwen3.pt")
    
    print("Initial losses")

    # Pure CLM baseline (if you have a 2-tensor collate loader)
    # train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)

    # Grammar-aware losses (new 3-tensor collate)
    with torch.no_grad():
        V_ATC_ids_dev = V_ATC_ids.to(device)
        train_loss = calc_loss_loader_grammar(
            train_loader, model, device, 
            V_ATC_ids=V_ATC_ids_dev, lambda_vocab=0.1,  # or your lambda
            num_batches=5
        )
        val_loss = calc_loss_loader_grammar(
            val_loader, model, device, 
            V_ATC_ids=V_ATC_ids_dev, lambda_vocab=0.1,
            num_batches=5
        )


    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 5 #2


    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen = train_model_simple_with_grammar(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_atc(val_data[0]), tokenizer=tokenizer,
        use_grammar_loss=True, V_ATC_ids=V_ATC_ids, lambda_vocab=0.1
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    model_size = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }"
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, model_size + "grammar-loss")

    # Save Losses data in file
    loss_data_path = f"loss-data-qwen-grammar-loss{re.sub(r'[ ()]', '', CHOOSE_MODEL) }.json"
    loss_data = num_epochs, tokens_seen, train_losses, val_losses, model_size + "grammar-loss", execution_time_minutes
    with open(loss_data_path, "w") as file:
        json.dump(loss_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Loss data saved as {loss_data_path}")

    print(50*"-")

    #######################################
    # Saving results
    #######################################
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        # Force the stop token to be <|im_end|> for Qwen Instruct
        stop_id = text_to_token_ids("<|im_end|>", tokenizer)

        input_text = format_atc(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=QWEN3_CONFIG["context_length"],
            eos_id= tokenizer.eos_token_id,
            repetition_penalty=1.2,
            top_k=1,
            temperature=0.
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        #response_text = generated_text[len(input_text):].replace("ATC:", "").replace("UAV Pilot:", "").strip()
        response_text = generated_text[len(input_text):].replace("### ATC:", "").replace("### UAV Pilot:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = f"atc-test-data-with-response-qwen-grammar-loss{re.sub(r'[ ()]', '', CHOOSE_MODEL) }.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Responses saved as {test_data_path}")

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-atc-qwen-grammar-loss.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")
    print(model)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Finetune QWEN model for atc dialogue"
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

    
