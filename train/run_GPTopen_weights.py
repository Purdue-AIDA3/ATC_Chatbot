from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_methods import *

from utils_downloads import *


def main(gpt_config, input_prompt, model_size, device):

    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    gpt = GPTModel(gpt_config)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=25,
        context_size=gpt_config["context_length"],
        top_k=50,
        temperature=1.0
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate text with a pretrained GPT-2 model.")
    parser.add_argument(
        "--prompt",
        default="Every effort moves you",
        help="Prompt text used to seed the generation (default matches the script's built-in prompt)."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for running inference, e.g., cpu, cuda, mps, or auto. Defaults to cpu."
    )

    #args = parser.parse_args()


    # parse_known_args returns a tuple (namespace, unknown_args)
    args, unknown = parser.parse_known_args()



    torch.manual_seed(123)

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = args.prompt
    DEVICE = torch.device(args.device)

    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)


    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    main(BASE_CONFIG, INPUT_PROMPT, model_size, DEVICE)

    