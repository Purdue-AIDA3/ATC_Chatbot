from utils_libs import *
from utils_dataset_gpt import *
from utils_models import *
from utils_methods import *

from utils_downloads import *
from pathlib import Path

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
    # Download and prepare dataset
    #######################################
    combined_file_path = "atc-communication-data.json"
    combined_atc_data = []

    '''
    file_166_ = "AAL166.json"
    file_736_ = "AAL736.json"
    file_94_ = "AAL94.json"

    with open(file_166_, "r", encoding="utf-8") as f:
        data = json.load(f)
        for dialogue in data["exchange_pairs"]:
            dialogue["callsign"] = data["aircraft_callsign"]
        data_166_dial = data["exchange_pairs"]
        
    with open(file_736_, "r", encoding="utf-8") as f:
        data = json.load(f)
        for dialogue in data["exchange_pairs"]:
            dialogue["callsign"] = data["aircraft_callsign"]
        data_736_dial = data["exchange_pairs"]

    with open(file_94_, "r", encoding="utf-8") as f:
        data = json.load(f)
        for dialogue in data["exchange_pairs"]:
            dialogue["callsign"] = data["aircraft_callsign"]
        data_94_dial = data["exchange_pairs"]


    # Get dialogues from each atc dataset and combine into one
    combined_atc_data += data_166_dial
    combined_atc_data += data_736_dial
    combined_atc_data += data_94_dial

    # Load Boston Data
    folder_path = Path("Boston_JSON/")

    # Use glob() for pattern matching in the current directory
    for file_path in folder_path.glob("*.json"):
        print(f"Loading {file_path.name}")
        # Open and process the file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for dialogue in data["exchange_pairs"]:
                dialogue["callsign"] = data["aircraft_callsign"]
            data_dial = data["exchange_pairs"]
            combined_atc_data += data_dial
    
    data = combined_atc_data

    # Write the combined atc_data into a file
    with open(combined_file_path, "w") as file:
        json.dump(combined_atc_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Combined atc data saved as {combined_file_path}")

    '''

    # LOAD DATA FROM FILE
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

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    customized_collate_fn_grammar = partial(custom_collate_fn_grammar, device=device, allowed_max_length=1024)

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

    # Small GPT model for testing purposes
    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 120,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"
        CHOOSE_MODEL = "Small test model"

    # Code as it is used in the main chapter
    else:
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

        #CHOOSE_MODEL = "gpt2-medium (355M)"
        #CHOOSE_MODEL = "gpt2-small (124M)"
        CHOOSE_MODEL = "gpt2-large (774M)"
        #CHOOSE_MODEL = "gpt2-xl (1558M)"

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        model.eval()
        model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    #######################################
    # Finetuning the model
    #######################################
    # Load the saved tensor of valid ATC vocabulary (token ids)
    V_ATC_ids = torch.load("V_ATC_ids.pt")
    
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
    loss_data_path = f"loss-data-gpt2-grammar-loss{re.sub(r'[ ()]', '', CHOOSE_MODEL) }.json"
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

        input_text = format_atc(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### ATC:", "").replace("### UAV Pilot:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = f"atc-test-data-with-response-grammar-loss{re.sub(r'[ ()]', '', CHOOSE_MODEL) }.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Responses saved as {test_data_path}")

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-atc-with-grammar-loss.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")
    print(model)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Finetune a GPT model for atc dialogue with grammar-informed loss"
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

    