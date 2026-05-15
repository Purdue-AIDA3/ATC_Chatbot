from logging import raiseExceptions
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoProcessor, WhisperForConditionalGeneration
import torch
import sys
import os
import json # Import json to read config.json
from huggingface_hub import login, hf_hub_download # Import login for Hugging Face authentication
from gtts import gTTS # Import gTTS for text-to-speech
from IPython.display import Audio # Import Audio for playing sound in Colab
import soundfile as sf # Import soundfile for loading audio
import sentencepiece as spm # Needed for LlamaTokenizer

# Add the path to your ATC_Chatbot directory so utils modules can be imported
sys.path.append('./')
sys.path.append('./train')
sys.path.append('./utils')
sys.path.append('./data')
sys.path.append('./docs')
sys.path.append('./evals')

# Import necessary custom classes and functions
from utils_models import GPTModel  # Assuming GPTModel is in utils_models.py
from utils_methods import generate, text_to_token_ids, token_ids_to_text
from utils_downloads import download_and_load_gpt2, load_weights_into_gpt
import tiktoken

# Llama specific imports
from utils_models_llama2 import Llama2Model
from run_LLAMAfinetune_with_Grammar_ATC import assign, permute, load_weights_into_llama as load_llama_weights

from openai import OpenAI
import os

# Get the secret value from config.json
config_path = "./config.json"
try:
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
        api_key = config.get('OPENAI_API_KEY')
        if api_key:
            # Set it as an environment variable in the current session
            os.environ['OPENAI_API_KEY'] = api_key
            print("Successfully set openai key in os.environ.")
        else:
            print("OPENAI_API_KEY not found in config.json. Proceeding without authentication.")
except FileNotFoundError:
    print(f"config.json not found at {config_path}. Proceeding without OpenAI authentication.")
    raise FileNotFoundError(f"config.json not found at {config_path}")
except Exception as e:
    print(f"Error during OpenAI key loading: {e}. Proceeding without authentication.")


# --- START: Added for Hugging Face Authentication ---
# Load Hugging Face token from config.json
config_path = "./config.json"
try:
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
        hf_token = config.get("HF_ACCESS_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=True)
        print("Successfully logged into Hugging Face.")
    else:
        print("HF_ACCESS_TOKEN not found in config.json. Proceeding without authentication.")
except FileNotFoundError:
    print(f"config.json not found at {config_path}. Proceeding without Hugging Face authentication.")
except Exception as e:
    print(f"Error during Hugging Face login: {e}. Proceeding without authentication.")
# --- END: Added for Hugging Face Authentication ---


# 1. Load fine-tuned model (Replace with local checkpoint path)
#model_weights_path = "./gpt2-large774M-atc-with-grammar-loss.pth"
model_weights_path = "./gpt2-large774M-atc-clm-loss.pth"
# Download a model from huggingface
model_weights_path = hf_hub_download(
    repo_id="FemiLanre/GPT-2-Large-Causal-Finetuned-on-ATC", 
    filename="gpt2-large774M-atc-clm-loss.pth",
    #use_auth_token=True
)
print(f"File downloaded to: {model_weights_path}")



# Define the GPT-2 large configuration (must match how the model was trained)
BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

MODEL_CHOICE = "gpt2-large (774M)"
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(MODEL_CONFIGS[MODEL_CHOICE])

# Llama2 config (must match training)
LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # Lower-precision dtype to reduce memory usage
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the custom GPTModel (default LM)
atc_model = GPTModel(BASE_CONFIG)

# Load the saved state dictionary
atc_model.load_state_dict(torch.load(model_weights_path, map_location=device))
atc_model.eval() # Set model to evaluation mode

# Move model to device
atc_model.to(device)

# Load the tokenizer (default LM tokenizer)
tokenizer = tiktoken.get_encoding("gpt2")

# LlamaTokenizer class definition for dynamic loading
class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.encode(text, out_type=int)

    def decode(self, ids):
        return self.tokenizer.decode(ids)


# 2. Load ASR Model (using Whisper for high-fidelity audio input)
# Explicitly load processor and model for Whisper
whisper_model_name = "openai/whisper-tiny"
processor = AutoProcessor.from_pretrained(whisper_model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)

def asr_pipe_custom(audio_path):
    # Load audio data from the provided path
    if not isinstance(audio_path, str) or not os.path.exists(audio_path):
        raise ValueError(f"Invalid audio_path: {audio_path}")

    # Read audio file
    audio_input, sampling_rate = sf.read(audio_path)

    # Convert to mono if stereo
    if audio_input.ndim > 1:
        audio_input = audio_input.mean(axis=1) # Average channels to get mono

    # Resample if necessary (Whisper expects 16kHz)
    if sampling_rate != 16000:
        # Simple resampling for demonstration. For production, consider `torchaudio.transforms.Resample`
        # or `librosa.resample` for higher quality.
        from scipy.signal import resample_poly
        audio_input = resample_poly(audio_input, 16000, sampling_rate)
        sampling_rate = 16000

    # Process audio with Whisper processor
    inputs = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

    # Generate output
    predicted_ids = whisper_model.generate(inputs, max_new_tokens=128)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return {"text": transcription}

def custom_generate_text(model, text_prompt, tokenizer, max_new_tokens=50, context_length=1024, eos_id=50256, device='cpu'):
    model.eval()
    input_ids = text_to_token_ids(text_prompt, tokenizer).to(device)
    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            context_size=context_length,
            eos_id=eos_id
        )
    full_text = token_ids_to_text(output_ids, tokenizer)
    return full_text[len(text_prompt):].strip()

def process_voice_request(audio, asr_model_choice, lm_model_choice):
    print("--- Starting process_voice_request ---")
    if audio is None:
        print("Error: No audio input received.")
        return "Error: No audio input received.", "Error: No audio input received."

    # --- ASR Processing ---
    # Re-initialize ASR model based on selection (or switch if already loaded)
    global whisper_model, processor, whisper_model_name, model_weights_path

    current_whisper_model_name = whisper_model_name # Use a local variable for initial check

    if asr_model_choice == "Whisper Tiny":
        new_whisper_model_name = "openai/whisper-tiny"
    elif asr_model_choice == "Whisper Medium":
        new_whisper_model_name = "openai/whisper-medium"
    elif asr_model_choice == "Whisper Large-v3":
        new_whisper_model_name = "openai/whisper-large-v3"
    else:
        new_whisper_model_name = "openai/whisper-tiny" # Default

    if current_whisper_model_name != new_whisper_model_name:
        print(f"Switching ASR model to {new_whisper_model_name}")
        whisper_model_name = new_whisper_model_name
        del whisper_model
        del processor
        torch.cuda.empty_cache()
        processor = AutoProcessor.from_pretrained(whisper_model_name)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)

    transcription_result = asr_pipe_custom(audio)
    transcription = transcription_result["text"]
    print(f"Transcription completed: {transcription}")

    # --- LM Processing ---
    # Dynamically load or select the LM model and tokenizer
    global atc_model, tokenizer, BASE_CONFIG # Assuming BASE_CONFIG also needs to be dynamic for LM
    current_lm_model = atc_model
    current_tokenizer = tokenizer
    current_base_config = BASE_CONFIG

    #if lm_model_choice == "GPT-2 Large Grammar":
    if lm_model_choice == "GPT-2 Large Causal":
        # Load GPT-2 Large Grammar (already loaded by default, but ensure if switched away and back)
        if not isinstance(current_lm_model, GPTModel) or BASE_CONFIG["emb_dim"] != 1280 or model_weights_path != "/content/drive/MyDrive/ATC_Chatbot/gpt2-large774M-atc-with-grammar-loss.pth":
            print("Switching LM to GPT-2 Large Grammar")
            #model_weights_path = "./gpt2-large774M-atc-with-grammar-loss.pth"
            #model_weights_path = "./gpt2-large774M-atc-clm-loss.pth"
            model_weights_path = hf_hub_download(
                repo_id="FemiLanre/GPT-2-Large-Causal-Finetuned-on-ATC", 
                filename="gpt2-large774M-atc-clm-loss.pth",
                #use_auth_token=True
            )
            current_base_config = MODEL_CONFIGS["gpt2-large (774M)"]
            current_base_config.update(BASE_CONFIG) # Ensure all common keys are present
            del current_lm_model
            torch.cuda.empty_cache()
            current_lm_model = GPTModel(current_base_config)
            current_lm_model.load_state_dict(torch.load(model_weights_path, map_location=device))
            current_lm_model.eval().to(device)
            current_tokenizer = tiktoken.get_encoding("gpt2")

    elif lm_model_choice == "GPT-2 Medium":
        if not isinstance(current_lm_model, GPTModel) or BASE_CONFIG["emb_dim"] != 1024:
            print("Switching LM to GPT-2 Medium (Off-the-shelf)")
            # Example of loading an off-the-shelf GPT-2 Medium
            current_base_config = {
                "vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0, "qkv_bias": True,
                "emb_dim": 1024, "n_layers": 24, "n_heads": 16 # GPT-2 Medium config
            }
            del current_lm_model
            torch.cuda.empty_cache()
            current_lm_model = GPTModel(current_base_config)
            _, params = download_and_load_gpt2(model_size="355M", models_dir="gpt2")
            load_weights_into_gpt(current_lm_model, params)
            current_lm_model.eval().to(device)
            current_tokenizer = tiktoken.get_encoding("gpt2")

    elif lm_model_choice == "Llama-2-7b":
        if not isinstance(current_lm_model, Llama2Model) or LLAMA2_CONFIG_7B["emb_dim"] != 4096:
            print("Switching LM to Llama-2-7b (Off-the-shelf)")
            current_base_config = LLAMA2_CONFIG_7B
            del current_lm_model
            torch.cuda.empty_cache()
            current_lm_model = Llama2Model(current_base_config)

            llama_weights_file = hf_hub_download(
                repo_id="meta-llama/Llama-2-7b-chat",
                filename="consolidated.00.pth",
                local_dir="Llama-2-7b-chat"
            )
            llama_weights = torch.load(llama_weights_file, map_location="cpu", weights_only=True)
            load_llama_weights(current_lm_model, current_base_config, llama_weights)
            del llama_weights
            import gc
            gc.collect()

            current_lm_model.eval().to(device)

            llama_tokenizer_file = hf_hub_download(
                repo_id="meta-llama/Llama-2-7b",
                filename="tokenizer.model",
                local_dir="Llama-2-7b"
            )
            current_tokenizer = LlamaTokenizer(llama_tokenizer_file)
    elif lm_model_choice == "GPT-5.4":
        if not isinstance(current_lm_model, GPTModel) or BASE_CONFIG["emb_dim"] != 1024:
            print("Switching LM to GPT-5.4 (Off-the-shelf)")
            # Example of loading an off-the-shelf GPT-5.4
            del current_lm_model
            torch.cuda.empty_cache()
            # The client automatically looks for an "OPENAI_API_KEY" environment variable
            current_lm_model = None
            current_tokenizer = None
            current_base_config = None


    # Generate suggested UAV response using your model
    n_instruction = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"+
        "### Instruction:\nBelow is a dialogue (communication exchange) between Air Traffic Control (ATC) and UAV Pilot in a controlled airspace. " +
        "Only one part of the conversation is provided i.e. either for the ATC or for the UAV Pilot. " +
        "Write a response that appropriately completes the other side of the conversation.\n"+
        "### ATC: "
    )

    prompt = n_instruction + transcription.upper() + "\n### UAV Pilot:"
    print(f"Prompt prepared: {prompt}")

    # Determine eos_id based on tokenizer type
    if isinstance(current_tokenizer, tiktoken.Encoding):
        eos_id = current_tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
        context_length_lm = current_base_config["context_length"]
    elif isinstance(current_tokenizer, LlamaTokenizer):
        eos_id = current_tokenizer.tokenizer.eos_id # LlamaTokenizer stores it differently
        context_length_lm = current_base_config["context_length"]
    elif lm_model_choice == "GPT-5.4":
        client = OpenAI()
        generated_response = client.responses.create(
            model="gpt-5.4",
            input=prompt,
            reasoning={
                "effort": "medium"  # Options: none, low, medium, high, xhigh
            }
        )
        print(f"Model: {generated_response.model}")
        print(f"Generated response from model: {generated_response.output_text}")
        return transcription, generated_response.output_text
    else:
        raise ValueError("Unsupported tokenizer type.")

    # Use the custom generation function
    generated_response = custom_generate_text(
        current_lm_model,
        prompt,
        current_tokenizer,
        max_new_tokens=50,
        context_length=context_length_lm,
        eos_id=eos_id,
        device=device
    )
    print(f"Generated response from model: {generated_response}")

    # Extract only the response part
    suggested_response = generated_response.split("### UAV Pilot:")[-1].strip()
    print(f"Suggested UAV response extracted: {suggested_response}")
    print("--- Finished process_voice_request ---")

    return transcription, suggested_response

def transmit_and_speak(text_to_speak):
    print(f"Transmitting: {text_to_speak}")
    if text_to_speak:
        tts = gTTS(text_to_speak)
        tts.save('generated_response.wav')
        return 'generated_response.wav' # Return the path to the audio file
    return None

# 3. Build UI
with gr.Blocks(title="Windracers ATC Operator Assistant") as demo:
    gr.Markdown("# \u2708\ufe0f Windracers UAV ATC Communication Assistant")
    gr.Markdown("Speak the incoming ATC request. Review the AI-generated compliant response before transmitting.")

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record ATC Request") # Added "upload" source

    with gr.Row():
        asr_model_selector = gr.Dropdown(
            ["Whisper Tiny", "Whisper Medium", "Whisper Large-v3"],
            label="Select ASR Model",
            value="Whisper Medium"
        )
        lm_model_selector = gr.Dropdown(
            ["GPT-2 Large Causal", "GPT-2 Medium", "Llama-2-7b", "GPT-5.4"],
            label="Select Language Model",
            value="GPT-5.4"
        )

    with gr.Column():
        transcribed_text = gr.Textbox(label="Transcribed ATC Request", interactive=False)
        suggested_output = gr.Textbox(label="Suggested UAV Response (Edit if necessary)", interactive=True)
        audio_output = gr.Audio(label="Generated Voice Response", interactive=False) # Add an audio output component

    with gr.Row():
        btn_process = gr.Button("Generate Suggestion", variant="primary")
        btn_transmit = gr.Button("Approve & Transmit", variant="success")

    # Link functions
    btn_process.click(
        process_voice_request,
        inputs=[audio_input, asr_model_selector, lm_model_selector],
        outputs=[transcribed_text, suggested_output]
    )

    # Update the transmit button to also generate and play speech
    btn_transmit.click(transmit_and_speak, inputs=suggested_output, outputs=audio_output)

#demo.launch(debug=True)
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)