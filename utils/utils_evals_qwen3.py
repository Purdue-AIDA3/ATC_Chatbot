#!/usr/bin/env python3
"""
Comprehensive ATC Model Evaluation Script (sentencepiece + custom Qwen3Model)
Compares CLM vs Grammar-Informed models across SOTA metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import re
import json
import tiktoken
from typing import Dict, List


# ---------------------------------------------------------------------------
# Import your project utilities (same style as training)
# ---------------------------------------------------------------------------
from utils_libs import *
from utils_dataset_qwen import *
from utils_models_qwen import *
from utils_methods import *
from utils_downloads import *


from huggingface_hub import login, hf_hub_download, snapshot_download
import sentencepiece as spm
#from run_QWENfinetune_with_Grammar_ATC import load_weights_into_qwen
#from run_LLAMAfinetune_ATC import *


from pathlib import Path
import os
from safetensors.torch import load_file

from tokenizers import Tokenizer
from transformers import AutoTokenizer



# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_BASE_PATH = "Qwen-2-8b8B-atc-qwen-clm-original-model.pth"      # your CLM checkpoint
MODEL_CLM_PATH = "Qwen-2-8b8B-atc-qwen-clm-loss.pth"      # your CLM checkpoint
MODEL_GRAMMAR_PATH = "Qwen-2-8b8B-atc-qwen-grammar-loss.pth"      # grammar-informed checkpoint
V_ATC_IDS_PATH = "V_ATC_ids_qwen3.pt"                 # 701-token ATC vocab
TEST_JSON_PATH = "test_dialogues.json"         # {"prompts": [...], "references": [...]}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = 64
NUM_SAMPLES = 100                               # for reference-based metrics

# LLAMA2 config (must match training)
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

# ============================================================================
# LOADING TOKENIZER AND MODELS (MATCH TRAINING)
# ============================================================================
def load_tokenizer():
    # Load your tokenizer (or pass it in)
    # login to huggngface to access model Qwena tokenizer
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        access_token = config["HF_ACCESS_TOKEN"]

    login(token=access_token)

    # tokenizer file path
    tokenizer_file_path = f"Qwen3-8B/tokenizer.json"
    repo_id=f"Qwen/Qwen3-8B"

    # Get Qwen3tokenizer
    tokenizer_file = hf_hub_download(
        repo_id=f"Qwen/Qwen3-8B",
        filename="tokenizer.json",
        local_dir="Qwen-3-8B"
    )

    # define a class to easily access tokenizer
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

    tokenizer = Qwen3Tokenizer(
                tokenizer_file_path=tokenizer_file_path,
                repo_id=repo_id,
                apply_chat_template= False, #True
                add_generation_prompt=True,
                add_thinking= False #USE_REASONING_MODEL
            )

    return tokenizer


def build_base_qwen3_model():
    """Construct QWEN Model with pretrained QWEN‑3 weights (same as training)."""
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
    model.to(DEVICE)

    return model

def load_models_and_vocab():
    """
    Load CLM and Grammar models using the same LLAMAModel + state_dict pattern
    as in training, and load V_ATC_ids.
    """
    tokenizer = load_tokenizer()

    # Base initialized from OpenAI GPT-2 weights
    model_base = build_base_qwen3_model()
    model_clm = build_base_qwen3_model()
    model_grammar = build_base_qwen3_model()

    # Load fine-tuned weights
    model_base.load_state_dict(torch.load(MODEL_BASE_PATH, map_location=DEVICE))
    model_clm.load_state_dict(torch.load(MODEL_CLM_PATH, map_location=DEVICE))
    model_grammar.load_state_dict(torch.load(MODEL_GRAMMAR_PATH, map_location=DEVICE))

    model_base.eval()
    model_clm.eval()
    model_grammar.eval()

    V_ATC_ids = torch.load(V_ATC_IDS_PATH, map_location=DEVICE).to(DEVICE)

    return tokenizer, model_base, model_clm, model_grammar, V_ATC_ids

# ============================================================================
# GENERATION (USING SENTENCEPIECE FOR ENCODING/DECODING)
# ============================================================================

def generate_with_sentence_piece(model, prompt: str, tokenizer, max_new=50, device=DEVICE):
    """
    Generate continuation using LLAMA Model + sentencepiece, mirroring training generation:
    - text_to_token_ids / token_ids_to_text style.
    """
    model.eval()
    # Encode prompt
    input_ids = text_to_token_ids(prompt, tokenizer).to(device)  # uses your utils_methods
    # Generate token ids
    with torch.no_grad():
        out_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=max_new,
            context_size=QWEN3_CONFIG["context_length"],
            eos_id=tokenizer.eos_token_id,
        )
    # Decode full text, then slice off prompt
    full_text = token_ids_to_text(out_ids, tokenizer)
    return full_text[len(prompt):].strip()

# ============================================================================
# COMMAND EXTRACTION (ATC-SPECIFIC)
# ============================================================================

def extract_commands(text: str) -> List[Dict]:
    """Extract structured ATC commands via regex."""
    patterns = {
        "callsign": r"\b([A-Z]{1,3}\d{1,4}[A-Z]{0,2})\b",
        "altitude": r"(?:FL|flight level)\s*(\d{2,3})",
        "heading": r"(?:HDG|heading)\s+(\d{2,3})",
        "speed": r"(\d{2,3})\s*knots?",
        "runway": r"(?:RWY|runway)\s*(\d{1,2}[LRC]?)",
        "frequency": r"\b(\d{3}\.\d{1,3})\b",
    }
    cmds = []
    for slot, pat in patterns.items():
        matches = re.findall(pat, text, re.IGNORECASE)
        cmds.extend({"slot": slot, "value": m} for m in matches)
    return cmds

def command_extraction_f1(gold_commands, pred_commands) -> float:
    """Slot-level F1 score."""
    gold_slots = set((c["slot"], str(c["value"]).upper()) for c in gold_commands)
    pred_slots = set((c["slot"], str(c["value"]).upper()) for c in pred_commands)

    tp = len(gold_slots & pred_slots)
    fp = len(pred_slots - gold_slots)
    fn = len(gold_slots - pred_slots)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def atc_token_density(text, tokenizer, v_atc_ids):
    ids = torch.tensor(tokenizer.encode(text)).to(DEVICE)
    atc_count = sum(1 for i in ids if i in v_atc_ids)
    return atc_count / len(ids) if len(ids) > 0 else 0
# ============================================================================
# REFERENCE-BASED METRICS
# ============================================================================

def compute_reference_metrics(gens: List[str], refs: List[str]) -> Dict[str, float]:
    """BERTScore / ROUGE-L / embedding-based semantic similarity."""
    # BERTScore
    P, R, F1 = bertscore(gens, refs, model_type="roberta-large", lang="en", verbose=False)
    bert_f1 = F1.mean().item()

    # ROUGE-L
    #scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    #rouge_vals = [scorer.score(r, g).rougeL.fmeasure for r, g in zip(refs, gens)]
    rouge_l = 0.0 #float(np.mean(rouge_vals))

    # Semantic similarity
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_embs = sentence_model.encode(refs, convert_to_tensor=True)
    gen_embs = sentence_model.encode(gens, convert_to_tensor=True)
    sim = util.cos_sim(ref_embs, gen_embs).diagonal().mean().item()


    return {
        "bertscore": bert_f1,
        "rouge_l": rouge_l,
        "semantic_similarity": sim,
    }

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    print("🚀 Starting ATC model evaluation (sentencepiece + LLAMAModel)...")

    # Load tokenizer, models, ATC vocab
    tokenizer, model_base, model_clm, model_grammar, V_ATC_ids = load_models_and_vocab()

    # Load test prompts + references
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_prompts = test_data["prompts"][:NUM_SAMPLES]
    test_refs = test_data["references"][:NUM_SAMPLES]
    print(f"Evaluating on {len(test_prompts)} samples")

    # Generate outputs
    print("🤖 Generating BASE model outputs...")
    gens_base = [
        generate_with_sentence_piece(model_base, p, tokenizer, MAX_NEW_TOKENS, DEVICE)
        for p in test_prompts
    ]

    print("🤖 Generating CLM outputs...")
    gens_clm = [
        generate_with_sentence_piece(model_clm, p, tokenizer, MAX_NEW_TOKENS, DEVICE)
        for p in test_prompts
    ]

    print("🤖 Generating Grammar-informed outputs...")
    gens_grammar = [
        generate_with_sentence_piece(model_grammar, p, tokenizer, MAX_NEW_TOKENS, DEVICE)
        for p in test_prompts
    ]

    # Reference-based metrics
    print("📈 Computing reference-based metrics...")
    ref_base = compute_reference_metrics(gens_base, test_refs)
    ref_clm = compute_reference_metrics(gens_clm, test_refs)
    ref_grammar = compute_reference_metrics(gens_grammar, test_refs)

    # Command extraction F1
    print("🎯 Computing ATC command extraction F1...")
    cmd_f1_base = float(
        np.mean([
            command_extraction_f1(extract_commands(r), extract_commands(g))
            for r, g in zip(test_refs, gens_base)
        ])
    )

    cmd_f1_clm = float(
        np.mean([
            command_extraction_f1(extract_commands(r), extract_commands(g))
            for r, g in zip(test_refs, gens_clm)
        ])
    )
    cmd_f1_grammar = float(
        np.mean([
            command_extraction_f1(extract_commands(r), extract_commands(g))
            for r, g in zip(test_refs, gens_grammar)
        ])
    )

    # Compute valid_token density
    token_density_base = sum([atc_token_density(text, tokenizer, V_ATC_ids) for text in gens_base]) / len(gens_base)
    token_density_clm = sum([atc_token_density(text, tokenizer, V_ATC_ids) for text in gens_clm]) / len(gens_clm)
    token_density_grammar = sum([atc_token_density(text, tokenizer, V_ATC_ids) for text in gens_grammar]) / len(gens_grammar)

    # Tabulate
    results = {
        "BASE": {
            "bertscore": ref_base["bertscore"],
            "rouge_l": ref_base["rouge_l"],
            "sem_sim": ref_base["semantic_similarity"],
            "cmd_f1": cmd_f1_base,
            "token_density": token_density_base
        },
        "CLM": {
            "bertscore": ref_clm["bertscore"],
            "rouge_l": ref_clm["rouge_l"],
            "sem_sim": ref_clm["semantic_similarity"],
            "cmd_f1": cmd_f1_clm,
            "token_density": token_density_clm
        },
        "Grammar": {
            "bertscore": ref_grammar["bertscore"],
            "rouge_l": ref_grammar["rouge_l"],
            "sem_sim": ref_grammar["semantic_similarity"],
            "cmd_f1": cmd_f1_grammar,
            "token_density": token_density_grammar
        },
    }

    df = pd.DataFrame(results).T.round(3)
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    print(df.to_string())

    df.to_csv("atc_evaluation_results_sentence_piece_qwenmodel.csv")
    print("\n💾 Saved to atc_evaluation_results_sentence_piece_qwenmodel.csv")

    print("\n🎯 KEY TAKEAWAYS:")
    for metric in df.columns:
        base_val = df.loc["BASE", metric]
        clm_val = df.loc["CLM", metric]
        gram_val = df.loc["Grammar", metric]  
        #winner = "Grammar" if gram_val > clm_val and gram_val > base_val else "CLM"
        if gram_val > clm_val and gram_val > base_val:
            winner = "Grammar"
            delta = gram_val - max(clm_val, base_val)
        elif clm_val > gram_val and clm_val > base_val:
            winner = "CLM"
            delta = clm_val - max(gram_val, base_val)
        else:
            winner = "BASE"
            delta = base_val - max(clm_val, gram_val)
        #winner = "Grammar" if gram_val > clm_val and gram_val > base_val else "CLM"
        #delta = gram_val - clm_val
        print(f"  {metric:15}: {winner} ({delta:+.3f})")

if __name__ == "__main__":
    main()
