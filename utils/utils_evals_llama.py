#!/usr/bin/env python3
"""
Comprehensive ATC Model Evaluation Script (sentencepiece + custom Llama2Model)
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
from utils_dataset_llama import *
from utils_models_llama2 import *
from utils_methods import *
from utils_downloads import *


from huggingface_hub import login, hf_hub_download
import sentencepiece as spm
from run_LLAMAfinetune_with_Grammar_ATC import assign, permute, load_weights_into_llama

#from run_LLAMAfinetune_ATC import *
# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_BASE_PATH = "Llama-2-7b7B-atc-llama2-clm-original-model.pth"      # your CLM checkpoint
MODEL_CLM_PATH = "Llama-2-7b7B-atc-Llama-clm-loss.pth"      # your CLM checkpoint
MODEL_GRAMMAR_PATH = "Llama-2-7b7B-atc-Llama-grammar-loss.pth"      # grammar-informed checkpoint
V_ATC_IDS_PATH = "V_ATC_ids_llama2.pt"                 # 701-token ATC vocab
TEST_JSON_PATH = "test_dialogues.json"         # {"prompts": [...], "references": [...]}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = 64
NUM_SAMPLES = 100                               # for reference-based metrics

# LLAMA2 config (must match training)
LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}
MODEL_CHOICE = "Llama-2-7b (7B)"
MODEL_CONFIGS = {
    "Llama-2-7b (7B)": {"emb_dim": 4096, "n_layers": 32, "n_heads": 32},
    "Llama2-7b-chat (7B)": {"emb_dim": 4096, "n_layers": 32, "n_heads": 32},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

LLAMA2_CONFIG_7B.update(MODEL_CONFIGS[MODEL_CHOICE])

# ============================================================================
# LOADING TOKENIZER AND MODELS (MATCH TRAINING)
# ============================================================================

def load_tokenizer():
    """Use the same LLAMA‑2 encoding as in training."""

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

    return tokenizer

def build_base_llama_model():
    weights_file = hf_hub_download(
      repo_id="meta-llama/Llama-2-7b-chat",
      filename="consolidated.00.pth",
      local_dir="Llama-2-7b-chat"
    )

    # Force load to CPU to save GPU space
    weights = torch.load(weights_file, map_location="cpu", weights_only=True)

    model = Llama2Model(LLAMA2_CONFIG_7B)
    load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)

    # Delete the weights dictionary IMMEDIATELY to free 13GB of RAM
    del weights
    import gc
    gc.collect()

    model.to(DEVICE) # Now move the finished model to GPU
    return model


def load_models_and_vocab():
    """
    Load CLM and Grammar models using the same LLAMAModel + state_dict pattern
    as in training, and load V_ATC_ids.
    """
    tokenizer = load_tokenizer()

    # Base initialized from OpenAI GPT-2 weights
    model_base = build_base_llama_model()
    model_clm = build_base_llama_model()
    model_grammar = build_base_llama_model()

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
            context_size=LLAMA2_CONFIG_7B["context_length"],
            eos_id=2,
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
    # --- ADD THESE TWO LINES TO PREVENT THE CRASH ---
    gens = [g.strip() if (isinstance(g, str) and g.strip()) else "..." for g in gens]
    refs = [r.strip() if (isinstance(r, str) and r.strip()) else "..." for r in refs]
    # -----------------------------------------------

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

    # Load tokenizer
    tokenizer = load_tokenizer()
    V_ATC_ids = torch.load(V_ATC_IDS_PATH, map_location=DEVICE).to(DEVICE)

    # LOAD TEST PROMPTS
    # Load test prompts + references
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_prompts = test_data["prompts"][:NUM_SAMPLES]
    test_refs = test_data["references"][:NUM_SAMPLES]
    print(f"Evaluating on {len(test_prompts)} samples")



    # 1. Load and Evaluate BASE model
    print("🤖 Loading CLM model...")
    model = build_base_llama_model()
    model.load_state_dict(torch.load(MODEL_BASE_PATH, map_location=DEVICE))
    gens_base = [generate_with_sentence_piece(model, p, tokenizer) for p in test_prompts]


    # CLEAR MEMORY COMPLETELY
    del model
    torch.cuda.empty_cache()


    # 1. Load and Evaluate CLM
    print("🤖 Loading CLM model...")
    model = build_base_llama_model()
    model.load_state_dict(torch.load(MODEL_CLM_PATH, map_location=DEVICE))
    gens_clm = [generate_with_sentence_piece(model, p, tokenizer) for p in test_prompts]

    # CLEAR MEMORY COMPLETELY
    del model
    torch.cuda.empty_cache()

    # 2. Load and Evaluate Grammar
    print("🤖 Loading Grammar model...")
    model = build_base_llama_model()
    model.load_state_dict(torch.load(MODEL_GRAMMAR_PATH, map_location=DEVICE))
    gens_grammar = [generate_with_sentence_piece(model, p, tokenizer) for p in test_prompts]

    # 3. Final metrics
    # Now you only have the text strings in RAM, no heavy models on GPU
    #metrics_clm = compute_reference_metrics(gens_clm, test_refs)
    #metrics_grammar = compute_reference_metrics(gens_grammar, test_refs)



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

    df.to_csv("atc_evaluation_results_sentence_piece_llamamodel.csv")
    print("\n💾 Saved to atc_evaluation_results_sentence_piece_llamamodel.csv")

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
