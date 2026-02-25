#!/usr/bin/env python3
"""
Comprehensive ATC Model Evaluation Script (tiktoken + custom GPTModel)
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
from utils_dataset import *
from utils_models import *
from utils_methods import *
from utils_downloads import *

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_CLM_PATH = "gpt2-large774M-atc-clm-loss.pth"      # your CLM checkpoint
MODEL_GRAMMAR_PATH = "gpt2-large774M-atc-with-grammar-loss.pth"      # grammar-informed checkpoint
V_ATC_IDS_PATH = "V_ATC_ids.pt"                 # 701-token ATC vocab
TEST_JSON_PATH = "test_dialogues.json"         # {"prompts": [...], "references": [...]}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = 64
NUM_SAMPLES = 100                               # for reference-based metrics

# GPT2 config (must match training)
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
MODEL_CHOICE = "gpt2-large (774M)"
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(MODEL_CONFIGS[MODEL_CHOICE])

# ============================================================================
# LOADING TOKENIZER AND MODELS (MATCH TRAINING)
# ============================================================================

def load_tokenizer():
    """Use the same tiktoken GPT‑2 encoding as in training."""
    return tiktoken.get_encoding("gpt2")

def build_base_gpt_model():
    """Construct GPTModel with pretrained GPT‑2 weights (same as training)."""
    model_size = MODEL_CHOICE.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(DEVICE)
    return model

def load_models_and_vocab():
    """
    Load CLM and Grammar models using the same GPTModel + state_dict pattern
    as in training, and load V_ATC_ids.
    """
    tokenizer = load_tokenizer()

    # Base initialized from OpenAI GPT-2 weights
    model_clm = build_base_gpt_model()
    model_grammar = build_base_gpt_model()

    # Load fine-tuned weights
    model_clm.load_state_dict(torch.load(MODEL_CLM_PATH, map_location=DEVICE))
    model_grammar.load_state_dict(torch.load(MODEL_GRAMMAR_PATH, map_location=DEVICE))

    model_clm.eval()
    model_grammar.eval()

    V_ATC_ids = torch.load(V_ATC_IDS_PATH, map_location=DEVICE).to(DEVICE)

    return tokenizer, model_clm, model_grammar, V_ATC_ids

# ============================================================================
# GENERATION (USING TIKTOKEN FOR ENCODING/DECODING)
# ============================================================================

def generate_with_tiktoken(model, prompt: str, tokenizer, max_new=50, device=DEVICE):
    """
    Generate continuation using GPTModel + tiktoken, mirroring training generation:
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
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,  # <|endoftext|>
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
    print("🚀 Starting ATC model evaluation (tiktoken + GPTModel)...")

    # Load tokenizer, models, ATC vocab
    tokenizer, model_clm, model_grammar, V_ATC_ids = load_models_and_vocab()

    # Load test prompts + references
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_prompts = test_data["prompts"][:NUM_SAMPLES]
    test_refs = test_data["references"][:NUM_SAMPLES]
    print(f"Evaluating on {len(test_prompts)} samples")

    # Generate outputs
    print("🤖 Generating CLM outputs...")
    gens_clm = [
        generate_with_tiktoken(model_clm, p, tokenizer, MAX_NEW_TOKENS, DEVICE)
        for p in test_prompts
    ]

    print("🤖 Generating Grammar-informed outputs...")
    gens_grammar = [
        generate_with_tiktoken(model_grammar, p, tokenizer, MAX_NEW_TOKENS, DEVICE)
        for p in test_prompts
    ]

    # Reference-based metrics
    print("📈 Computing reference-based metrics...")
    ref_clm = compute_reference_metrics(gens_clm, test_refs)
    ref_grammar = compute_reference_metrics(gens_grammar, test_refs)

    # Command extraction F1
    print("🎯 Computing ATC command extraction F1...")
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

    # Tabulate
    results = {
        "CLM": {
            "bertscore": ref_clm["bertscore"],
            "rouge_l": ref_clm["rouge_l"],
            "sem_sim": ref_clm["semantic_similarity"],
            "cmd_f1": cmd_f1_clm,
        },
        "Grammar": {
            "bertscore": ref_grammar["bertscore"],
            "rouge_l": ref_grammar["rouge_l"],
            "sem_sim": ref_grammar["semantic_similarity"],
            "cmd_f1": cmd_f1_grammar,
        },
    }

    df = pd.DataFrame(results).T.round(3)
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    print(df.to_string())

    df.to_csv("atc_evaluation_results_tiktoken_gptmodel.csv")
    print("\n💾 Saved to atc_evaluation_results_tiktoken_gptmodel.csv")

    print("\n🎯 KEY TAKEAWAYS:")
    for metric in df.columns:
        clm_val = df.loc["CLM", metric]
        gram_val = df.loc["Grammar", metric]
        winner = "Grammar" if gram_val > clm_val else "CLM"
        delta = gram_val - clm_val
        print(f"  {metric:15}: {winner} ({delta:+.3f})")

if __name__ == "__main__":
    main()
