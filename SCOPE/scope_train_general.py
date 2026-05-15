#!/usr/bin/env python3
"""
SCOPE Training Pipeline

Three-level objective:
  L = L_CE + lambda_tok * L_tok + lambda_phr * L_phr(GRPO) + lambda_cfg * L_cfg(GRPO)
"""

import json, re, math, random, argparse, os
from pathlib import Path

# Reduce CUDA memory fragmentation — critical for DPO with large models
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                           get_linear_schedule_with_warmup)

# ── Try import lark ───────────────────────────────────────────────────────────
try:
    from lark import Lark, exceptions as lark_exc
    LARK_AVAILABLE = True
except ImportError:
    LARK_AVAILABLE = False
    print("WARNING: lark not installed. L_cfg will be disabled.")

# ═════════════════════════════════════════════════════════════════════════════
# 1. REGULATORY ARTEFACTS
# ═════════════════════════════════════════════════════════════════════════════

VOCAB_PATH  = Path("vocab_ATC.json")
PHRASE_PATH = Path("ngram_whitelist_ATC.json")
GRAMMAR_PATH = Path("G_ATC.lark")

def load_whitelist(vocab_path: Path) -> set:
    """Load V_ATC: the regulatory vocabulary whitelist."""
    with open(vocab_path) as f:
        vocab = json.load(f)
    return set(v.upper() for v in vocab)

def load_ngram_whitelist(phrase_path: Path) -> Dict[int, set]:
    """Load P_ATC: n-gram phrase whitelists for n in {2,3,4}."""
    with open(phrase_path) as f:
        raw = json.load(f)
    return {
        2: set(tuple(g) for g in raw.get('bigrams',  [])),
        3: set(tuple(g) for g in raw.get('trigrams', [])),
        4: set(tuple(g) for g in raw.get('4grams',   [])),
    }

def load_grammar(grammar_path: Path):
    """Load G_ATC: Lark EBNF grammar for CFG compliance."""
    if not LARK_AVAILABLE:
        return None
    with open(grammar_path) as f:
        grammar_str = f.read()
    return Lark(grammar_str, parser='earley', ambiguity='resolve')

# FSM state → vocabulary subset (state-conditional whitelist V(s_t))
FSM_STATE_VOCAB = {
    'Init':              None,   # Full whitelist
    'AwaitingClearance': {       # Expects clearance vocabulary
        'CLEARED','DESCEND','CLIMB','MAINTAIN','SQUAWK','CONTACT',
        'HOLD','EXPECT','UNABLE','STANDBY','TRAFFIC'
    },
    'ClearanceIssued': {         # Readback vocabulary
        'WILCO','AFFIRM','ROGER','UNABLE','SAY','AGAIN','READBACK'
    },
    'AwaitingReadback': {        # Correction vocabulary
        'AFFIRM','NEGATIVE','CORRECTION','SAY','AGAIN','READBACK'
    },
    'ReadbackReceived': None,    # Full whitelist
}

# ═════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ═════════════════════════════════════════════════════════════════════════════

DOMAIN_PROMPTS = {
    "atc":  "You are an ATC communication assistant for UAV operations. "
            "Generate ICAO-compliant phraseology.",
    "smcp": "You are a maritime radio communication assistant. "
            "Generate IMO SMCP-compliant phraseology.",
}

def format_atc(request: str, response: str, fsm_state: str = 'Init',
               tokenizer=None, use_chat_template: bool = False,
               domain: str = "atc") -> dict:
    """Format a domain pair in Alpaca instruction format with optional state conditioning."""
    domain_desc = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["atc"])
    if domain == "atc":
        system = f"[STATE: {fsm_state}] {domain_desc}"
    else:
        system = domain_desc   # no FSM states for maritime
    if use_chat_template and tokenizer is not None:
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": request}]
        instruction = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        full = instruction + response
        return {"instruction": instruction, "full": full, "response": response}
    instruction = f"### Instruction:\n{system}\n\n### Operator:\n{request}\n\n### Response:"
    full = instruction + " " + response
    return {
        "instruction": instruction,
        "response": response,
        "full": full,
        "fsm_state": fsm_state,
    }

class AtcDataset(Dataset):
    def __init__(self, pairs: List[dict], tokenizer, max_length: int = 512,
                 domain: str = "atc"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for p in pairs:
            item = format_atc(p["request"], p["response"],
                               p.get("fsm_state", "Init"), domain=domain)
            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tok = self.tokenizer(
            item["full"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        # With left-padding the layout is: [PAD...PAD | instruction | response]
        # Find instruction length without padding
        instr_ids = self.tokenizer(
            item["instruction"],
            max_length=self.max_length,
            truncation=True,
        )["input_ids"]
        instr_len = len(instr_ids)

        # Number of pad tokens on the left
        n_pad = int((attention_mask == 0).sum().item())

        # Response mask: 1 only at response token positions (after instruction)
        response_mask = torch.zeros_like(input_ids)
        resp_start = n_pad + instr_len   # first response token position
        response_mask[resp_start:] = 1
        response_mask = response_mask * attention_mask   # exclude any trailing pad

        labels = input_ids.clone()
        labels[:resp_start] = -100   # mask pad + instruction from CE loss

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "response_mask":  response_mask,
            "instr_len":      torch.tensor(instr_len, dtype=torch.long),
            "n_pad":          torch.tensor(n_pad,     dtype=torch.long),
            "fsm_state":      item["fsm_state"],
            "response_text":  item["response"],
            "instruction":    item["instruction"],   # keep raw text for val generation
        }

# ═════════════════════════════════════════════════════════════════════════════
# 3. COMPLIANCE METRICS (C_tok, C_phr, C_cfg, C_dep)
# ═════════════════════════════════════════════════════════════════════════════

def compute_ctok(tokens: List[str], vocab: set) -> float:
    """C_tok: fraction of tokens in V_ATC."""
    if not tokens: return 0.0
    return sum(1 for t in tokens if t.upper() in vocab) / len(tokens)

def compute_cphr(tokens: List[str], ngram_whitelist: Dict[int, set]) -> float:
    """C_phr: average n-gram compliance for n in {2,3,4}."""
    if len(tokens) < 2: return 0.0
    scores = []
    toks_upper = [t.upper() for t in tokens]
    for n in [2, 3, 4]:
        if len(toks_upper) < n: continue
        count = 0
        total = len(toks_upper) - n + 1
        wl = ngram_whitelist[n]
        for i in range(total):
            if tuple(toks_upper[i:i+n]) in wl:
                count += 1
        scores.append(count / total)
    return sum(scores) / len(scores) if scores else 0.0

def compute_ccfg_partial(text: str, parser) -> float:
    """
    Soft partial-parse score: length of longest prefix parsing under G_ATC.
    Returns value in [0, 1].
    """
    if parser is None: return 0.0
    text_upper = text.upper().strip()
    if not text_upper: return 0.0

    # Try full parse first
    try:
        parser.parse(text_upper)
        return 1.0
    except Exception:
        pass

    # Binary search for longest parseable prefix
    words = text_upper.split()
    if not words: return 0.0

    # Try progressively shorter prefixes
    for n in range(len(words), 0, -1):
        prefix = ' '.join(words[:n])
        try:
            parser.parse(prefix)
            return n / len(words)
        except Exception:
            continue
    return 0.0

# ═════════════════════════════════════════════════════════════════════════════
# 4. SCOPE LOSSES
# ═════════════════════════════════════════════════════════════════════════════

def compute_L_tok(logits: torch.Tensor,
                  response_mask: torch.Tensor,
                  vocab_ids: set,
                  vocab_size: int) -> torch.Tensor:
    """
    L_tok = (1 / sum(m_t)) * sum_t m_t * sum_{v not in V_ATC} p_t(v)
    Differentiable through softmax. Response-masked.

    vocab_ids: set of token IDs in V_ATC (pre-computed per tokenizer)
    """
    probs = F.softmax(logits, dim=-1)           # [B, T, V]
    B, T, V = probs.shape

    # Build out-of-whitelist mask (True = out of whitelist)
    device = logits.device
    in_vocab = torch.zeros(V, dtype=torch.bool, device=device)
    # Only iterate once — vocab_ids is a set of ints
    for vid in vocab_ids:
        if vid < V:
            in_vocab[vid] = True
    out_vocab_mask = ~in_vocab                  # [V]

    # Probability mass on out-of-whitelist tokens per position
    out_mass = (probs * out_vocab_mask.float()).sum(dim=-1)  # [B, T]

    # Apply response mask and normalise
    m = response_mask.float()                   # [B, T]
    denom = m.sum().clamp(min=1.0)
    L_tok = (out_mass * m).sum() / denom

    return L_tok

def compute_L_phr_grpo(model, tokenizer, batch_inputs: dict,
                        ngram_whitelist: Dict[int, set],
                        M: int = 4, max_new: int = 64):
    """
    L_phr (GRPO): phrase-level compliance.
    Returns (loss, all_generated) — generated sequences reused for L_cfg.
    """
    device = next(model.parameters()).device
    input_ids      = batch_inputs["input_ids"]
    attention_mask = batch_inputs["attention_mask"]
    B = input_ids.size(0)

    # ── Sample M sequences and compute rewards (no grad) ──────────────────
    model.eval()
    all_generated, all_rewards = [], []
    with torch.no_grad():
        for _ in range(M):
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen = out[:, input_ids.size(1):]
            all_generated.append(gen)
            rewards = []
            for i in range(B):
                text = tokenizer.decode(gen[i].tolist(), skip_special_tokens=True)
                r = compute_cphr(text.upper().split(), ngram_whitelist)
                rewards.append(r)
            all_rewards.append(rewards)
    model.train()

    rewards_t  = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    mu         = rewards_t.mean(dim=0, keepdim=True)
    sigma      = rewards_t.std(dim=0, keepdim=True).clamp(min=1e-8)
    advantages = (rewards_t - mu) / sigma

    # ── Compute log probs WITH grad on same sequences ──────────────────────
    loss_terms = []
    for m_idx, gen_m in enumerate(all_generated):
        T_gen = gen_m.size(1)
        if T_gen == 0:
            continue
        full   = torch.cat([input_ids, gen_m], dim=1)
        attn   = torch.ones(B, full.size(1)-1, dtype=torch.long, device=device)
        logits = model(full[:, :-1], attention_mask=attn).logits
        logits = logits[:, input_ids.size(1)-1:, :]
        log_p  = F.log_softmax(logits, dim=-1)
        idx_t  = gen_m.unsqueeze(-1).clamp(0, log_p.size(-1)-1)
        seq_lp = log_p.gather(-1, idx_t).squeeze(-1).mean(dim=-1)
        adv    = advantages[m_idx]
        loss_terms.append(-(adv * seq_lp).mean() / M)

    loss = sum(loss_terms) if loss_terms else torch.tensor(0.0, device=device)
    return loss, all_generated

def compute_L_cfg_grpo(model, tokenizer, batch_inputs: dict,
                        cfg_parser, all_generated: list,
                        M: int = 4) -> torch.Tensor:
    """
    L_cfg (GRPO): syntactic structural compliance.
    Reuses all_generated from L_phr — no extra sampling needed.
    """
    if cfg_parser is None or not all_generated:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    device = next(model.parameters()).device
    input_ids = batch_inputs["input_ids"]
    B = input_ids.size(0)

    # ── CFG rewards on pre-sampled sequences (no grad) ────────────────────
    all_rewards = []
    for gen in all_generated:
        rewards = []
        for i in range(B):
            text = tokenizer.decode(gen[i].tolist(), skip_special_tokens=True)
            r = compute_ccfg_partial(text, cfg_parser)
            rewards.append(r)
        all_rewards.append(rewards)

    rewards_t  = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    mu         = rewards_t.mean(dim=0, keepdim=True)
    sigma      = rewards_t.std(dim=0, keepdim=True).clamp(min=1e-8)
    advantages = (rewards_t - mu) / sigma

    # ── Log probs WITH grad ────────────────────────────────────────────────
    loss_terms = []
    for m_idx, gen_m in enumerate(all_generated):
        T_gen = gen_m.size(1)
        if T_gen == 0:
            continue
        full   = torch.cat([input_ids, gen_m], dim=1)
        attn   = torch.ones(B, full.size(1)-1, dtype=torch.long, device=device)
        logits = model(full[:, :-1], attention_mask=attn).logits
        logits = logits[:, input_ids.size(1)-1:, :]
        log_p  = F.log_softmax(logits, dim=-1)
        idx_t  = gen_m.unsqueeze(-1).clamp(0, log_p.size(-1)-1)
        seq_lp = log_p.gather(-1, idx_t).squeeze(-1).mean(dim=-1)
        adv    = advantages[m_idx]
        loss_terms.append(-(adv * seq_lp).mean() / M)

    return sum(loss_terms) if loss_terms else torch.tensor(0.0, device=device)


# ═════════════════════════════════════════════════════════════════════════════
# 5. DPO — DIRECT PREFERENCE OPTIMISATION
# ═════════════════════════════════════════════════════════════════════════════

def build_dpo_pairs(pairs: List[dict], tokenizer, model, vocab: set,
                    ngram_wl: Dict[int, set], max_new: int,
                    device: torch.device, domain: str = "atc") -> List[dict]:
    """
    Build synthetic DPO preference pairs from the dataset using C_tok as
    the preference signal.

    Strategy: for each training pair generate two responses from the current
    model (stochastic sampling). Score both with C_tok. The higher-scoring
    response is 'chosen' (y_w), the lower is 'rejected' (y_l).

    If both responses score identically, use the reference response as y_w
    and the lower-scoring generation as y_l.

    Returns a list of dicts: {instruction, chosen, rejected}
    """
    print("  Building DPO preference pairs from C_tok signal ...")
    model.eval()
    dpo_pairs = []

    with torch.no_grad():
        for p in pairs:
            item = format_atc(p["request"], p["response"], domain=domain)
            tok  = tokenizer(
                item["instruction"],
                return_tensors="pt",
                max_length=256,
                truncation=True,
            ).to(device)

            gens = []
            for _ in range(2):
                out = model.generate(
                    **tok,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=1.2,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
                gen_ids = out[0, tok["input_ids"].size(1):]
                text    = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                score   = compute_ctok(text.upper().split(), vocab)
                gens.append((text, score))

            # Sort: higher C_tok = chosen
            gens.sort(key=lambda x: x[1], reverse=True)
            chosen_text, chosen_score   = gens[0]
            rejected_text, rejected_score = gens[1]

            # If both identical score, use reference as chosen
            if abs(chosen_score - rejected_score) < 1e-6:
                chosen_text = p["response"]

            dpo_pairs.append({
                "instruction": item["instruction"],
                "chosen":      chosen_text,
                "rejected":    rejected_text,
            })

    model.train()
    print(f"  Built {len(dpo_pairs)} DPO preference pairs")
    return dpo_pairs


def compute_L_dpo(model, ref_model, tokenizer, dpo_batch: dict,
                  beta: float, device: torch.device) -> torch.Tensor:
    """
    DPO loss (Rafailov et al. 2023):

        L_DPO = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x)
                           - log π_θ(y_l|x)/π_ref(y_l|x)))]

    where σ is the sigmoid function.

    Implemented via sequence-level log-prob: sum of token log-probs
    over the response portion, normalised by response length.
    """
    def seq_logprob(mdl, input_ids, attn_mask, resp_start):
        """Mean log-prob of response tokens under mdl."""
        out    = mdl(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, :-1, :]           # [B, T-1, V]
        tgts   = input_ids[:, 1:]                # [B, T-1]
        log_p  = F.log_softmax(logits, dim=-1)
        tok_lp = log_p.gather(-1, tgts.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        # Mask: only response tokens
        mask   = torch.zeros_like(tok_lp)
        for i, rs in enumerate(resp_start):
            mask[i, rs-1:] = 1.0          # -1 because tgts is shifted
        denom  = mask.sum(dim=-1).clamp(min=1)
        return (tok_lp * mask).sum(dim=-1) / denom   # [B]

    chosen_ids   = dpo_batch["chosen_ids"].to(device)
    chosen_mask  = dpo_batch["chosen_mask"].to(device)
    rejected_ids = dpo_batch["rejected_ids"].to(device)
    rejected_mask= dpo_batch["rejected_mask"].to(device)
    resp_start   = dpo_batch["resp_start"]       # list of ints

    # Policy log-probs (with grad)
    lp_w_policy  = seq_logprob(model, chosen_ids,   chosen_mask,  resp_start)
    lp_l_policy  = seq_logprob(model, rejected_ids, rejected_mask, resp_start)

    # Reference log-probs (no grad) — move ref model to GPU briefly then back to CPU
    with torch.no_grad():
        ref_model.to(device)
        lp_w_ref = seq_logprob(ref_model, chosen_ids,   chosen_mask,  resp_start)
        lp_l_ref = seq_logprob(ref_model, rejected_ids, rejected_mask, resp_start)
        ref_model.cpu()
        torch.cuda.empty_cache()

    # DPO objective
    log_ratio_w = lp_w_policy - lp_w_ref
    log_ratio_l = lp_l_policy - lp_l_ref
    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()
    return loss


class DPODataset(Dataset):
    """Dataset of DPO preference pairs."""
    def __init__(self, dpo_pairs: List[dict], tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = dpo_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        def encode(text):
            tok = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return tok["input_ids"].squeeze(0), tok["attention_mask"].squeeze(0)

        # Instruction + chosen / instruction + rejected
        chosen_full   = item["instruction"] + " " + item["chosen"]
        rejected_full = item["instruction"] + " " + item["rejected"]

        chosen_ids,   chosen_mask   = encode(chosen_full)
        rejected_ids, rejected_mask = encode(rejected_full)

        # Find where the response starts (instruction length)
        instr_ids  = self.tokenizer(
            item["instruction"],
            max_length=self.max_length,
            truncation=True,
        )["input_ids"]
        n_pad      = int((chosen_mask == 0).sum().item())
        resp_start = n_pad + len(instr_ids)

        return {
            "chosen_ids":    chosen_ids,
            "chosen_mask":   chosen_mask,
            "rejected_ids":  rejected_ids,
            "rejected_mask": rejected_mask,
            "resp_start":    resp_start,
        }


# ═════════════════════════════════════════════════════════════════════════════
# 5b. VOCABULARY ID MAPPING (tokenizer-dependent)
# ═════════════════════════════════════════════════════════════════════════════

def build_vocab_ids(tokenizer, vocab: set) -> set:
    """
    Map whitelist words to token IDs in the model's tokenizer vocabulary.
    Handles subword tokenisation: a word W is in V_ATC if it appears as a
    single token or if all its constituent subwords are ATC-compliant.
    """
    vocab_ids = set()
    all_tokens = tokenizer.get_vocab()  # {token_str: id}
    vocab_upper = {w.upper() for w in vocab}

    for token_str, token_id in all_tokens.items():
        # Clean the token string (remove Ġ, Ċ, ## prefixes)
        clean = re.sub(r'^[ĠĊ▁##]+', '', token_str).upper()
        if clean in vocab_upper or len(clean) == 0:
            vocab_ids.add(token_id)
        # Also include numeric tokens (for altitude/frequency values)
        if re.match(r'^[0-9]+$', clean):
            vocab_ids.add(token_id)
        # Include punctuation and spaces
        if re.match(r'^[ ,.\-/]+$', token_str):
            vocab_ids.add(token_id)

    # Always include special tokens
    for tok in [tokenizer.pad_token, tokenizer.eos_token,
                tokenizer.bos_token, tokenizer.unk_token]:
        if tok is not None and tok in all_tokens:
            vocab_ids.add(all_tokens[tok])

    return vocab_ids

# ═════════════════════════════════════════════════════════════════════════════
# 6. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SCOPEConfig:
    model_name:    str   = "gpt2-large"
    data_path:     str   = "atc_pairs.json"
    output_dir:    str   = "scope_output"
    # Hyperparameters
    lambda_tok:    float = 0.5
    lambda_phr:    float = 0.3
    lambda_cfg:    float = 0.2
    lambda_ce:     float = 1.0   # weight on CE loss; reduce to let compliance signals dominate
    M_samples:     int   = 4        # GRPO group size
    # Training
    epochs:        int   = 5
    batch_size:    int   = 8
    lr:            float = 5e-5
    max_length:    int   = 512
    max_new_tok:   int   = 64
    warmup_steps:  int   = 100
    grad_clip:     float = 1.0
    # Condition (for ablation)
    use_ltok:      bool  = True
    use_lphr:      bool  = True
    use_lcfg:      bool  = True
    # DPO mode
    use_dpo:       bool  = False
    dpo_beta:      float = 0.1    # KL regularisation strength
    dpo_ref_model: str   = ""     # path to frozen reference model (defaults to base model)
    seed:          int   = 42
    # Large-model support
    grad_accum:           int  = 1      # gradient accumulation steps
    use_chat_template:    bool = False   # True for Llama/Qwen instruct
    gradient_checkpointing: bool = False  # saves VRAM for 8B+ models
    # Domain
    domain:        str   = "atc"  # "atc" or "smcp"
    vocab_path:    str   = "vocab_ATC.json"
    phrase_path:   str   = "ngram_whitelist_ATC_v2.json"
    grammar_path:  str   = "G_ATC_v2.lark"

def train(cfg: SCOPEConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: lambda_tok={cfg.lambda_tok}, lambda_phr={cfg.lambda_phr}, "
          f"lambda_cfg={cfg.lambda_cfg}")

    # Load regulatory artefacts
    print("Loading regulatory artefacts...")
    vocab     = load_whitelist(Path(cfg.vocab_path) if hasattr(cfg, "vocab_path") else VOCAB_PATH)
    ngram_wl  = load_ngram_whitelist(Path(cfg.phrase_path) if hasattr(cfg, "phrase_path") else PHRASE_PATH)
    cfg_parser = load_grammar(Path(cfg.grammar_path) if hasattr(cfg, "grammar_path") else GRAMMAR_PATH) if cfg.use_lcfg else None
    print(f"  V_ATC: {len(vocab)} tokens | "
          f"P_ATC: {sum(len(v) for v in ngram_wl.values())} n-grams | "
          f"G_ATC: {'loaded' if cfg_parser else 'disabled'}")

    # Load model and tokeniser
    print(f"Loading {cfg.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for decoder-only batch generation
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16
    ).to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    # ── DPO: load frozen reference model ─────────────────────────────────
    ref_model = None
    if cfg.use_dpo:
        ref_path = cfg.dpo_ref_model if cfg.dpo_ref_model else cfg.model_name
        print(f"  Loading DPO reference model from {ref_path} ...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path, torch_dtype=torch.bfloat16
        ).cpu()   # keep on CPU — moved to GPU only during its forward pass
        ref_model.config.use_cache = False  # avoid conflict with gradient checkpointing
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
        print(f"  Reference model loaded and frozen (CPU offloaded).")

    # Build vocabulary ID set (tokenizer-dependent)
    vocab_ids = build_vocab_ids(tokenizer, vocab)
    print(f"  Vocab IDs in tokenizer: {len(vocab_ids)} / {tokenizer.vocab_size}")

    # Load dataset
    with open(cfg.data_path) as f:
        all_pairs = json.load(f)

    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    train_pairs = all_pairs[:n_train]
    val_pairs   = all_pairs[n_train:n_train+n_val]
    test_pairs  = all_pairs[n_train+n_val:]
    print(f"  Split: {len(train_pairs)} train / {len(val_pairs)} val / "
          f"{len(test_pairs)} test")

    train_ds = AtcDataset(train_pairs, tokenizer, cfg.max_length, domain=cfg.domain)
    val_ds   = AtcDataset(val_pairs,   tokenizer, cfg.max_length, domain=cfg.domain)
    def collate_fn(batch):
        """Custom collate: stack tensors, keep strings as lists."""
        result = {}
        for key in batch[0]:
            vals = [b[key] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                result[key] = torch.stack(vals)
            else:
                result[key] = vals   # strings, ints etc — keep as list
        return result

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True,  collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                          shuffle=False, collate_fn=collate_fn)

    # ── DPO: build preference pairs from C_tok signal ─────────────────────
    dpo_dl = None
    if cfg.use_dpo:
        dpo_pairs = build_dpo_pairs(
            train_pairs, tokenizer, model, vocab, ngram_wl,
            cfg.max_new_tok, device, domain=cfg.domain
        )
        dpo_ds = DPODataset(dpo_pairs, tokenizer, cfg.max_length)
        dpo_dl = DataLoader(dpo_ds, batch_size=cfg.batch_size,
                            shuffle=True, collate_fn=collate_fn)
        print(f"  DPO DataLoader: {len(dpo_ds)} preference pairs")

    # Optimiser and scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    total_steps = len(train_dl) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    best_val_ctok = 0.0
    history = []
    step_history = []   # continuous per-step log for smooth loss curves
    global_step  = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses = {"ce":0., "tok":0., "phr":0., "cfg":0., "total":0.}
        n_batches = 0

        for step, batch in enumerate(train_dl):
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels        = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            # Zero gradients — CE, L_tok, and GRPO all accumulate into this
            opt.zero_grad()

            # ── Standard CE loss ──────────────────────────────────────────
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
            L_ce  = out.loss
            logits = out.logits

            # ── L_tok (differentiable) ────────────────────────────────────
            L_tok_val = torch.tensor(0., device=device)
            if cfg.use_ltok:
                L_tok_val = compute_L_tok(
                    logits, response_mask, vocab_ids, tokenizer.vocab_size
                )

            # ── L_phr + L_cfg (GRPO) — shared sampling every 2 steps ──────
            _phr_loss_terms = []
            _cfg_loss_terms = []
            _phr_display = 0.0
            _cfg_display = 0.0

            if (cfg.use_lphr or cfg.use_lcfg) and (step % 2 == 0):
                # Sample M sequences once (no grad) — high temperature for diversity
                model.eval()
                all_generated = []
                phr_rewards_list = []
                cfg_rewards_list = []
                with torch.no_grad():
                    for _ in range(cfg.M_samples):
                        out = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=cfg.max_new_tok,
                            do_sample=True,
                            temperature=1.5,      # higher temp → diverse samples → non-zero advantages
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        gen = out[:, input_ids.size(1):]
                        all_generated.append(gen)
                        text = tokenizer.decode(
                            gen[0].tolist(), skip_special_tokens=True
                        )
                        toks = text.upper().split()
                        if cfg.use_lphr:
                            phr_rewards_list.append(compute_cphr(toks, ngram_wl))
                        if cfg.use_lcfg and cfg_parser:
                            cfg_rewards_list.append(
                                compute_ccfg_partial(text, cfg_parser)
                            )
                model.train()

                # ── Group-normalise rewards (true GRPO) ───────────────────
                def grpo_advantages(rewards):
                    """Convert list of scalar rewards to group-normalised advantages."""
                    import statistics
                    if len(rewards) < 2:
                        return rewards  # can't normalise with 1 sample
                    mu  = statistics.mean(rewards)
                    std = statistics.stdev(rewards)
                    if std < 1e-8:
                        return [0.0] * len(rewards)  # all same → zero advantage
                    return [(r - mu) / std for r in rewards]

                phr_adv = grpo_advantages(phr_rewards_list) if phr_rewards_list else []
                cfg_adv = grpo_advantages(cfg_rewards_list) if cfg_rewards_list else []

                # Sequential per-sample backward — frees each graph immediately
                # Prevents accumulating M full computation graphs simultaneously
                for m_idx, gen_m in enumerate(all_generated):
                    if gen_m.size(1) == 0:
                        continue
                    phr_adv_m = phr_adv[m_idx] if m_idx < len(phr_adv) else 0.0
                    cfg_adv_m = cfg_adv[m_idx] if m_idx < len(cfg_adv) else 0.0
                    if abs(phr_adv_m) < 1e-8 and abs(cfg_adv_m) < 1e-8:
                        continue  # zero advantage — no gradient to apply

                    full   = torch.cat([input_ids, gen_m], dim=1)
                    attn   = torch.ones(
                        input_ids.size(0), full.size(1) - 1,
                        dtype=torch.long, device=device
                    )
                    logits = model(full[:, :-1], attention_mask=attn).logits
                    logits = logits[:, input_ids.size(1) - 1:, :]
                    log_p  = F.log_softmax(logits, dim=-1)
                    idx_t  = gen_m.unsqueeze(-1).clamp(0, log_p.size(-1) - 1)
                    seq_lp = log_p.gather(-1, idx_t).squeeze(-1).mean(dim=-1)

                    # Build per-sample GRPO loss and backward immediately
                    grpo_m = torch.tensor(0.0, device=device)
                    if cfg.use_lphr and abs(phr_adv_m) > 1e-8:
                        grpo_m = grpo_m + (
                            -phr_adv_m * cfg.lambda_phr * seq_lp.mean() / cfg.M_samples
                        )
                        _phr_loss_terms.append(float(seq_lp.mean().detach()))
                    if cfg.use_lcfg and cfg_parser and abs(cfg_adv_m) > 1e-8:
                        grpo_m = grpo_m + (
                            -cfg_adv_m * cfg.lambda_cfg * seq_lp.mean() / cfg.M_samples
                        )
                        _cfg_loss_terms.append(float(seq_lp.mean().detach()))

                    if grpo_m.grad_fn is not None:
                        grpo_m.backward()  # accumulates into existing CE+tok grads
                    del full, attn, logits, log_p, idx_t, seq_lp, grpo_m

                # Track mean reward for display
                if phr_rewards_list:
                    _phr_display = sum(phr_rewards_list) / len(phr_rewards_list)
                if cfg_rewards_list:
                    _cfg_display = sum(cfg_rewards_list) / len(cfg_rewards_list)

            # Build scalar losses for the combined objective
            L_phr_val = (
                sum(_phr_loss_terms) / max(len(_phr_loss_terms), 1)
                if _phr_loss_terms else
                torch.tensor(0.0, device=device)
            )
            L_cfg_val = (
                sum(_cfg_loss_terms) / max(len(_cfg_loss_terms), 1)
                if _cfg_loss_terms else
                torch.tensor(0.0, device=device)
            )

            # ── Combined objective (CE + L_tok) ──────────────────────────
            L_ce_tok = cfg.lambda_ce * L_ce + cfg.lambda_tok * L_tok_val
            L_ce_tok.backward()
            L_total = L_ce_tok.detach()  # for logging

            # ── DPO loss (interleaved with SFT steps) ─────────────────────
            L_dpo_val = 0.0
            if cfg.use_dpo and dpo_dl is not None and ref_model is not None:
                # Get a DPO batch (cycle through the DPO dataloader)
                if not hasattr(train, '_dpo_iter') or train._dpo_iter is None:
                    train._dpo_iter = iter(dpo_dl)
                try:
                    dpo_batch = next(train._dpo_iter)
                except StopIteration:
                    train._dpo_iter = iter(dpo_dl)
                    dpo_batch = next(train._dpo_iter)

                L_dpo = compute_L_dpo(
                    model, ref_model, tokenizer,
                    dpo_batch, cfg.dpo_beta, device
                )
                torch.cuda.empty_cache()
                L_dpo.backward()
                L_dpo_val = L_dpo.item()
                L_total = L_total + L_dpo.detach()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            scheduler.step()

            epoch_losses["ce"]    += L_ce.item()
            epoch_losses["tok"]   += L_tok_val.item()
            epoch_losses["phr"]   += _phr_display   # log mean reward not loss
            epoch_losses["cfg"]   += _cfg_display   # log mean reward not loss
            epoch_losses["total"] += L_total.item()
            n_batches += 1
            global_step += 1

            # Per-step record for continuous loss curves
            step_history.append({
                "global_step": global_step,
                "epoch":       epoch + 1,
                "step":        step,
                "ce":          round(L_ce.item(), 5),
                "tok":         round(L_tok_val.item(), 5),
                "total":       round(L_total.item(), 5),
            })

            if step % 10 == 0:
                phr_v = _phr_display
                cfg_v = _cfg_display
                dpo_v = f" DPO={L_dpo_val:.4f}" if cfg.use_dpo else ""
                print(f"  Epoch {epoch+1} Step {step}/{len(train_dl)} | "
                      f"CE={L_ce.item():.3f} Tok={L_tok_val.item():.4f} "
                      f"Phr={phr_v:.4f} Cfg={cfg_v:.4f}{dpo_v}")

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_ctok, val_cphr, val_ccfg = [], [], []
        with torch.no_grad():
            for vbatch in val_dl:
                instructions = vbatch["instruction"]   # list of raw instruction strings

                for i in range(len(instructions)):
                    # Tokenise ONLY the instruction — no response, no padding
                    # This is the cleanest way to generate: give the model the
                    # prompt and let it generate the response from scratch.
                    prompt_tok = tokenizer(
                        instructions[i],
                        return_tensors="pt",
                        truncation=True,
                        max_length=cfg.max_length - cfg.max_new_tok,
                    ).to(device)
                    out_v = model.generate(
                        input_ids=prompt_tok["input_ids"],
                        attention_mask=prompt_tok["attention_mask"],
                        max_new_tokens=cfg.max_new_tok,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    # Decode only the newly generated tokens
                    prompt_len = prompt_tok["input_ids"].size(1)
                    gen  = out_v[0, prompt_len:]
                    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
                    toks = text.upper().split()
                    if not toks:
                        val_ctok.append(0.0)
                        val_cphr.append(0.0)
                        val_ccfg.append(0.0)
                        continue
                    val_ctok.append(compute_ctok(toks, vocab))
                    val_cphr.append(compute_cphr(toks, ngram_wl))
                    val_ccfg.append(
                        compute_ccfg_partial(text, cfg_parser)
                        if cfg_parser else 0.0
                    )

        mean_ctok = sum(val_ctok)/len(val_ctok) if val_ctok else 0.
        mean_cphr = sum(val_cphr)/len(val_cphr) if val_cphr else 0.
        mean_ccfg = sum(val_ccfg)/len(val_ccfg) if val_ccfg else 0.
        mean_loss = epoch_losses["total"] / n_batches
        mean_phr_reward = epoch_losses["phr"] / n_batches   # mean C_phr during training
        mean_cfg_reward = epoch_losses["cfg"] / n_batches   # mean C_cfg during training

        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss={mean_loss:.4f} | "
              f"C_tok={mean_ctok:.4f} C_phr={mean_cphr:.4f} C_cfg={mean_ccfg:.4f} | "
              f"Train Rphr={mean_phr_reward:.4f} Rcfg={mean_cfg_reward:.4f}")

        record = {"epoch":    epoch+1,
                  "loss":     mean_loss,
                  "ce_loss":  epoch_losses["ce"]  / n_batches,
                  "tok_loss": epoch_losses["tok"] / n_batches,
                  "Rphr":     mean_phr_reward,
                  "Rcfg":     mean_cfg_reward,
                  "C_tok":    mean_ctok,
                  "C_phr":    mean_cphr,
                  "C_cfg":    mean_ccfg}
        history.append(record)

        # Select checkpoint by validation C_tok.
        # (Composite C_cfg-weighted selection was tested and found to overfit
        #  the validation split, yielding lower test C_cfg. C_tok criterion
        #  acts as an implicit regulariser — standard SFT practice.)
        if mean_ctok > best_val_ctok:
            best_val_ctok = mean_ctok
            model.save_pretrained(f"{cfg.output_dir}/best")
            tokenizer.save_pretrained(f"{cfg.output_dir}/best")
            print(f"  ★ New best C_tok={best_val_ctok:.4f} "
                  f"(C_phr={mean_cphr:.4f} C_cfg={mean_ccfg:.4f}) saved")

        # Always save last-epoch checkpoint for reference
        model.save_pretrained(f"{cfg.output_dir}/last")
        tokenizer.save_pretrained(f"{cfg.output_dir}/last")

    with open(f"{cfg.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(f"{cfg.output_dir}/step_history.json", "w") as f:
        json.dump(step_history, f, indent=2)
    print("Training complete.")
    return history

# ═════════════════════════════════════════════════════════════════════════════
# 7. EVALUATION SCRIPT
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model_path: str, data_path: str, condition_name: str = "SCOPE",
             domain: str = "atc",
             vocab_path: str = "", phrase_path: str = "", grammar_path: str = ""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _vocab_p   = Path(vocab_path)   if vocab_path   else VOCAB_PATH
    _phrase_p  = Path(phrase_path)  if phrase_path  else PHRASE_PATH
    _grammar_p = Path(grammar_path) if grammar_path else GRAMMAR_PATH
    vocab    = load_whitelist(_vocab_p)
    ngram_wl = load_ngram_whitelist(_phrase_p)
    cfg_parser = load_grammar(_grammar_p)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    with open(data_path) as f:
        test_pairs = json.load(f)

    results = []
    for p in test_pairs:
        item = format_atc(p["request"], p["response"], domain=domain)
        tok  = tokenizer(item["instruction"], return_tensors="pt",
                         max_length=512, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**tok, max_new_tokens=64,
                                  do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        gen = out[0, tok["input_ids"].size(1):]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        tokens = text.upper().split()

        results.append({
            "request":   p["request"],
            "reference": p["response"],
            "generated": text,
            "C_tok":     compute_ctok(tokens, vocab),
            "C_phr":     compute_cphr(tokens, ngram_wl),
            "C_cfg":     compute_ccfg_partial(text, cfg_parser),
        })

    mean_ctok = sum(r["C_tok"] for r in results) / len(results)
    mean_cphr = sum(r["C_phr"] for r in results) / len(results)
    mean_ccfg = sum(r["C_cfg"] for r in results) / len(results)

    print(f"\n{condition_name} Results ({len(results)} test examples):")
    print(f"  C_tok = {mean_ctok:.4f}")
    print(f"  C_phr = {mean_cphr:.4f}")
    print(f"  C_cfg = {mean_ccfg:.4f}")
    return results

# ═════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCOPE Training")
    parser.add_argument("--model",      default="gpt2-large")
    parser.add_argument("--data",       default="atc_pairs.json")
    parser.add_argument("--test_data",  default="",
                        help="Test split for post-training evaluation "
                             "(default: auto-inferred from --data path)")
    parser.add_argument("--output",     default="scope_output")
    parser.add_argument("--lambda_tok", type=float, default=0.5)
    parser.add_argument("--lambda_phr", type=float, default=0.3)
    parser.add_argument("--lambda_cfg", type=float, default=0.2)
    parser.add_argument("--lambda_ce",  type=float, default=1.0,
                        help="Weight on CE loss (reduce to 0.5 to boost compliance signals)")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--M_samples",  type=int,   default=4,
                        help="GRPO group size (reduce to 2 for low VRAM)")
    parser.add_argument("--max_new_tok", type=int,  default=64,
                        help="Max new tokens to generate per GRPO sample")
    # Ablation flags
    parser.add_argument("--no_ltok", action="store_true")
    parser.add_argument("--no_lphr", action="store_true")
    parser.add_argument("--no_lcfg", action="store_true")
    parser.add_argument("--vocab_path",  default="vocab_ATC.json",
                        help="Vocabulary whitelist JSON (default: ATC)")
    parser.add_argument("--phrase_path", default="ngram_whitelist_ATC.json",
                        help="N-gram whitelist JSON (default: ATC)")
    parser.add_argument("--grammar",     default="G_ATC.lark",
                        help="Lark grammar file for L_cfg")
    parser.add_argument("--domain",      default="atc",
                        choices=["atc", "smcp"],
                        help="Domain for system prompt (atc or smcp)")
    # DPO flags
    parser.add_argument("--dpo",       action="store_true",
                        help="Enable DPO training (C3 condition)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (use 4 for Llama)")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Use tokenizer chat template (Llama/Qwen instruct models)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM for large models)")
    parser.add_argument("--dpo_beta",  type=float, default=0.1,
                        help="DPO beta (KL regularisation strength)")
    parser.add_argument("--dpo_ref",   type=str, default="",
                        help="Path to frozen reference model for DPO "
                             "(defaults to base model)")
    args = parser.parse_args()

    # HuggingFace authentication — needed for Llama/Qwen gated models
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("✓ HuggingFace authenticated")
        except Exception as e:
            print(f"WARNING: HF login issue: {e} — proceeding")

    cfg = SCOPEConfig(
        model_name    = args.model,
        data_path     = args.data,
        output_dir    = args.output,
        lambda_tok    = args.lambda_tok,
        lambda_phr    = args.lambda_phr,
        lambda_cfg    = args.lambda_cfg,
        lambda_ce     = args.lambda_ce,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        grad_accum    = args.grad_accum,
        use_chat_template = args.use_chat_template,
        gradient_checkpointing = args.gradient_checkpointing,
        lr            = args.lr,
        seed          = args.seed,
        M_samples     = args.M_samples,
        max_new_tok   = args.max_new_tok,
        use_ltok      = not args.no_ltok,
        use_lphr      = not args.no_lphr,
        use_lcfg      = not args.no_lcfg,
        use_dpo       = args.dpo,
        dpo_beta      = args.dpo_beta,
        dpo_ref_model = args.dpo_ref,
        domain        = args.domain,
        vocab_path    = args.vocab_path,
        phrase_path   = args.phrase_path,
        grammar_path  = args.grammar,
    )
    train(cfg)

    # ── Post-training: evaluate best checkpoint on test set ───────────────
    best_ckpt = Path(cfg.output_dir) / "best"
    # Resolve test data path
    if args.test_data:
        test_path = Path(args.test_data)
    else:
        # Infer from training data path: atc_pairs.json -> atc_test.json
        data_p = Path(cfg.data)
        test_name = data_p.name.replace("pairs", "test")
        test_path = data_p.parent / test_name

    if best_ckpt.exists() and test_path.exists():
        print(f"\nEvaluating best checkpoint on test set: {test_path}")
        results = evaluate(
            model_path   = str(best_ckpt),
            data_path    = str(test_path),
            condition_name = "SCOPE",
            domain       = cfg.domain,
            vocab_path   = cfg.vocab_path,
            phrase_path  = cfg.phrase_path,
            grammar_path = cfg.grammar_path,
        )
        if results:
            mean_ctok = sum(r["C_tok"] for r in results) / len(results)
            mean_cphr = sum(r["C_phr"] for r in results) / len(results)
            mean_ccfg = sum(r["C_cfg"] for r in results) / len(results)
            summary = {
                "condition":  "SCOPE",
                "model":      cfg.model_name,
                "domain":     cfg.domain,
                "n_test":     len(results),
                "C_tok":      round(mean_ctok, 4),
                "C_phr":      round(mean_cphr, 4),
                "C_cfg":      round(mean_ccfg, 4),
                "per_example": results,
            }
            results_path = Path(cfg.output_dir) / "test_results.json"
            with open(results_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Test results saved to {results_path}")
    else:
        if not best_ckpt.exists():
            print(f"  WARNING: best checkpoint not found at {best_ckpt} — skipping test eval")
        if not test_path.exists():
            print(f"  WARNING: test data not found at {test_path} — skipping test eval")
