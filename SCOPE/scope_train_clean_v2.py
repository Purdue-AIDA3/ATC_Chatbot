#!/usr/bin/env python3
"""
scope_train_v2.py — SCOPE Training Pipeline with TensorBoard, Config File,
                     FORMAT B Data Loading, Grammar-Rule Diagnostics,
                     and Data Curriculum Strategies A/B/C.

Key additions over scope_train_curriculum.py
─────────────────────────────────────────────
TB1.  TensorBoard integration — step-level and epoch-level logging.
TB2.  External YAML/JSON config file — all hyperparams in one place.
TB3.  Per-component CSV logging — ce/tok/phr/cfg/total per step and epoch.
TB4.  Grammar-rule failure diagnostics — which G_ATC section drives C_cfg loss.
TB5.  FORMAT B data loading — reads atc_pairs_enriched.jsonl directly.
TB6.  Data curriculum strategies A / B / C (orthogonal to loss curriculum).
TB7.  Console output mirrored to log file — full reproducibility.
TB8.  Lambda sensitivity tracking — logs effective λ values per step.

Data curriculum strategies
──────────────────────────
A — Two-phase:      Phase 1 trains on gold+silver targets only.
                    Phase 2 adds bronze (all pairs).
B — Weighted loss:  All pairs in one run; CE loss scaled by quality_tier.
                    gold×3, silver×1.5, bronze×1 (configurable).
C — Silver+gold:    Bronze pairs excluded entirely throughout training.

Loss curriculum (unchanged from scope_train_curriculum.py)
──────────────────────────────────────────────────────────
Phase 1: CE only.
Phase 2: CE + L_tok.
Phase 3: CE + L_tok + L_phr [+ L_cfg per condition].

The two curricula are orthogonal and can be combined freely.

Lambda update for new P_ATC
────────────────────────────
P_ATC shrank from 46,990 (corpus-derived, contaminated) to 1,169
(specification-first, clean). C_phr scores lower for identical outputs
because fewer n-grams match. lambda_phr is increased from 0.3→0.5 to
maintain training signal strength. Use GradNorm to observe and self-correct.
"""

import json, re, math, random, argparse, os, sys, csv, statistics
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          get_cosine_schedule_with_warmup)

# Optional: TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    print("WARNING: tensorboard not installed. pip install tensorboard")

# Optional: YAML config
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Optional: Lark grammar
try:
    from lark import Lark, exceptions as lark_exc
    LARK_AVAILABLE = True
except ImportError:
    LARK_AVAILABLE = False
    print("WARNING: lark not installed. L_cfg will be disabled.")

# Optional: BERTScore
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 0. LOGGING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class TeeLogger:
    """Mirror stdout/stderr to a log file."""
    def __init__(self, path: str):
        self.file    = open(path, 'w', buffering=1)
        self.stdout  = sys.stdout
        sys.stdout   = self

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


class StepCSVWriter:
    """Write per-step loss components to a CSV file."""
    def __init__(self, path: str):
        self.path   = path
        self.file   = open(path, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=[
            'global_step','epoch','step','phase',
            'loss_ce','loss_tok','loss_phr','loss_cfg','loss_total',
            'reward_phr','reward_cfg',
            'lam_ce','lam_tok','lam_phr','lam_cfg',
            'lr','grad_norm',
        ])
        self.writer.writeheader()

    def write(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


class EpochCSVWriter:
    """Write per-epoch metrics to a CSV file."""
    def __init__(self, path: str):
        self.path   = path
        self.file   = open(path, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=[
            'epoch','phase','data_phase',
            'train_ce','train_tok','train_phr','train_cfg','train_total',
            'val_ctok','val_cphr','val_ccfg','val_cbar',
            'slot_f1','da_f1','halluc_pct','bertscore',
            'lam_ce','lam_tok','lam_phr','lam_cfg',
        ])
        self.writer.writeheader()

    def write(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG — load from YAML/JSON or argparse
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SCOPEConfig:
    # Run identity
    run_name:    str   = "scope_run"
    output_dir:  str   = "scope_output"
    seed:        int   = 42

    # Model
    model_name:           str  = "gpt2-large"
    domain:               str  = "atc"
    use_chat_template:    bool = False
    gradient_checkpointing: bool = False

    # Artefacts
    vocab_path:    str = "vocab_ATC_clean.json"
    phrase_path:   str = "ngram_whitelist_ATC_clean.json"
    grammar_path:  str = "G_ATC_unified.lark"

    # Data (FORMAT B)
    data_path:   str = "atc_pairs_enriched.jsonl"
    train_data:  str = ""
    val_data:    str = ""
    test_data:   str = ""

    # Data curriculum strategy: "A" | "B" | "C"
    data_curriculum:    str   = "B"
    tier_weight_gold:   float = 3.0
    tier_weight_silver: float = 1.5
    tier_weight_bronze: float = 1.0
    data_phase1_frac:   float = 0.5   # for strategy A

    # Loss weights
    lambda_ce:   float = 1.0
    lambda_tok:  float = 0.5
    lambda_phr:  float = 0.5   # increased from 0.3 — see docstring
    lambda_cfg:  float = 0.2

    # Loss curriculum
    loss_curriculum:        bool  = True
    loss_curriculum_phase1: float = 1/3
    loss_curriculum_phase2: float = 2/3
    loss_curriculum_ramp:   int   = 50

    # Ablation flags
    use_ltok: bool = True
    use_lphr: bool = True
    use_lcfg: bool = True

    # Training
    epochs:       int   = 5
    batch_size:   int   = 4
    grad_accum:   int   = 4
    lr:           float = 2e-5
    warmup_ratio: float = 0.05
    warmup_steps: int   = 0
    grad_clip:    float = 1.0
    max_length:   int   = 512
    max_new_tok:  int   = 80
    M_samples:    int   = 4

    # GradNorm
    use_gradnorm:         bool  = True
    gradnorm_alpha:       float = 1.5
    gradnorm_update_freq: int   = 20

    # DPO
    use_dpo:       bool  = False
    dpo_beta:      float = 0.1
    dpo_ref_model: str   = ""

    # Checkpoint
    checkpoint_metric:       str   = "c_bar"
    early_stop_patience:     int   = 2
    hallucination_threshold: float = 0.10

    # BERTScore
    bertscore_model:  str   = "bert-base-uncased"
    bertscore_weight: float = 0.0
    finetune_bert:    bool  = False
    bert_mlm_epochs:  int   = 3
    bert_mlm_batch:   int   = 16
    bert_mlm_lr:      float = 2e-5

    # TensorBoard / logging
    tensorboard_dir:   str  = "runs"
    log_step_freq:     int  = 10
    csv_log_steps:     bool = True
    csv_log_epochs:    bool = True
    log_grammar_rules: bool = True

    # Multi-GPU
    use_8bit_adam: bool = False


def load_config(path: str) -> dict:
    """Load YAML or JSON config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if p.suffix in ('.yaml', '.yml'):
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for .yaml config. pip install pyyaml")
        with open(p) as f:
            return yaml.safe_load(f)
    with open(p) as f:
        return json.load(f)


def config_from_dict(d: dict) -> SCOPEConfig:
    """Build SCOPEConfig from a dict, ignoring unknown keys."""
    import dataclasses
    known = {f.name for f in dataclasses.fields(SCOPEConfig)}
    return SCOPEConfig(**{k: v for k, v in d.items() if k in known})


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MULTI-GPU HELPERS  (unchanged from curriculum script)
# ═══════════════════════════════════════════════════════════════════════════════

def _n_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def _device_map(force_single: bool = False) -> Optional[str]:
    if not force_single and _n_gpus() >= 2:
        return "auto"
    return None

def _tensor_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _load_model(model_name: str, dtype=torch.bfloat16,
                gradient_checkpointing: bool = False):
    dm = _device_map()
    kw = dict(dtype=dtype)
    if dm:
        kw["device_map"] = dm
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    if dm is None:
        model = model.to(_tensor_device())
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    n = _n_gpus()
    print(f"  Model: {'split across '+str(n)+' GPUs' if dm else 'single device'}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REGULATORY ARTEFACTS
# ═══════════════════════════════════════════════════════════════════════════════

def load_whitelist(path: Path) -> set:
    with open(path) as f:
        v = json.load(f)
    return set(w.upper() for w in v)

def load_ngram_whitelist(path: Path) -> Dict[int, set]:
    with open(path) as f:
        raw = json.load(f)
    return {
        2: set(tuple(g) for g in raw.get('bigrams',  [])),
        3: set(tuple(g) for g in raw.get('trigrams', [])),
        4: set(tuple(g) for g in raw.get('4grams',   [])),
    }

def _strict_src(grammar_path: str) -> str:
    """Return grammar with catch-all rules removed (drop lines, not comment)."""
    src = open(grammar_path).read()
    lines, out, skip = src.splitlines(), [], False
    for line in lines:
        is_catchall = any(re.match(p, line) for p in [
            r'^\s*\|\s*general_instr\b', r'^general_instr\s*:',
            r'^\s*\|\s*pilot_general\b', r'^pilot_general\s*:',
            r'^word\s*:'])
        if is_catchall:
            if re.match(r'^(general_instr|pilot_general|word)\s*:', line):
                skip = True
        elif skip:
            cont = re.match(r'^\s*\|', line) or (line.strip() and
                   not re.match(r'^\s*\w+\s*:', line) and
                   not line.strip().startswith(('//', '%')))
            if cont:
                pass
            else:
                skip = False
                out.append(line)
        else:
            out.append(line)
    return '\n'.join(out)

def load_grammar_parsers(grammar_path: str) -> Optional[dict]:
    """Return role-gated strict parsers or None if lark unavailable."""
    if not LARK_AVAILABLE:
        return None
    strict = _strict_src(grammar_path)
    return {
        'controller': Lark(strict, parser='earley',
                           start='controller_utterance', ambiguity='resolve'),
        'pilot':      Lark(strict, parser='earley',
                           start='pilot_utterance',      ambiguity='resolve'),
        'unified':    Lark(strict, parser='earley',
                           start='start',                ambiguity='resolve'),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMPLIANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ctok(tokens: list, vocab: set) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in vocab) / len(tokens)

def compute_cphr(tokens: list, ngram_wl: Dict[int, set]) -> float:
    if not tokens:
        return 0.0
    total, hits = 0, 0
    for n, wl in ngram_wl.items():
        for i in range(len(tokens) - n + 1):
            total += 1
            if tuple(tokens[i:i+n]) in wl:
                hits += 1
    return hits / total if total > 0 else 0.0

def compute_ccfg(text: str, parsers: dict, role: str = 'unified') -> float:
    """Binary parse check. Returns 1.0 if text parses, 0.0 otherwise."""
    if parsers is None:
        return 0.0
    parser = parsers.get(role, parsers['unified'])
    try:
        parser.parse(text.upper().strip())
        return 1.0
    except Exception:
        return 0.0

def compute_ccfg_partial(text: str, parsers: dict, role: str = 'unified') -> float:
    """Partial credit: try both role-specific and unified parser."""
    if parsers is None:
        return 0.0
    for key in [role, 'unified']:
        try:
            parsers[key].parse(text.upper().strip())
            return 1.0
        except Exception:
            pass
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4b. GRAMMAR RULE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

# Map Lark section keywords → paper §3.x label
RULE_SECTION_MAP = {
    'altitude_instr':    '§3.1 Altitude',
    'heading_instr':     '§3.2 Heading',
    'speed_instr':       '§3.3 Speed',
    'frequency_instr':   '§3.4 Frequency',
    'clearance_instr':   '§3.5 Clearance',
    'taxi_instr':        '§3.6 Taxi',
    'takeoff_instr':     '§3.7 Takeoff',
    'approach_instr':    '§3.8 Approach',
    'hold_instr':        '§3.9 Hold',
    'traffic_instr':     '§3.10 Traffic',
    'surface_wind_instr':'§3.10b SurfWind',
    'ident_instr':       '§3.11 Transponder',
    'report_instr':      '§3.12 Report',
    'confirm_instr':     '§3.13 Confirm',
    'route_instr':       '§3.14 Route',
    'information_instr': '§3.15 ATIS',
    'leaving_instr':     '§3.16 Leaving',
    'pilot_action_report':'§5 Pilot',
    'callsign_phrase':   'Callsign',
    'number_word':       'Number',
    'runway_id':         'RunwayID',
}

def diagnose_parse_failure(text: str, parsers: dict,
                           role: str = 'unified') -> str:
    """
    Attempt to parse text and extract the failing rule name from the
    Lark exception. Returns a section label from RULE_SECTION_MAP
    or 'Unknown' if the rule cannot be identified.
    """
    if parsers is None:
        return 'NoParser'
    parser = parsers.get(role, parsers['unified'])
    try:
        parser.parse(text.upper().strip())
        return 'ParseOK'  # shouldn't happen when called after a failure
    except Exception as e:
        err_str = str(e)
        # Lark error format: "Unexpected token ... in rule_name"
        # or "Expected ... in <rule_name>"
        for rule, label in RULE_SECTION_MAP.items():
            if rule in err_str:
                return label
        # Try to extract any rule name pattern
        m = re.search(r"in rule '?(\w+)'?", err_str)
        if m:
            return m.group(1)
        return 'Other'


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FORMAT B DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

# ── Prompt components ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_ATC = (
    "You are an expert in Air Traffic Control (ATC) radio communication "
    "for UAV (Unmanned Aerial Vehicle) operations in controlled airspace.\n\n"
    "Your task is to generate the RESPONSE side of an ATC radio exchange, "
    "given the INPUT side. The response must conform strictly to "
    "ICAO Doc 4444 Chapter 12 standard phraseology.\n\n"
    "MANDATORY RULES:\n"
    "1. INDIVIDUAL DIGIT PRONUNCIATION — speak each digit separately:\n"
    "   correct: ONE EIGHT ZERO  |  wrong: ONE EIGHTY  or  180\n"
    "   correct: NINER THOUSAND  |  wrong: NINE THOUSAND\n"
    "   correct: ONE ONE EIGHT DECIMAL THREE  |  wrong: ONE-ONE-EIGHT-POINT-THREE\n"
    "2. NO DISFLUENCIES — omit all filler words:\n"
    "   forbidden: AH, UH, UM, OK, OKAY, RIGHT, YEAH, SURE, SO, WELL\n"
    "3. TERSE REGISTER — use the minimum words required by procedure:\n"
    "   correct: DELTA SIX TURN LEFT HEADING ONE EIGHT ZERO\n"
    "   wrong:   DELTA SIX PLEASE TURN LEFT TO A HEADING OF ONE EIGHTY DEGREES\n"
    "4. CALLSIGN REQUIRED — include the aircraft callsign where procedure demands:\n"
    "   controller addressing pilot: begin with callsign (DELTA SIX DESCEND AND MAINTAIN...)\n"
    "   pilot readback: include callsign (DESCEND AND MAINTAIN THREE THOUSAND DELTA SIX)\n"
    "5. STANDARD WORDS — use only ICAO-prescribed vocabulary:\n"
    "   AFFIRM not AFFIRMATIVE | NINER not NINE | WILCO not WILL DO\n"
    "   ROGER for acknowledgement | CORRECTION when self-correcting"
)

SYSTEM_PROMPT_SMCP = (
    "You are an expert in maritime radio communication for vessel operations.\n\n"
    "Your task is to generate the RESPONSE side of a maritime radio exchange, "
    "given the INPUT side. The response must conform strictly to "
    "IMO SMCP (Standard Marine Communication Phrases) standards.\n\n"
    "MANDATORY RULES:\n"
    "1. Use standard IMO SMCP phrases — do not paraphrase.\n"
    "2. Be terse — maritime radio uses minimum necessary words.\n"
    "3. Include vessel name/callsign where required by procedure.\n"
    "4. No disfluencies: AH, UM, OK, OKAY are forbidden."
)

EXCHANGE_TYPE_LABELS = {
    'instruction_readback': 'ATC instruction followed by pilot readback',
    'pilot_to_controller':  'Pilot report or request followed by ATC response',
    'handoff':              'Frequency handoff or acknowledgement exchange',
    'other':                'General ATC/Pilot radio exchange',
}

ROLE_LABELS = {
    'controller': 'Air Traffic Controller (ATC)',
    'pilot':      'UAV Pilot',
    'unified':    'ATC or Pilot',
}


def load_jsonl(path: str) -> List[dict]:
    """Load a JSONL file, skipping blank lines."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def select_target(pair: dict) -> str:
    """
    Select the training target from a FORMAT B pair.
    Priority: expert_corrected (gold) > judge_corrected (silver) > resp_text (bronze).
    """
    tier = pair.get('quality_tier', 'bronze')
    if tier == 'gold':
        t = pair.get('expert_corrected') or pair.get('judge_corrected')
        if t and t.strip():
            return t.strip()
    if tier in ('gold', 'silver'):
        t = pair.get('judge_corrected')
        if t and t.strip():
            return t.strip()
    return (pair.get('resp_text') or pair.get('resp_text_raw', '')).strip()


def filter_training_pairs(pairs: List[dict],
                           data_curriculum: str,
                           phase: int) -> List[dict]:
    """
    Apply data curriculum filtering.
    Strategy A phase 1: gold+silver only.
    Strategy B / A phase 2: all non-unintelligible pairs.
    """
    if data_curriculum == 'A' and phase == 1:
        return [p for p in pairs
                if p.get('quality_tier', 'bronze') in ('gold', 'silver')
                and not p.get('resp_unintelligible', False)]
    return [p for p in pairs if not p.get('resp_unintelligible', False)]


def tier_weight(pair: dict, cfg: SCOPEConfig) -> float:
    """Return CE loss weight for strategy B; 1.0 otherwise."""
    if cfg.data_curriculum != 'B':
        return 1.0
    return {
        'gold':   cfg.tier_weight_gold,
        'silver': cfg.tier_weight_silver,
        'bronze': cfg.tier_weight_bronze,
    }.get(pair.get('quality_tier', 'bronze'), 1.0)


def format_pair_b(pair: dict, tokenizer=None,
                  use_chat_template: bool = False,
                  domain: str = "atc") -> dict:
    """
    Format a FORMAT B pair into instruction/response/full strings.

    Prompt structure (Alpaca-style):
      ### System:   static ICAO rules — digit pronunciation, no disfluencies,
                    terse register, callsign convention, standard vocabulary
      ### Context:  dynamic per pair — exchange type, airport, callsign,
                    which role is being generated
      ### Input:    the provided utterance with explicit speaker label
      ### Response: header with responding role label — model generates here

    This replaces the earlier "[STATE: fsm] [ROLE: role] Generate ICAO-compliant
    phraseology." system string, which gave the model no actionable guidance.
    The new prompt names each rule concretely with correct/wrong examples.
    """
    req_text      = pair.get('req_text',      pair.get('request',  '')).strip()
    req_role      = pair.get('req_role',       'unknown')
    resp_role     = pair.get('resp_role',      'unified')
    exchange_type = pair.get('exchange_type',  'other')
    airport       = pair.get('airport',        '')
    callsign      = pair.get('callsign',       '')
    tier          = pair.get('quality_tier',   'bronze')
    response      = select_target(pair)

    req_label  = ROLE_LABELS.get(req_role,  req_role.title())
    resp_label = ROLE_LABELS.get(resp_role, resp_role.title())
    exch_desc  = EXCHANGE_TYPE_LABELS.get(exchange_type, exchange_type)
    system     = SYSTEM_PROMPT_SMCP if domain == "smcp" else SYSTEM_PROMPT_ATC

    # Dynamic context block — airport and callsign only when available
    ctx_lines  = [f"Exchange type: {exch_desc}"]
    if airport:
        ctx_lines.append(f"Airport / facility: {airport}")
    if callsign:
        ctx_lines.append(f"Aircraft callsign: {callsign}")
    ctx_lines.append(f"Generate the response for: {resp_label}")
    context_str = "\n".join(ctx_lines)

    if use_chat_template and tokenizer is not None:
        user_msg = (
            "Below is a radio communication exchange in controlled airspace. "
            "Complete the response side following strict ICAO phraseology standards.\n\n"
            f"### Context:\n{context_str}\n\n"
            f"### Input ({req_label}):\n{req_text}"
        )
        messages  = [{"role": "system", "content": system},
                     {"role": "user",   "content": user_msg}]
        instruction = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        full = instruction + response
        return {"instruction": instruction, "full": full,
                "response": response, "role": resp_role, "tier": tier,
                "response_text": response}

    # Alpaca-style (default — works with all base models)
    instruction = (
        "Below is a radio communication exchange in controlled airspace. "
        "Complete the response side following strict ICAO phraseology standards.\n\n"
        f"### System:\n{system}\n\n"
        f"### Context:\n{context_str}\n\n"
        f"### Input ({req_label}):\n{req_text}\n\n"
        f"### Response ({resp_label}):"
    )
    full = instruction + " " + response
    return {
        "instruction":   instruction,
        "response":      response,
        "full":          full,
        "role":          resp_role,
        "tier":          tier,
        "response_text": response,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 6. DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def build_vocab_ids(tokenizer, vocab: set) -> set:
    vocab_ids  = set()
    all_tokens = tokenizer.get_vocab()
    vocab_upper = {w.upper() for w in vocab}
    for token_str, token_id in all_tokens.items():
        clean = re.sub(r'^[ĠĊ▁##]+', '', token_str).upper()
        if clean in vocab_upper or len(clean) == 0:
            vocab_ids.add(token_id)
        if re.match(r'^[0-9]+$', clean):
            vocab_ids.add(token_id)
        if re.match(r'^[ ,.\-/]+$', token_str):
            vocab_ids.add(token_id)
    for tok in [tokenizer.pad_token, tokenizer.eos_token,
                tokenizer.bos_token, tokenizer.unk_token]:
        if tok and tok in all_tokens:
            vocab_ids.add(all_tokens[tok])
    return vocab_ids

class AtcDatasetB(Dataset):
    """Dataset for FORMAT B pairs (atc_pairs_enriched.jsonl)."""
    def __init__(self, pairs: List[dict], tokenizer, cfg: SCOPEConfig):
        self.tokenizer  = tokenizer
        self.max_length = cfg.max_length
        self.cfg        = cfg
        self.samples    = []
        for p in pairs:
            item = format_pair_b(p, tokenizer,
                                 cfg.use_chat_template, cfg.domain)
            item['weight'] = tier_weight(p, cfg)
            item['role']   = p.get('resp_role', 'unified')
            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tok  = self.tokenizer(
            item["full"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        # Build labels: -100 for instruction, token ids for response
        instr_tok = self.tokenizer(
            item["instruction"],
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )
        n_instr     = len(instr_tok["input_ids"])
        labels      = input_ids.clone()
        labels[:n_instr] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        # Response mask for L_tok
        response_mask = torch.zeros_like(input_ids)
        response_mask[n_instr:] = attention_mask[n_instr:]

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "response_mask":  response_mask,
            "instruction":    item["instruction"],
            "response_text":  item.get("response_text", item.get("response", "")),
            "weight":         torch.tensor(item["weight"], dtype=torch.float32),
            "role":           item["role"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_L_tok(logits: torch.Tensor, response_mask: torch.Tensor,
                  vocab_ids: set, vocab_size: int) -> torch.Tensor:
    """
    L_tok: penalise probability mass on out-of-vocabulary tokens
    in the response positions. Differentiable.
    """
    if not vocab_ids:
        return torch.tensor(0.0, device=logits.device)
    oov_ids = [i for i in range(vocab_size) if i not in vocab_ids]
    if not oov_ids:
        return torch.tensor(0.0, device=logits.device)
    oov_t   = torch.tensor(oov_ids, device=logits.device)
    probs   = F.softmax(logits[:, :-1, :], dim=-1)
    oov_p   = probs[:, :, oov_t].sum(dim=-1)
    mask    = response_mask[:, 1:].float()
    denom   = mask.sum().clamp(min=1)
    return (oov_p * mask).sum() / denom

def grpo_advantages(rewards: list) -> list:
    if len(rewards) < 2:
        return rewards
    mu  = statistics.mean(rewards)
    std = statistics.stdev(rewards)
    if std < 1e-8:
        return [0.0] * len(rewards)
    return [(r - mu) / std for r in rewards]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CURRICULUM — loss schedule (unchanged logic, cleaner interface)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LossCurriculumConfig:
    phase1_frac: float = 1/3
    phase2_frac: float = 2/3
    ramp_steps:  int   = 50

def loss_curriculum_phase(
    epoch: int, total_epochs: int,
    cfg: SCOPEConfig,
    cur: Optional[LossCurriculumConfig],
) -> Tuple[str, bool, bool, bool, float, float, float]:
    """Return (phase_name, use_tok, use_phr, use_cfg, lam_tok, lam_phr, lam_cfg)."""
    if cur is None:
        return ("no-curriculum",
                cfg.use_ltok, cfg.use_lphr, cfg.use_lcfg,
                cfg.lambda_tok, cfg.lambda_phr, cfg.lambda_cfg)

    end_p1 = max(1, math.ceil(cur.phase1_frac * total_epochs))
    end_p2 = max(end_p1 + 1, math.ceil(cur.phase2_frac * total_epochs))

    if epoch < end_p1:
        return ("Phase-1:CE", False, False, False, 0., 0., 0.)
    elif epoch < end_p2:
        use_tok = cfg.use_ltok
        return ("Phase-2:CE+tok", use_tok, False, False,
                cfg.lambda_tok if use_tok else 0., 0., 0.)
    else:
        return ("Phase-3:full",
                cfg.use_ltok, cfg.use_lphr, cfg.use_lcfg,
                cfg.lambda_tok if cfg.use_ltok else 0.,
                cfg.lambda_phr if cfg.use_lphr else 0.,
                cfg.lambda_cfg if cfg.use_lcfg else 0.)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. GRADNORM
# ═══════════════════════════════════════════════════════════════════════════════

class GradNorm:
    def __init__(self, initial_weights: list, alpha: float, device: torch.device):
        self.weights  = torch.nn.Parameter(
            torch.tensor(initial_weights, dtype=torch.float32, device=device))
        self.alpha    = alpha
        self.opt      = torch.optim.Adam([self.weights], lr=1e-3)
        self._L0      = None

    def step(self, losses: list, shared_params: list):
        losses_t = torch.stack([l if isinstance(l, torch.Tensor)
                                 else torch.tensor(l, device=self.weights.device)
                                 for l in losses])
        if self._L0 is None:
            self._L0 = losses_t.detach().clamp(min=1e-8)
        w = F.softmax(self.weights, dim=0) * len(losses)

        # Compute per-loss gradient norms (one scalar per loss term)
        # by computing grad of each weighted loss w.r.t. shared_params separately.
        grad_norms = []
        for i, (wi, li) in enumerate(zip(w, losses_t)):
            try:
                grads_i = torch.autograd.grad(
                    wi * li, shared_params,
                    retain_graph=True, allow_unused=True,
                    create_graph=False,
                )
                # Aggregate across all shared params: single L2 norm
                g_norm = torch.stack([
                    g.norm() if g is not None
                    else torch.tensor(0., device=self.weights.device)
                    for g in grads_i
                ]).norm()
            except Exception:
                g_norm = torch.tensor(0., device=self.weights.device)
            grad_norms.append(g_norm)

        grad_norms = torch.stack(grad_norms)   # shape: (n_losses,) == weights.shape ✓
        mean_gn = grad_norms.mean()
        ri = (losses_t.detach() / self._L0).clamp(min=1e-8)
        ri_bar = ri / ri.mean()
        target = (mean_gn * ri_bar ** self.alpha).detach()
        gn_loss = F.l1_loss(grad_norms, target)
        self.opt.zero_grad()
        gn_loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.weights.clamp_(min=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SEMANTIC METRICS (simplified — Slot-F1, DA-F1, Hallucination)
# ═══════════════════════════════════════════════════════════════════════════════

ATC_SLOT_PATTERNS = {
    'callsign':  re.compile(r'\b([A-Z]{2,}\s+(?:[A-Z0-9]+\s*)+)(?=\s+(?:TURN|DESCEND|CLIMB|MAINTAIN|CONTACT|CLEARED|SQUAWK|RADAR|TRAFFIC|SURFACE|REPORT|CONFIRM|SAY|RESUME|HOLD|EXPECT|CROSS|LINE|VACATE|TAXI|CANCEL|TAKEOFF|GO|CONTINUE|IDENT|STOP|DIRECT|FLY|HEADING|FREQUENCY|INFORMATION|ROGER|WILCO|AFFIRM|NEGATIVE|UNABLE|STANDBY|CORRECTION|GOOD|THANK|READBACK))'),
    'altitude':  re.compile(r'\b(\d{1,2}\s*(?:THOUSAND|HUNDRED)|FLIGHT\s+LEVEL\s+\w+)\b'),
    'heading':   re.compile(r'\bHEADING\s+(\w+\s+\w+\s+\w+|\w+\s+\w+|\w+)\b'),
    'frequency': re.compile(r'\b(\d{3}[\s\.]+\d+)\b'),
}

DA_PATTERNS = [
    ('clearance',  re.compile(r'\b(CLEARED|CLEARANCE|LINE\s+UP|TAKEOFF|TAKE-OFF)\b')),
    ('readback',   re.compile(r'\b(WILCO|AFFIRM|ROGER|READBACK\s+CORRECT)\b')),
    ('hold',       re.compile(r'\b(HOLD|STANDBY|UNABLE)\b')),
    ('advisory',   re.compile(r'\b(TRAFFIC|SURFACE\s+WIND|INFORMATION)\b')),
    ('handoff',    re.compile(r'\b(CONTACT|FREQUENCY\s+CHANGE|GOOD\s+DAY)\b')),
    ('correction', re.compile(r'\b(CORRECTION|SAY\s+AGAIN)\b')),
]

ENTITY_PATTERNS = [
    re.compile(r'\b[A-Z]{2,3}\s+\d+\b'),                    # callsign
    re.compile(r'\b(?:FL|FLIGHT\s+LEVEL)\s+\d+\b'),         # flight level
    re.compile(r'\b\d{3}[\s\.]+\d+\b'),                     # frequency
]

def extract_slots(text: str) -> dict:
    t = text.upper()
    slots = {}
    for name, pat in ATC_SLOT_PATTERNS.items():
        m = pat.search(t)
        slots[name] = m.group(1).strip() if m else None
    return slots

def predict_da(text: str) -> str:
    t = text.upper()
    for label, pat in DA_PATTERNS:
        if pat.search(t):
            return label
    return 'other'

def slot_f1(ref_slots: dict, gen_slots: dict) -> float:
    keys = set(ref_slots) | set(gen_slots)
    if not keys:
        return 1.0
    hits = sum(1 for k in keys
               if ref_slots.get(k) and gen_slots.get(k)
               and ref_slots[k].split() == gen_slots[k].split())
    return hits / len(keys)

def da_f1(ref_da: str, gen_da: str) -> float:
    return 1.0 if ref_da == gen_da else 0.0

def hallucination(gen: str, ref: str, req: str) -> float:
    context = (ref + ' ' + req).upper()
    gen_u   = gen.upper()
    for pat in ENTITY_PATTERNS:
        for m in pat.finditer(gen_u):
            entity = m.group(0).strip()
            if entity not in context:
                return 1.0
    return 0.0

def compute_semantic_metrics(examples: list, bertscore_model: str = "bert-base-uncased") -> dict:
    sf1_vals, da_vals, hall_vals, bs_vals = [], [], [], []
    for ex in examples:
        ref, gen = ex['reference'].upper(), ex['generated'].upper()
        req = ex.get('request', '').upper()
        sf1_vals.append(slot_f1(extract_slots(ref), extract_slots(gen)))
        da_vals.append(da_f1(predict_da(ref), predict_da(gen)))
        hall_vals.append(hallucination(gen, ref, req))
    if BERTSCORE_AVAILABLE and examples:
        refs = [e['reference'] for e in examples]
        gens = [e['generated'] for e in examples]
        try:
            _, _, F = bert_score_fn(gens, refs, model_type=bertscore_model,
                                    verbose=False)
            bs_vals = F.tolist()
        except Exception:
            bs_vals = [0.0] * len(examples)
    else:
        bs_vals = [0.0] * len(examples)

    def safe_mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    return {
        'slot_f1':    safe_mean(sf1_vals),
        'da_f1':      safe_mean(da_vals),
        'halluc_pct': safe_mean(hall_vals),
        'bertscore':  safe_mean(bs_vals),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 11. CHECKPOINT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def verify_checkpoint(ckpt_dir: str) -> None:
    import time
    ckpt = Path(ckpt_dir)
    try:
        os.sync()
    except Exception:
        pass
    time.sleep(1)
    shards = list(ckpt.glob("*.safetensors"))
    if not shards:
        return
    bad = [s.name for s in shards if s.stat().st_size < 1_000_000]
    if bad:
        raise RuntimeError(f"Truncated checkpoint shards: {bad}")
    gb = sum(s.stat().st_size for s in shards) / 1e9
    print(f"  ✓ Checkpoint: {len(shards)} shard(s), {gb:.2f} GB")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg: SCOPEConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device  = _tensor_device()
    n_gpus  = _n_gpus()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging setup ──────────────────────────────────────────────────────────
    log_path = out_dir / f"{cfg.run_name}.log"
    tee      = TeeLogger(str(log_path))

    step_csv  = StepCSVWriter(str(out_dir / f"{cfg.run_name}_steps.csv"))  if cfg.csv_log_steps  else None
    epoch_csv = EpochCSVWriter(str(out_dir / f"{cfg.run_name}_epochs.csv")) if cfg.csv_log_epochs else None

    tb_dir = out_dir / cfg.tensorboard_dir / cfg.run_name
    writer = SummaryWriter(log_dir=str(tb_dir)) if TB_AVAILABLE else None
    if writer:
        print(f"TensorBoard: tensorboard --logdir {tb_dir}")

    # Log config
    with open(out_dir / f"{cfg.run_name}_config.json", 'w') as f:
        import dataclasses
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    print(f"\n{'═'*65}")
    print(f"  SCOPE v2 — {cfg.run_name}")
    print(f"  Model:    {cfg.model_name}")
    print(f"  Domain:   {cfg.domain}")
    print(f"  Seed:     {cfg.seed}")
    print(f"  Data curriculum: {cfg.data_curriculum}")
    print(f"  Loss curriculum: {'enabled' if cfg.loss_curriculum else 'disabled'}")
    print(f"  λ: ce={cfg.lambda_ce} tok={cfg.lambda_tok} phr={cfg.lambda_phr} cfg={cfg.lambda_cfg}")
    print(f"  GPUs: {n_gpus}  Device: {device}")
    print(f"{'═'*65}\n")

    # Log hparams to TensorBoard
    if writer:
        writer.add_hparams(
            {k: str(v) for k, v in [
                ('model', cfg.model_name), ('seed', cfg.seed),
                ('lr', cfg.lr), ('batch_size', cfg.batch_size),
                ('grad_accum', cfg.grad_accum),
                ('lambda_ce', cfg.lambda_ce), ('lambda_tok', cfg.lambda_tok),
                ('lambda_phr', cfg.lambda_phr), ('lambda_cfg', cfg.lambda_cfg),
                ('data_curriculum', cfg.data_curriculum),
                ('loss_curriculum', cfg.loss_curriculum),
            ]},
            metric_dict={'hparams/placeholder': 0.0},
        )

    # ── Artefacts ──────────────────────────────────────────────────────────────
    print("Loading artefacts...")
    vocab      = load_whitelist(Path(cfg.vocab_path))
    ngram_wl   = load_ngram_whitelist(Path(cfg.phrase_path))
    parsers    = load_grammar_parsers(cfg.grammar_path) if cfg.use_lcfg else None
    print(f"  V_ATC: {len(vocab)} words | "
          f"P_ATC: {sum(len(v) for v in ngram_wl.values())} n-grams | "
          f"G_ATC: {'loaded (role-gated)' if parsers else 'disabled'}")

    # ── Model + tokeniser ──────────────────────────────────────────────────────
    print(f"\nLoading {cfg.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = _load_model(cfg.model_name,
                        gradient_checkpointing=cfg.gradient_checkpointing)
    vocab_ids = build_vocab_ids(tokenizer, vocab)
    print(f"  Vocab IDs: {len(vocab_ids)} / {tokenizer.vocab_size}")

    # ── Data loading (FORMAT B) ────────────────────────────────────────────────
    print("\nLoading data (FORMAT B)...")
    if cfg.train_data and cfg.val_data:
        all_train = load_jsonl(cfg.train_data)
        all_val   = load_jsonl(cfg.val_data)
    else:
        all_pairs = load_jsonl(cfg.data_path)
        random.shuffle(all_pairs)
        n       = len(all_pairs)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        all_train = all_pairs[:n_train]
        all_val   = all_pairs[n_train:n_train+n_val]

    # Filter unintelligible always
    all_train = [p for p in all_train if not p.get('resp_unintelligible')]
    all_val   = [p for p in all_val   if not p.get('resp_unintelligible')]

    # Strategy C: exclude bronze permanently
    if cfg.data_curriculum == 'C':
        all_train = [p for p in all_train
                     if p.get('quality_tier', 'bronze') in ('gold', 'silver')]
        print(f"  Strategy C: gold+silver only — {len(all_train)} train pairs")

    tiers = defaultdict(int)
    for p in all_train:
        tiers[p.get('quality_tier', 'bronze')] += 1
    print(f"  Train: {len(all_train)} pairs "
          f"(gold={tiers['gold']}, silver={tiers['silver']}, bronze={tiers['bronze']})")
    print(f"  Val:   {len(all_val)} pairs")

    def make_loader(pairs, shuffle=True):
        ds = AtcDatasetB(pairs, tokenizer, cfg)
        def collate(batch):
            result = {}
            for key in batch[0]:
                vals = [b[key] for b in batch]
                if isinstance(vals[0], torch.Tensor):
                    result[key] = torch.stack(vals)
                else:
                    result[key] = vals
            return result
        return DataLoader(ds, batch_size=cfg.batch_size,
                          shuffle=shuffle, collate_fn=collate)

    val_dl   = make_loader(all_val, shuffle=False)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    # (Use strategy B or A phase-1 estimate for total_steps)
    total_steps_est = (len(all_train) // cfg.batch_size) * cfg.epochs
    if cfg.warmup_ratio > 0:
        warmup = max(1, int(cfg.warmup_ratio * total_steps_est))
    else:
        warmup = cfg.warmup_steps or 100

    if _n_gpus() >= 2 or cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=cfg.lr, weight_decay=0.01)
            print("Optimizer: AdamW8bit")
        except ImportError:
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup, num_training_steps=total_steps_est)

    # ── GradNorm ──────────────────────────────────────────────────────────────
    gradnorm = None
    shared_params = None
    lam_ce  = cfg.lambda_ce
    lam_tok = cfg.lambda_tok
    lam_phr = cfg.lambda_phr
    lam_cfg = cfg.lambda_cfg

    if cfg.use_gradnorm:
        try:
            shared_params = list(model.model.layers[-1].parameters())
        except AttributeError:
            try:
                shared_params = list(model.transformer.h[-1].parameters())
            except AttributeError:
                shared_params = list(model.parameters())[-10:]
        gradnorm = GradNorm([lam_ce, lam_tok, lam_phr, lam_cfg],
                             cfg.gradnorm_alpha, device)
        print(f"GradNorm enabled (α={cfg.gradnorm_alpha})")

    # ── Loss curriculum config ────────────────────────────────────────────────
    cur_cfg = LossCurriculumConfig(
        phase1_frac=cfg.loss_curriculum_phase1,
        phase2_frac=cfg.loss_curriculum_phase2,
        ramp_steps=cfg.loss_curriculum_ramp,
    ) if cfg.loss_curriculum else None

    # ── Training state ────────────────────────────────────────────────────────
    best_metric  = 0.0
    no_improve   = 0
    global_step  = 0
    history      = []

    # Grammar rule failure counter
    rule_fail_counts: Dict[str, int] = defaultdict(int)

    print(f"\nTraining for {cfg.epochs} epochs...\n")

    for epoch in range(cfg.epochs):
        # ── Data curriculum phase ──────────────────────────────────────────
        if cfg.data_curriculum == 'A':
            data_phase = 1 if epoch < int(cfg.data_phase1_frac * cfg.epochs) else 2
            epoch_pairs = filter_training_pairs(all_train, 'A', data_phase)
            data_phase_label = f"A-phase{data_phase}"
        else:
            epoch_pairs = all_train
            data_phase  = 1
            data_phase_label = cfg.data_curriculum

        train_dl = make_loader(epoch_pairs, shuffle=True)

        # ── Loss curriculum phase ──────────────────────────────────────────
        phase_name, e_use_ltok, e_use_lphr, e_use_lcfg, \
        e_lam_tok, e_lam_phr, e_lam_cfg = loss_curriculum_phase(
            epoch, cfg.epochs, cfg, cur_cfg)

        print(f"\n{'─'*65}")
        print(f"  Epoch {epoch+1}/{cfg.epochs} | Loss: {phase_name} | Data: {data_phase_label}")
        print(f"  Active losses: CE=✓ tok={'✓' if e_use_ltok else '✗'} "
              f"phr={'✓' if e_use_lphr else '✗'} cfg={'✓' if e_use_lcfg else '✗'}")
        print(f"  λ: ce={lam_ce:.3f} tok={e_lam_tok:.3f} phr={e_lam_phr:.3f} cfg={e_lam_cfg:.3f}")
        print(f"  Data pairs: {len(epoch_pairs)}")
        print(f"{'─'*65}")

        if writer:
            writer.add_text('curriculum/loss_phase', phase_name, epoch)
            writer.add_text('curriculum/data_phase', data_phase_label, epoch)

        # ── Ramp state for this epoch ─────────────────────────────────────
        _prev_phase   = getattr(train, '_prev_phase', None)
        _phase_changed = (phase_name != _prev_phase)
        train._prev_phase = phase_name
        _ramp_counter = 0

        model.train()
        epoch_losses = defaultdict(float)
        n_batches    = 0

        for step, batch in enumerate(train_dl):

            # Ramp new lambdas at phase transitions
            if _phase_changed and cur_cfg and cur_cfg.ramp_steps > 0:
                ramp = min(1.0, _ramp_counter / max(1, cur_cfg.ramp_steps))
                _step_lam_tok = ramp * e_lam_tok
                _step_lam_phr = ramp * e_lam_phr
                _step_lam_cfg = ramp * e_lam_cfg
                _ramp_counter += 1
            else:
                _step_lam_tok = e_lam_tok
                _step_lam_phr = e_lam_phr
                _step_lam_cfg = e_lam_cfg

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            response_mask  = batch["response_mask"].to(device)
            weights        = batch["weight"].to(device)    # tier weights
            roles          = batch["role"]                 # list of role strings

            opt.zero_grad()

            # ── CE loss ────────────────────────────────────────────────────
            out    = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels)
            L_ce   = out.loss   # base CE (mean over valid tokens)

            # Apply tier weights for strategy B
            if cfg.data_curriculum == 'B':
                # Recompute weighted CE
                logits_ce = out.logits[:, :-1, :].contiguous()
                tgts_ce   = labels[:, 1:].contiguous()
                per_tok   = F.cross_entropy(
                    logits_ce.view(-1, logits_ce.size(-1)),
                    tgts_ce.view(-1),
                    ignore_index=-100,
                    reduction='none',
                ).view(labels.size(0), -1)
                valid_mask = (tgts_ce != -100).float()
                per_sample = (per_tok * valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1).clamp(min=1)
                L_ce = (per_sample * weights).mean()

            logits = out.logits

            # ── L_tok ──────────────────────────────────────────────────────
            L_tok_val = torch.tensor(0., device=device)
            if e_use_ltok:
                L_tok_val = compute_L_tok(logits, response_mask,
                                          vocab_ids, tokenizer.vocab_size)

            # ── L_phr + L_cfg (GRPO every other step) ─────────────────────
            _phr_display = 0.0
            _cfg_display = 0.0
            _phr_rule_failures = defaultdict(int)

            if (e_use_lphr or e_use_lcfg) and (step % 2 == 0):
                model.eval()
                all_generated    = []
                phr_rewards_list = []
                cfg_rewards_list = []

                with torch.no_grad():
                    for _ in range(cfg.M_samples):
                        out_g = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=cfg.max_new_tok,
                            do_sample=True,
                            temperature=1.5,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        gen  = out_g[:, input_ids.size(1):]
                        all_generated.append(gen)
                        text = tokenizer.decode(gen[0].tolist(),
                                                skip_special_tokens=True)
                        toks = text.upper().split()
                        role = roles[0] if roles else 'unified'

                        if e_use_lphr:
                            phr_rewards_list.append(compute_cphr(toks, ngram_wl))

                        if e_use_lcfg and parsers:
                            cfg_r = compute_ccfg_partial(text, parsers, role)
                            cfg_rewards_list.append(cfg_r)
                            # Grammar rule diagnostics
                            if cfg_r < 1.0 and cfg.log_grammar_rules:
                                rule = diagnose_parse_failure(text, parsers, role)
                                _phr_rule_failures[rule] += 1
                                rule_fail_counts[rule]   += 1

                model.train()

                phr_adv = grpo_advantages(phr_rewards_list) if phr_rewards_list else []
                cfg_adv = grpo_advantages(cfg_rewards_list) if cfg_rewards_list else []

                for m_idx, gen_m in enumerate(all_generated):
                    if gen_m.size(1) == 0:
                        continue
                    phr_adv_m = phr_adv[m_idx] if m_idx < len(phr_adv) else 0.
                    cfg_adv_m = cfg_adv[m_idx] if m_idx < len(cfg_adv) else 0.
                    if abs(phr_adv_m) < 1e-8 and abs(cfg_adv_m) < 1e-8:
                        continue

                    full     = torch.cat([input_ids, gen_m], dim=1)
                    attn_g   = torch.ones(input_ids.size(0), full.size(1) - 1,
                                          dtype=torch.long, device=device)
                    logits_g = model(full[:, :-1], attention_mask=attn_g).logits
                    logits_g = logits_g[:, input_ids.size(1) - 1:, :]
                    log_p    = F.log_softmax(logits_g, dim=-1)
                    idx_t    = gen_m.unsqueeze(-1).clamp(0, log_p.size(-1) - 1)
                    seq_lp   = log_p.gather(-1, idx_t).squeeze(-1).mean(dim=-1)

                    grpo_m = torch.tensor(0., device=device)
                    if e_use_lphr and abs(phr_adv_m) > 1e-8:
                        grpo_m = grpo_m + (
                            -phr_adv_m * _step_lam_phr * seq_lp.mean() / cfg.M_samples)
                    if e_use_lcfg and parsers and abs(cfg_adv_m) > 1e-8:
                        grpo_m = grpo_m + (
                            -cfg_adv_m * _step_lam_cfg * seq_lp.mean() / cfg.M_samples)
                    if grpo_m.grad_fn is not None:
                        grpo_m.backward()
                    del full, attn_g, logits_g, log_p, idx_t, seq_lp, grpo_m

                if phr_rewards_list:
                    _phr_display = sum(phr_rewards_list) / len(phr_rewards_list)
                if cfg_rewards_list:
                    _cfg_display = sum(cfg_rewards_list) / len(cfg_rewards_list)

            # ── GradNorm ───────────────────────────────────────────────────
            # GradNorm: only run when at least 2 losses are active
            # (Phase 1 has CE only — no rebalancing needed)
            _n_active = sum([1, e_use_ltok, e_use_lphr, e_use_lcfg])
            if gradnorm and _n_active >= 2 and (step % cfg.gradnorm_update_freq == 0):
                l_phr_proxy = torch.tensor(
                    1. - _phr_display, device=device, requires_grad=True)
                l_cfg_proxy = torch.tensor(
                    1. - _cfg_display, device=device, requires_grad=True)
                try:
                    gradnorm.step([L_ce, L_tok_val, l_phr_proxy, l_cfg_proxy],
                                  shared_params)
                    ws = F.softmax(gradnorm.weights, dim=0).tolist()
                    lam_ce, lam_tok, lam_phr, lam_cfg = [
                        w * len(ws) * v for w, v in zip(ws, [
                            cfg.lambda_ce, cfg.lambda_tok,
                            cfg.lambda_phr, cfg.lambda_cfg])]
                except Exception as e:
                    pass  # GradNorm failure is non-fatal — keep current lambdas

            # ── CE + L_tok backward ────────────────────────────────────────
            L_ce_tok = lam_ce * L_ce + _step_lam_tok * L_tok_val
            L_ce_tok.backward()
            L_total = L_ce_tok.detach()

            # ── Gradient clip + optimiser step ────────────────────────────
            grad_norm_val = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip).item()
            opt.step()
            scheduler.step()

            epoch_losses["ce"]    += L_ce.item()
            epoch_losses["tok"]   += L_tok_val.item()
            epoch_losses["phr"]   += _phr_display
            epoch_losses["cfg"]   += _cfg_display
            epoch_losses["total"] += L_total.item()
            n_batches += 1
            global_step += 1

            lr_now = scheduler.get_last_lr()[0]

            # ── Step logging ───────────────────────────────────────────────
            if global_step % cfg.log_step_freq == 0:
                if writer:
                    writer.add_scalar('train/loss_ce',    L_ce.item(),      global_step)
                    writer.add_scalar('train/loss_tok',   L_tok_val.item(), global_step)
                    writer.add_scalar('train/loss_phr',   _phr_display,     global_step)
                    writer.add_scalar('train/loss_cfg',   _cfg_display,     global_step)
                    writer.add_scalar('train/loss_total', L_total.item(),   global_step)
                    writer.add_scalar('train/reward_phr', _phr_display,     global_step)
                    writer.add_scalar('train/reward_cfg', _cfg_display,     global_step)
                    writer.add_scalar('train/lr',          lr_now,           global_step)
                    writer.add_scalar('train/grad_norm',   grad_norm_val,    global_step)
                    writer.add_scalar('lambdas/ce',  lam_ce,  global_step)
                    writer.add_scalar('lambdas/tok', lam_tok, global_step)
                    writer.add_scalar('lambdas/phr', lam_phr, global_step)
                    writer.add_scalar('lambdas/cfg', lam_cfg, global_step)
                    # Grammar rule failure breakdown
                    if cfg.log_grammar_rules and _phr_rule_failures:
                        for rule, count in _phr_rule_failures.items():
                            writer.add_scalar(f'grammar_rules/{rule}', count, global_step)

                if step_csv:
                    step_csv.write({
                        'global_step': global_step, 'epoch': epoch+1, 'step': step,
                        'phase': phase_name,
                        'loss_ce':    round(L_ce.item(), 6),
                        'loss_tok':   round(L_tok_val.item(), 6),
                        'loss_phr':   round(_phr_display, 6),
                        'loss_cfg':   round(_cfg_display, 6),
                        'loss_total': round(L_total.item(), 6),
                        'reward_phr': round(_phr_display, 6),
                        'reward_cfg': round(_cfg_display, 6),
                        'lam_ce':  round(lam_ce, 4),
                        'lam_tok': round(lam_tok, 4),
                        'lam_phr': round(lam_phr, 4),
                        'lam_cfg': round(lam_cfg, 4),
                        'lr':       round(lr_now, 8),
                        'grad_norm': round(grad_norm_val, 4),
                    })

                print(f"  E{epoch+1} S{step}/{len(train_dl)} gs={global_step} | "
                      f"CE={L_ce.item():.3f} tok={L_tok_val.item():.4f} "
                      f"phr={_phr_display:.4f} cfg={_cfg_display:.4f} | "
                      f"lr={lr_now:.2e} gn={grad_norm_val:.2f}")

        # ── Grammar rule summary per epoch ─────────────────────────────────
        if cfg.log_grammar_rules and rule_fail_counts:
            print(f"\n  Grammar rule failure summary (epoch {epoch+1}):")
            for rule, cnt in sorted(rule_fail_counts.items(),
                                    key=lambda x: -x[1])[:10]:
                print(f"    {rule:<30} {cnt:>6}")
            if writer:
                for rule, cnt in rule_fail_counts.items():
                    writer.add_scalar(f'grammar_epoch/{rule}', cnt, epoch+1)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_ctok, val_cphr, val_ccfg, val_examples = [], [], [], []

        with torch.no_grad():
            for vbatch in val_dl:
                instructions   = vbatch["instruction"]
                response_texts = vbatch["response_text"]
                v_roles        = vbatch["role"]
                for i in range(len(instructions)):
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
                    prompt_len = prompt_tok["input_ids"].size(1)
                    gen  = out_v[0, prompt_len:]
                    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
                    toks = text.upper().split()
                    role = v_roles[i] if v_roles else 'unified'

                    c_tok = compute_ctok(toks, vocab)
                    c_phr = compute_cphr(toks, ngram_wl)
                    c_cfg = compute_ccfg_partial(text, parsers, role) if parsers else 0.0
                    val_ctok.append(c_tok)
                    val_cphr.append(c_phr)
                    val_ccfg.append(c_cfg)
                    val_examples.append({
                        'request':   instructions[i],
                        'reference': response_texts[i] if i < len(response_texts) else '',
                        'generated': text,
                    })

        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        v_ctok = safe_mean(val_ctok)
        v_cphr = safe_mean(val_cphr)
        v_ccfg = safe_mean(val_ccfg)
        v_cbar = (v_ctok + v_cphr + v_ccfg) / 3.0

        sem = compute_semantic_metrics(val_examples, cfg.bertscore_model)

        # Epoch average training losses
        avg = {k: v / max(1, n_batches) for k, v in epoch_losses.items()}

        print(f"\n  ── Epoch {epoch+1} results ──")
        print(f"  Train:  CE={avg['ce']:.4f} tok={avg['tok']:.4f} "
              f"phr={avg['phr']:.4f} cfg={avg['cfg']:.4f}")
        print(f"  Val:    C_tok={v_ctok:.4f} C_phr={v_cphr:.4f} "
              f"C_cfg={v_ccfg:.4f} C_bar={v_cbar:.4f}")
        print(f"  Sem:    Slot-F1={sem['slot_f1']:.4f} DA-F1={sem['da_f1']:.4f} "
              f"Hall%={sem['halluc_pct']:.4f} BERT={sem['bertscore']:.4f}")

        if writer:
            writer.add_scalar('val/c_tok',    v_ctok, epoch+1)
            writer.add_scalar('val/c_phr',    v_cphr, epoch+1)
            writer.add_scalar('val/c_cfg',    v_ccfg, epoch+1)
            writer.add_scalar('val/c_bar',    v_cbar, epoch+1)
            writer.add_scalar('val/slot_f1',  sem['slot_f1'],   epoch+1)
            writer.add_scalar('val/da_f1',    sem['da_f1'],     epoch+1)
            writer.add_scalar('val/halluc',   sem['halluc_pct'],epoch+1)
            writer.add_scalar('val/bertscore',sem['bertscore'], epoch+1)
            writer.add_scalar('train/avg_ce', avg['ce'],  epoch+1)
            writer.add_scalar('train/avg_tok',avg['tok'], epoch+1)
            writer.add_scalar('train/avg_phr',avg['phr'], epoch+1)
            writer.add_scalar('train/avg_cfg',avg['cfg'], epoch+1)

        if epoch_csv:
            epoch_csv.write({
                'epoch': epoch+1, 'phase': phase_name, 'data_phase': data_phase_label,
                'train_ce': round(avg['ce'],5), 'train_tok': round(avg['tok'],5),
                'train_phr': round(avg['phr'],5), 'train_cfg': round(avg['cfg'],5),
                'train_total': round(avg['total'],5),
                'val_ctok': round(v_ctok,4), 'val_cphr': round(v_cphr,4),
                'val_ccfg': round(v_ccfg,4), 'val_cbar': round(v_cbar,4),
                'slot_f1': round(sem['slot_f1'],4), 'da_f1': round(sem['da_f1'],4),
                'halluc_pct': round(sem['halluc_pct'],4),
                'bertscore':  round(sem['bertscore'],4),
                'lam_ce': round(lam_ce,4), 'lam_tok': round(lam_tok,4),
                'lam_phr': round(lam_phr,4), 'lam_cfg': round(lam_cfg,4),
            })

        # ── Checkpoint ──────────────────────────────────────────────────────
        metric = v_cbar if cfg.checkpoint_metric == 'c_bar' else v_ctok
        if metric > best_metric and sem['halluc_pct'] <= cfg.hallucination_threshold:
            best_metric = metric
            no_improve  = 0
            ckpt_path   = out_dir / "best"
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            verify_checkpoint(str(ckpt_path))
            print(f"  ✓ Best checkpoint saved ({cfg.checkpoint_metric}={metric:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{cfg.early_stop_patience or '∞'})")

        history.append({
            'epoch': epoch+1, 'phase': phase_name, 'data_phase': data_phase_label,
            **{f'train_{k}': round(v, 6) for k, v in avg.items()},
            'val_ctok': round(v_ctok,4), 'val_cphr': round(v_cphr,4),
            'val_ccfg': round(v_ccfg,4), 'val_cbar': round(v_cbar,4),
            **{k: round(v,4) for k,v in sem.items()},
            'lam_ce': round(lam_ce,4), 'lam_tok': round(lam_tok,4),
            'lam_phr': round(lam_phr,4), 'lam_cfg': round(lam_cfg,4),
        })

        if cfg.early_stop_patience and no_improve >= cfg.early_stop_patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    # ── Final grammar rule summary ─────────────────────────────────────────
    if cfg.log_grammar_rules and rule_fail_counts:
        print(f"\n{'═'*65}")
        print("  GRAMMAR RULE FAILURE SUMMARY (full training)")
        print("  (which G_ATC sections drive C_cfg loss the most)")
        total_fails = sum(rule_fail_counts.values())
        for rule, cnt in sorted(rule_fail_counts.items(), key=lambda x: -x[1]):
            pct = cnt / max(1, total_fails) * 100
            print(f"  {rule:<30} {cnt:>8}  ({pct:.1f}%)")
        rule_summary = dict(sorted(rule_fail_counts.items(), key=lambda x: -x[1]))
        with open(out_dir / f"{cfg.run_name}_grammar_failures.json", 'w') as f:
            json.dump(rule_summary, f, indent=2)

    # ── Save history ───────────────────────────────────────────────────────
    with open(out_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    if writer:
        writer.close()
    if step_csv:
        step_csv.close()
    if epoch_csv:
        epoch_csv.close()
    tee.close()

    print("\nTraining complete.")
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# 13. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCOPE v2 Training")
    parser.add_argument("--config",  default="",
                        help="Path to YAML or JSON config file (recommended)")
    # Override any config key via CLI
    parser.add_argument("--model",           default="")
    parser.add_argument("--data",            default="")
    parser.add_argument("--train_data",      default="")
    parser.add_argument("--val_data",        default="")
    parser.add_argument("--test_data",       default="")
    parser.add_argument("--output",          default="")
    parser.add_argument("--run_name",        default="")
    parser.add_argument("--seed",            type=int,   default=None)
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--batch_size",      type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--lambda_ce",       type=float, default=None)
    parser.add_argument("--lambda_tok",      type=float, default=None)
    parser.add_argument("--lambda_phr",      type=float, default=None)
    parser.add_argument("--lambda_cfg",      type=float, default=None)
    parser.add_argument("--data_curriculum", default=None, choices=['A','B','C'])
    parser.add_argument("--no_ltok",         action="store_true")
    parser.add_argument("--no_lphr",         action="store_true")
    parser.add_argument("--no_lcfg",         action="store_true")
    parser.add_argument("--no_loss_curriculum", action="store_true")
    parser.add_argument("--gradnorm",        action="store_true")
    parser.add_argument("--vocab_path",      default="")
    parser.add_argument("--phrase_path",     default="")
    parser.add_argument("--grammar",         default="")
    parser.add_argument("--grad_accum",      type=int,   default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--domain",          default=None, choices=["atc","smcp"])
    args = parser.parse_args()

    # Load config from file first, then override with CLI
    if args.config:
        cfg_dict = load_config(args.config)
        cfg = config_from_dict(cfg_dict)
    else:
        cfg = SCOPEConfig()

    # CLI overrides
    if args.model:           cfg.model_name      = args.model
    if args.data:            cfg.data_path        = args.data
    if args.train_data:      cfg.train_data       = args.train_data
    if args.val_data:        cfg.val_data         = args.val_data
    if args.test_data:       cfg.test_data        = args.test_data
    if args.output:          cfg.output_dir       = args.output
    if args.run_name:        cfg.run_name         = args.run_name
    if args.seed is not None:     cfg.seed        = args.seed
    if args.epochs is not None:   cfg.epochs      = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.lr is not None:       cfg.lr          = args.lr
    if args.lambda_ce  is not None: cfg.lambda_ce  = args.lambda_ce
    if args.lambda_tok is not None: cfg.lambda_tok = args.lambda_tok
    if args.lambda_phr is not None: cfg.lambda_phr = args.lambda_phr
    if args.lambda_cfg is not None: cfg.lambda_cfg = args.lambda_cfg
    if args.data_curriculum:  cfg.data_curriculum = args.data_curriculum
    if args.no_ltok:          cfg.use_ltok        = False
    if args.no_lphr:          cfg.use_lphr        = False
    if args.no_lcfg:          cfg.use_lcfg        = False
    if args.no_loss_curriculum: cfg.loss_curriculum = False
    if args.gradnorm:         cfg.use_gradnorm    = True
    if args.vocab_path:       cfg.vocab_path      = args.vocab_path
    if args.phrase_path:      cfg.phrase_path      = args.phrase_path
    if args.grammar:          cfg.grammar_path     = args.grammar
    if args.grad_accum is not None: cfg.grad_accum = args.grad_accum
    if args.gradient_checkpointing: cfg.gradient_checkpointing = True
    if args.domain:           cfg.domain           = args.domain

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("✓ HuggingFace authenticated")
        except Exception as e:
            print(f"WARNING: HF login: {e}")

    train(cfg)
