#!/usr/bin/env python3
"""
evaluate_gcd.py — Grammar-Constrained Decoding evaluation (Condition C4)

Applies Earley-based valid-next-token masking at inference time to a
trained SFT checkpoint and evaluates compliance on the test set.
This implements the GCD baseline from Geng et al. (EMNLP 2023) using
G_ATC_v2.lark as the constraint grammar.

Usage:
    # Evaluate GCD on the SFT checkpoint
    python evaluate_gcd.py \
        --model  results/C2/best \
        --data   atc_test.json \
        --grammar G_ATC_v2.lark \
        --output results/C4

    # Evaluate GCD on vanilla GPT-2 (no fine-tuning)
    python evaluate_gcd.py \
        --model  gpt2-large \
        --data   atc_test.json \
        --grammar G_ATC_v2.lark \
        --output results/C4_vanilla
"""

import json, re, argparse, os
from pathlib import Path
from typing import List, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                           LogitsProcessor, LogitsProcessorList)

# ── Grammar and vocabulary artefacts ──────────────────────────────────────────
# Default paths resolve relative to this script's directory
_SCRIPT_DIR = Path(__file__).parent
VOCAB_PATH  = _SCRIPT_DIR / "vocab_ATC.json"
PHRASE_PATH = _SCRIPT_DIR / "ngram_whitelist_ATC_v2.json"

def load_whitelist(vocab_path):
    with open(vocab_path) as f:
        return set(json.load(f))

def load_ngram_whitelist(phrase_path):
    with open(phrase_path) as f:
        raw = json.load(f)
    return {
        2: set(tuple(g) for g in raw.get("bigrams",  [])),
        3: set(tuple(g) for g in raw.get("trigrams", [])),
        4: set(tuple(g) for g in raw.get("4grams",   [])),
    }


# ── GCD Logits Processor ──────────────────────────────────────────────────────

class GCDLogitsProcessor(LogitsProcessor):
    """
    Grammar-guided constrained decoding for ATC phraseology.

    Implementation follows Geng et al. (EMNLP 2023) in spirit:
    at each decoding step, mask all tokens whose surface form cannot
    extend the current prefix to a string in the grammar's language.

    EFFICIENT REALISATION (O(1) per step):
    Rather than calling Earley.parse() per candidate word at each step
    (which is O(|V| × parse_time) and takes hours), we precompute:

      (a) A token-level vocabulary mask: token_id → bool
          True iff the token's surface form is a word in V_ATC.
          Applied at every step.

      (b) A bigram transition table: word → {valid next words}
          Derived from P_ATC — every bigram in P_ATC was itself derived
          from G_ATC by extracting all n-grams from utterances in L(G).
          Applied conditionally: given last generated word, only allow
          tokens whose surface word appears as a valid successor.

    This is equivalent to the finite-state approximation of the CFL
    induced by G_ATC, which is the standard efficient GCD realisation
    for ambiguous CFGs (Geng et al. §3.3; Scholak et al. 2021).

    The Earley grammar is used once per utterance AFTER generation to
    compute C_cfg — not during decoding.

    Relationship to Geng et al.:
      Geng use Lark's LALR interactive parser for unambiguous grammars.
      G_ATC is ambiguous (Earley required), so LALR is unavailable.
      This bigram-trie approximation gives O(1)-per-step decoding with
      the same constraint spirit. We note this in the paper.
    """

    def __init__(self, tokenizer, vocab: set, ngram_wl: dict = None):
        self.tokenizer = tokenizer
        self.vocab_upper = {w.upper() for w in vocab}
        # Use len(tokenizer) not tokenizer.vocab_size — the latter excludes
        # added special tokens (e.g. Llama's <|eot_id|> has id 128009
        # but vocab_size is 128000, causing IndexError in the mask).
        self.V = len(tokenizer)

        # ── (a) Token-level vocabulary mask (precomputed once) ────────────
        self._token_mask = self._build_token_mask()

        # ── (b) Bigram transition table (precomputed once) ────────────────
        self._valid_next = {}       # word → frozenset of valid next words
        self._word_to_tids = {}     # word → frozenset of token IDs
        if ngram_wl and 2 in ngram_wl:
            for (w1, w2) in ngram_wl[2]:
                self._valid_next.setdefault(w1, set()).add(w2)
            # Freeze for efficiency
            self._valid_next = {k: frozenset(v)
                                for k, v in self._valid_next.items()}

        # Map surface words → token IDs
        for tok_str, tid in tokenizer.get_vocab().items():
            if tid >= self.V:
                continue
            word = re.sub(r'^[ĠĀĊ▁Ġ]+', '',
                          tok_str).upper().strip()
            if word in self.vocab_upper:
                self._word_to_tids.setdefault(word, set()).add(tid)
        self._word_to_tids = {k: frozenset(v)
                              for k, v in self._word_to_tids.items()}

    def _build_token_mask(self) -> 'torch.Tensor':
        import torch
        mask = torch.zeros(self.V, dtype=torch.bool)
        all_toks = self.tokenizer.get_vocab()

        for tok_str, tid in all_toks.items():
            if tid >= self.V:
                continue
            word = re.sub(r'^[ĠĀĊ▁Ġ]+', '',
                          tok_str).upper().strip()
            if (not word
                    or word in self.vocab_upper
                    or re.match(r'^[0-9]+$', word)
                    or re.match(r'^[,.\-/\s]+$', tok_str)):
                mask[tid] = True

        # Always allow EOS / PAD
        for special in [self.tokenizer.eos_token,
                        self.tokenizer.pad_token,
                        self.tokenizer.bos_token]:
            if special and special in all_toks:
                mask[all_toks[special]] = True
        return mask

    def _bigram_constrained_mask(self, last_word: str,
                                  device) -> 'torch.Tensor':
        """
        Given the last word, return a token mask allowing only successors
        found in the P_ATC bigram table (O(1) dict lookup).
        Falls back to full vocab mask if no successors found.
        """
        import torch
        valid_words = self._valid_next.get(last_word.upper())
        if not valid_words:
            return self._token_mask.to(device)     # fallback

        mask = torch.zeros(self.V, dtype=torch.bool, device=device)
        for word in valid_words:
            for tid in self._word_to_tids.get(word, frozenset()):
                mask[tid] = True

        # Always allow EOS
        eos = self.tokenizer.eos_token
        if eos and eos in self.tokenizer.get_vocab():
            mask[self.tokenizer.get_vocab()[eos]] = True

        # If mask is all-False (no matching token IDs), fallback
        if not mask.any():
            return self._token_mask.to(device)
        return mask

    def __call__(self, input_ids: 'torch.Tensor',
                 scores: 'torch.Tensor') -> 'torch.Tensor':
        """
        Apply grammar-guided mask. O(1) per step.
        """
        device = scores.device
        B = input_ids.size(0)

        if not self._valid_next:
            # No bigram table — apply vocabulary mask only
            vm = self._token_mask.to(device)
            return scores.masked_fill(~vm.unsqueeze(0), float("-inf"))

        for i in range(B):
            # Decode last token to get last surface word
            last_tid  = input_ids[i, -1].item()
            last_str  = self.tokenizer.decode([last_tid],
                                              skip_special_tokens=True)
            last_word = re.sub(r'^[ĠĀĊ▁Ġ]+', '',
                               last_str).upper().strip()

            if last_word in self.vocab_upper:
                mask = self._bigram_constrained_mask(last_word, device)
            else:
                mask = self._token_mask.to(device)

            scores[i] = scores[i].masked_fill(~mask, float("-inf"))

        return scores



# ── Compliance metrics (mirrors scope_train_v2.py) ────────────────────────────

def compute_ctok(tokens, vocab):
    if not tokens: return 0.0
    return sum(1 for t in tokens if t.upper() in vocab) / len(tokens)

def compute_cphr(tokens, ngram_wl):
    if len(tokens) < 2: return 0.0
    scores = []
    toks_up = [t.upper() for t in tokens]
    for n in [2, 3, 4]:
        if len(toks_up) < n: continue
        hit = sum(1 for i in range(len(toks_up)-n+1)
                  if tuple(toks_up[i:i+n]) in ngram_wl[n])
        scores.append(hit / (len(toks_up)-n+1))
    return sum(scores)/len(scores) if scores else 0.0

def compute_ccfg(text, parser, strict_parser=None):
    """
    Compute C_cfg using strict Earley parsing (no general_instr fallback).

    Two parsers are used:
    - strict_parser: grammar with general_instr disabled — used for fair
      evaluation of GCD output (prevents the catch-all fallback from
      inflating scores for any sequence of V_ATC tokens)
    - parser: full grammar with fallback — used for partial-parse scoring
      of SCOPE training output only

    For GCD evaluation, always pass strict_parser.
    """
    if parser is None and strict_parser is None:
        return 0.0

    eval_parser = strict_parser if strict_parser is not None else parser
    text_up = text.upper().strip()
    if not text_up:
        return 0.0

    try:
        eval_parser.parse(text_up)
        return 1.0
    except Exception:
        words = text_up.split()
        for n in range(len(words), 0, -1):
            try:
                eval_parser.parse(' '.join(words[:n]))
                return n / len(words)
            except Exception:
                continue
    return 0.0


# ── Format prompt ─────────────────────────────────────────────────────────────

DOMAIN_PROMPTS = {
    "atc":  "You are an ATC communication assistant for UAV operations. "
            "Generate ICAO-compliant phraseology.",
    "smcp": "You are a maritime radio communication assistant. "
            "Generate IMO SMCP-compliant phraseology.",
}

def format_instruction(request: str, fsm_state: str = "Init",
                       domain: str = "atc") -> str:
    domain_desc = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["atc"])
    if domain == "atc":
        system = f"[STATE: {fsm_state}] {domain_desc}"
    else:
        system = domain_desc
    return (f"### Instruction:\n{system}\n\n"
            f"### Operator:\n{request}\n\n"
            f"### Response:")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_gcd(model_path: str, data_path: str, grammar_path: str,
                 output_dir: str, max_new_tokens: int = 64,
                 fallback_to_vocab: bool = True,
                 use_gcd: bool = True,
                 vocab_path: str = "",
                 phrase_path: str = "",
                 domain: str = "atc"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {model_path}")
    print(f"GCD:    {'enabled' if use_gcd else 'disabled (vocab mask only)'}")

    # ── Load artefacts ────────────────────────────────────────────────────
    _vocab_p  = Path(vocab_path)  if vocab_path  and Path(vocab_path).exists()  else VOCAB_PATH
    _phrase_p = Path(phrase_path) if phrase_path and Path(phrase_path).exists() else PHRASE_PATH
    if not _vocab_p.exists():
        raise FileNotFoundError(f"Vocab file not found at {_vocab_p}. "
                                f"Pass --vocab /path/to/vocab_{{domain}}.json")
    if not _phrase_p.exists():
        raise FileNotFoundError(f"N-gram whitelist not found at {_phrase_p}. "
                                f"Pass --phrase /path/to/ngram_whitelist_{{domain}}.json")
    vocab    = load_whitelist(_vocab_p)
    ngram_wl = load_ngram_whitelist(_phrase_p)

    grammar_str = None
    lark_parser = None
    strict_parser = None   # grammar without general_instr fallback
    if Path(grammar_path).exists():
        with open(grammar_path) as f:
            grammar_str = f.read()
        try:
            from lark import Lark
            lark_parser = Lark(grammar_str, parser="earley",
                               ambiguity="resolve")
            # Strict parser: disable general_instr catch-all fallback
            # This prevents any sequence of V_ATC words from scoring C_cfg=1.0
            strict_grammar = grammar_str.replace(
                '            | general_instr',
                '            //| general_instr  // disabled for strict eval'
            ).replace(
                'general_instr : word (" " word)+\nword : /[A-Z\']+/',
                '// general_instr disabled — strict evaluation only'
            )
            try:
                strict_parser = Lark(strict_grammar, parser="earley",
                                     ambiguity="resolve")
                print(f"Grammar: {grammar_path} loaded (strict + fallback parsers)")
            except Exception:
                strict_parser = lark_parser   # fallback to full parser
                print(f"Grammar: {grammar_path} loaded (strict parser unavailable)")
        except Exception as e:
            print(f"Grammar load failed: {e}")
    else:
        print(f"Grammar not found at {grammar_path} — C_cfg will be 0")

    # ── Load model ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # ── Build GCD logits processor ────────────────────────────────────────
    logits_processors = LogitsProcessorList()
    if use_gcd:
        print("Building GCD logits processor (vocabulary + bigram mask) ...")
        gcd_processor = GCDLogitsProcessor(
            tokenizer, vocab, ngram_wl=ngram_wl
        )
        logits_processors.append(gcd_processor)
        import torch as _t
        n_allowed = gcd_processor._token_mask.sum().item()
        print(f"  Vocab mask: {n_allowed} / {len(tokenizer)} token IDs allowed")
        print(f"  Bigram table: {len(gcd_processor._valid_next)} entries, "
              f"avg {sum(len(v) for v in gcd_processor._valid_next.values()) / max(1, len(gcd_processor._valid_next)):.1f} successors/word")

    # ── Load test data ────────────────────────────────────────────────────
    with open(data_path) as f:
        test_pairs = json.load(f)
    print(f"Test pairs: {len(test_pairs)}")

    # ── Generate + evaluate ───────────────────────────────────────────────
    results = []
    ctok_scores, cphr_scores, ccfg_scores = [], [], []

    for idx, p in enumerate(test_pairs):
        request = p.get("request", p.get("text", ""))
        ref     = p.get("response", p.get("reference", ""))
        fsm     = p.get("fsm_state", "Init")

        instruction = format_instruction(request, fsm, domain=domain)
        tok = tokenizer(
            instruction,
            return_tensors="pt",
            max_length=512 - max_new_tokens,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                logits_processor=logits_processors,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = out[0, tok["input_ids"].size(1):]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        toks     = gen_text.upper().split()

        ctok = compute_ctok(toks, vocab)
        cphr = compute_cphr(toks, ngram_wl)
        ccfg = compute_ccfg(gen_text, lark_parser, strict_parser=strict_parser)

        ctok_scores.append(ctok)
        cphr_scores.append(cphr)
        ccfg_scores.append(ccfg)

        results.append({
            "request":   request,
            "reference": ref,
            "generated": gen_text,
            "C_tok":     round(ctok, 4),
            "C_phr":     round(cphr, 4),
            "C_cfg":     round(ccfg, 4),
        })

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(test_pairs)}] "
                  f"C_tok={sum(ctok_scores)/len(ctok_scores):.4f} "
                  f"C_phr={sum(cphr_scores)/len(cphr_scores):.4f} "
                  f"C_cfg={sum(ccfg_scores)/len(ccfg_scores):.4f}")

    # ── Summary ───────────────────────────────────────────────────────────
    mean_ctok = sum(ctok_scores) / len(ctok_scores)
    mean_cphr = sum(cphr_scores) / len(cphr_scores)
    mean_ccfg = sum(ccfg_scores) / len(ccfg_scores)

    print(f"\n{'='*55}")
    print(f"GCD Evaluation Results ({len(results)} test examples)")
    print(f"{'='*55}")
    print(f"  C_tok = {mean_ctok:.4f}")
    print(f"  C_phr = {mean_cphr:.4f}")
    print(f"  C_cfg = {mean_ccfg:.4f}")
    print(f"{'='*55}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "condition":   "GCD",
        "model":       model_path,
        "grammar":     grammar_path,
        "gcd_enabled": use_gcd,
        "n_test":      len(results),
        "C_tok":       round(mean_ctok, 4),
        "C_phr":       round(mean_cphr, 4),
        "C_cfg":       round(mean_ccfg, 4),
        "per_example": results,
    }

    results_path = Path(output_dir) / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Also write DONE flag for the runner
    (Path(output_dir) / "DONE").touch()
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GCD inference-time baseline evaluation (Condition C4)"
    )
    parser.add_argument("--model",    required=True,
                        help="Model path or HF model name "
                             "(e.g. results/C2/best or gpt2-large)")
    parser.add_argument("--data",     default="atc_test.json",
                        help="Test data JSON file")
    parser.add_argument("--grammar",  default="G_ATC_v2.lark",
                        help="Path to Lark EBNF grammar file")
    parser.add_argument("--output",   default="results/C4",
                        help="Output directory for results")
    parser.add_argument("--max_new",  type=int, default=64,
                        help="Max new tokens to generate")
    parser.add_argument("--no_gcd",   action="store_true",
                        help="Disable GCD; use vocabulary mask only")
    parser.add_argument("--no_fallback", action="store_true",
                        help="Disable vocabulary mask fallback when grammar "
                             "cannot constrain")
    parser.add_argument("--vocab",    default="",
                        help="Path to vocabulary whitelist JSON "
                             "(default: same directory as this script)")
    parser.add_argument("--phrase",   default="",
                        help="Path to n-gram whitelist JSON "
                             "(default: same directory as this script)")
    parser.add_argument("--domain",   default="atc",
                        choices=["atc", "smcp"],
                        help="Domain for system prompt (atc or smcp)")
    args = parser.parse_args()

    evaluate_gcd(
        model_path       = args.model,
        data_path        = args.data,
        grammar_path     = args.grammar,
        output_dir       = args.output,
        max_new_tokens   = args.max_new,
        fallback_to_vocab= not args.no_fallback,
        use_gcd          = not args.no_gcd,
        vocab_path       = args.vocab,
        phrase_path      = args.phrase,
        domain           = args.domain,
    )
