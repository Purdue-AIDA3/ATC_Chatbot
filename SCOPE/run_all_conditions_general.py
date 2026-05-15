#!/usr/bin/env python3
"""
SCOPE Experimental Runner
Runs all 13 experimental conditions sequentially, evaluates on test set,
saves results, and produces comparison table + plots.

Usage:
    python run_all_conditions_general.py \
        --data smcp_pairs.json --test_data smcp_test.json \
        --model gpt2-large --domain smcp \
        --grammar G_SMCP.lark \
        --vocab_path vocab_SMCP.json \
        --phrase_path ngram_whitelist_SMCP.json \
        --script scope_train_general.py \
        --gcd_script evaluate_gcd_general.py \
        --output_root results_maritime/gpt \
        --conditions C2 C3 C11 C4

    python run_all_conditions_general.py \
        --data atc_pairs.json \
        --model gpt2-large \
        --epochs 5 \
        --batch_size 16 \
        --M_samples 4 \
        --output_root results/

    # To resume from a partial run (skip completed conditions):
    python run_all_conditions_general.py --resume
"""

import os, sys, json, time, argparse, subprocess
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Condition definitions ─────────────────────────────────────────────────────

def make_conditions(args):
    """
    Returns ordered list of (condition_id, label, description, extra_args).
    extra_args is a list of CLI flags to append to the scope_train_general.py call.
    """
    BASE = [
        "--data",       args.data,
        "--test_data",  args.test_data,
        "--epochs",     str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr",         str(args.lr),
        "--max_new_tok",str(args.max_new_tok),
        "--seed",       str(args.seed),
        "--vocab_path", args.vocab_path,
        "--phrase_path",args.phrase_path,
        "--grammar",    args.grammar,
        "--domain",     args.domain,
    ]
    GRPO = ["--M_samples", str(args.M_samples)]

    # DPO reference model: use base model by default (SFT ckpt set later at runtime)
    dpo_ref = getattr(args, "dpo_ref", "")

    conditions = [
        # ── Baselines ─────────────────────────────────────────────────────
        ("C1",  "Vanilla",
         "GPT-2 Large, no fine-tuning",
         "train",
         ["--model", args.model, "--epochs", "0",
          "--no_ltok", "--no_lphr", "--no_lcfg"]),

        ("C2",  "SFT",
         "Standard instruction-tuned SFT (CE only)",
         "train",
         ["--model", args.model, "--lambda_ce", "1.0",
          "--no_ltok", "--no_lphr", "--no_lcfg"]),

        ("C3",  "DPO",
         "Direct Preference Optimisation (synthetic C_tok preference pairs)",
         "train",
         ["--model", args.model, "--lambda_ce", "1.0",
          "--no_ltok", "--no_lphr", "--no_lcfg",
          "--dpo", "--dpo_beta", str(getattr(args, "dpo_beta", 0.1))]
         + (["--dpo_ref", dpo_ref] if dpo_ref else [])),

        # ── Ablation: single-level ─────────────────────────────────────────
        ("C5",  "SCOPE-tok",
         "L_tok only — lexical constraint loss",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_tok", "1.0",
          "--no_lphr", "--no_lcfg"]),

        ("C6",  "SCOPE-phr-REINFORCE",
         "L_phr REINFORCE (GRPO with M=1, no group normalisation)",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_phr", "0.5",
          "--no_ltok", "--no_lcfg",
          "--M_samples", "1"]),   # M=1 → REINFORCE (no GRPO appended)

        ("C7",  "SCOPE-phr-GRPO",
         "L_phr with GRPO (group-normalised advantage)",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_phr", "0.5",
          "--no_ltok", "--no_lcfg"] + GRPO),

        ("C8",  "SCOPE-cfg",
         "L_cfg only — syntactic CFG structural loss",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_cfg", "0.3",
          "--no_ltok", "--no_lphr"] + GRPO),

        # ── Ablation: two-level ────────────────────────────────────────────
        ("C9",  "SCOPE-2L",
         "L_tok + L_phr (no L_cfg)",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_tok", "1.0", "--lambda_phr", "0.5",
          "--no_lcfg"] + GRPO),

        ("C10", "SCOPE-REINFORCE",
         "Full SCOPE with REINFORCE (M=1) instead of GRPO",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_tok", "1.0", "--lambda_phr", "0.5", "--lambda_cfg", "0.3",
          "--M_samples", "1"]),   # M=1 → REINFORCE (no GRPO appended)

        # ── Proposed method ────────────────────────────────────────────────
        ("C11", "SCOPE-full",
         "Full three-level SCOPE (proposed method)",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_tok", "1.0", "--lambda_phr", "0.5", "--lambda_cfg", "0.3",
          ] + GRPO),

        # ── GCD inference-time baselines ──────────────────────────────────
        ("C4",  "GCD",
         "Grammar-Constrained Decoding on SFT checkpoint",
         "gcd",
         {"ckpt_source": "C2"}),   # apply GCD to C2/best

        ("C4a",  "GCD_Vanilla",
         "Grammar-Constrained Decoding on Vanilla checkpoint",
         "gcd",
         {"ckpt_source": "C1"}),   # apply GCD to C1/best

        ("C4b", "SCOPE+GCD",
         "Grammar-Constrained Decoding on SCOPE-full checkpoint",
         "gcd",
         {"ckpt_source": "C11"}),  # apply GCD to C11/best

        ("C4c", "SCOPE+GCD_Vanilla",
         "Grammar-Constrained Decoding+Vanilla on SCOPE-full checkpoint",
         "gcd",
         {"ckpt_source": "C11"}),  # apply GCD_vanilla to C11/best

        # ── GAD inference-time baselines (Park et al. NeurIPS 2024) ─────────
        ("C4_GAD",  "GAD (SFT)",
         "Grammar-Aligned Decoding (ASAp) on SFT checkpoint",
         "gad",
         {"ckpt_source": "C2"}),

        ("C4a_GAD", "GAD (Vanilla)",
         "Grammar-Aligned Decoding (ASAp) on Vanilla checkpoint",
         "gad",
         {"ckpt_source": "gpt2-large"}),

        ("C4b_GAD", "GAD (SCOPE)",
         "Grammar-Aligned Decoding (ASAp) on SCOPE-full checkpoint",
         "gad",
         {"ckpt_source": "C11"}),

        # ── Frontier closed-weight baselines ──────────────────────────────
        ("C12", "GPT-5.4-0shot",
         "GPT-5.4 zero-shot (no fine-tuning)",
         "manual", None),

        ("C13", "GPT-5.4-5shot",
         "GPT-5.4 five-shot (no fine-tuning)",
         "manual", None),
    ]

    return BASE, conditions  # each entry: (cond_id, label, desc, kind, extra_args)


# ── Runner helpers ────────────────────────────────────────────────────────────

def _print_header(cond_id, label, description):
    print(f"\n{'='*70}")
    print(f"  {cond_id}: {label}")
    print(f"  {description}")
    print(f"{'='*70}")


def _run_subprocess(cmd, log_path, dry_run=False):
    """Run a command, tee output to log file, return success bool."""
    print(f"  CMD: {' '.join(cmd)}\n")
    if dry_run:
        print("  [DRY RUN — not executing]")
        return True
    t0 = time.time()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd, stderr=subprocess.STDOUT, text=True,
            stdout=log_f,
        )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  ✗ FAILED (exit {proc.returncode}) — see {log_path}")
        return False
    print(f"  ✓ Complete in {elapsed/60:.1f} min")
    return True


def run_condition(cond_id, label, description, kind, extra_args, base_args,
                  output_root, script_path, dry_run=False):
    """Dispatch one training condition to the appropriate runner."""
    out_dir   = Path(output_root) / cond_id
    done_flag = out_dir / "DONE"
    _print_header(cond_id, label, description)

    if done_flag.exists():
        print(f"  ✓ Already complete — skipping")
        return out_dir

    if kind == "manual":
        print(f"  ⚠  Manual condition — run separately and save results to")
        print(f"     {out_dir}/test_results.json")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # Training conditions (kind == "train")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ([sys.executable, str(script_path)]
           + base_args
           + ["--output", str(out_dir)]
           + (extra_args or []))
    ok = _run_subprocess(cmd, out_dir / "training.log", dry_run)
    if ok and not dry_run:
        done_flag.touch()
    return out_dir


def run_gcd_condition(cond_id, label, description, output_root,
                      gcd_script_path, ckpt_path, test_data_path,
                      grammar_path, max_new_tok=64, dry_run=False,
                      vocab_path=None, phrase_path=None, domain="atc"):
    """
    Run GCD evaluation on any trained checkpoint.
    C4:  GCD on SFT checkpoint  (ckpt_path = results/C2/best)
    C4a:  GCD on Vanilla checkpoint  (ckpt_path = results/C1/best)
    C4b: GCD on SCOPE-full ckpt (ckpt_path = results/C11/best)
    C4c: GCD_vanilla on SCOPE-full ckpt (ckpt_path = results/C11/best)
    """
    out_dir   = Path(output_root) / cond_id
    done_flag = out_dir / "DONE"
    _print_header(cond_id, label, description)

    if done_flag.exists():
        print(f"  ✓ Already complete — skipping")
        return out_dir

    if not Path(ckpt_path).exists():
        print(f"  ⚠  Checkpoint not found at {ckpt_path}")
        print(f"     Ensure the source condition has completed first.")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    if not Path(gcd_script_path).exists():
        print(f"  ✗ GCD script not found at {gcd_script_path}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve vocab and phrase paths — use passed args or fall back to GCD script dir
    gcd_dir     = Path(gcd_script_path).parent
    vocab_path  = Path(vocab_path)  if vocab_path  else gcd_dir / "vocab_ATC.json"
    phrase_path = Path(phrase_path) if phrase_path else gcd_dir / "ngram_whitelist_ATC.json"

    cmd = [
        sys.executable, str(gcd_script_path),
        "--model",   str(ckpt_path),
        "--data",    str(test_data_path),
        "--grammar", str(grammar_path),
        "--output",  str(out_dir),
        "--max_new", str(max_new_tok),
        "--vocab",   str(vocab_path),
        "--phrase",  str(phrase_path),
        "--domain",  domain,
    ]
    _run_subprocess(cmd, out_dir / "gcd_eval.log", dry_run)
    return out_dir


def run_gad_condition(cond_id, label, description, output_root,
                      gad_script_path, ckpt_path, test_data_path,
                      gad_grammar_path, max_new_tok=64, dry_run=False):
    """
    Run GAD (ASAp) evaluation on any trained checkpoint.
    Uses evaluate_gad.py and the GBNF grammar G_ATC_v2.ebnf.
    C4_GAD:  GAD on SFT checkpoint
    C4a_GAD: GAD on Vanilla (gpt2-large directly)
    C4b_GAD: GAD on SCOPE-full checkpoint
    """
    out_dir   = Path(output_root) / cond_id
    done_flag = out_dir / "DONE"
    _print_header(cond_id, label, description)

    if done_flag.exists():
        print(f"  ✓ Already complete — skipping")
        return out_dir

    if not Path(gad_script_path).exists():
        print(f"  ✗ GAD script not found at {gad_script_path}")
        print(f"     Clone https://github.com/ebmoon/transformers-GAD.git "
              f"alongside {gad_script_path}")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    if not Path(gad_grammar_path).exists():
        print(f"  ✗ GBNF grammar not found at {gad_grammar_path}")
        print(f"     This is the GBNF format grammar (not G_ATC_v2.lark).")
        print(f"     Download from outputs or generate with:")
        print(f"       cp G_ATC_v2.ebnf {gad_grammar_path}")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # Resolve checkpoint: ckpt_path may be a model ID (gpt2-large) or a path
    ckpt = str(ckpt_path)
    if Path(ckpt).exists() or not ckpt.startswith("results"):
        pass  # HuggingFace model ID or absolute path — use directly
    else:
        if not Path(ckpt).exists():
            print(f"  ⚠  Checkpoint not found at {ckpt}")
            print(f"     Ensure the source condition has completed first.")
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir

    # Resolve vocab and phrase paths relative to the GAD script location
    gad_dir    = Path(gad_script_path).parent
    vocab_path  = gad_dir / "vocab_ATC.json"
    phrase_path = gad_dir / "ngram_whitelist_ATC_v2.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(gad_script_path),
        "--model",   ckpt,
        "--data",    str(test_data_path),
        "--grammar", str(gad_grammar_path),
        "--output",  str(out_dir),
        "--max_new", str(max_new_tok),
        "--mode",    "gad",
        "--vocab",   str(vocab_path),
        "--phrase",  str(phrase_path),
    ]
    _run_subprocess(cmd, out_dir / "gad_eval.log", dry_run)
    return out_dir


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_condition(cond_id, label, out_dir, data_path,
                       script_path, test_split_path,
                       vocab_path="vocab_ATC.json",
                       phrase_path="ngram_whitelist_ATC.json",
                       domain="atc"):
    """
    Evaluate one condition on the test set and return metrics dict.

    - If test_results.json already exists (e.g. written by evaluate_gcd.py
      or a previous run), load and return it directly.
    - C4 (GCD) results are written by evaluate_gcd.py — just load them.
    - Manual conditions (C12, C13) return None if not present.
    - All others: evaluate the best/ checkpoint inline.
    """
    out_dir      = Path(out_dir)
    results_path = out_dir / "test_results.json"
    best_ckpt    = out_dir / "best"

    # Load pre-existing results (GCD, manual, or previous eval)
    if results_path.exists():
        with open(results_path) as f:
            r = json.load(f)
        print(f"  {cond_id}: loaded existing results "
              f"C_tok={r.get('C_tok',0):.4f} "
              f"C_phr={r.get('C_phr',0):.4f} "
              f"C_cfg={r.get('C_cfg',0):.4f}")
        return r

    # GCD/GAD conditions: results written by their evaluators
    if cond_id in ("C4", "C4a", "C4b", "C4c"):
        print(f"  ⚠  {cond_id} (GCD): no test_results.json found. "
              f"Did run_gcd_condition complete?")
        return None
    if cond_id in ("C4_GAD", "C4a_GAD", "C4b_GAD"):
        print(f"  ⚠  {cond_id} (GAD): no test_results.json found. "
              f"Did run_gad_condition complete?")
        return None

    # Manual conditions
    if cond_id in ("C12", "C13"):
        print(f"  ⚠  {cond_id} requires manual evaluation — skipping")
        return None

    # Determine model path
    if not best_ckpt.exists():
        if cond_id == "C1":
            model_path = "gpt2-large"   # vanilla: evaluate base model
        else:
            print(f"  ⚠  No checkpoint for {cond_id} — skipping")
            return None
    else:
        model_path = str(best_ckpt)

    print(f"  Evaluating {cond_id} ({label}) ← {model_path}")
    return _eval_inline(model_path, test_split_path, results_path, label,
                        vocab_path=vocab_path, phrase_path=phrase_path, domain=domain)

def _eval_inline(model_path, test_split_path, results_path, label,
                 vocab_path="vocab_ATC.json",
                 phrase_path="ngram_whitelist_ATC.json",
                 domain="atc"):
    """Inline evaluation — import training module functions directly."""
    import importlib.util, torch, json

    # Use scope_train_general (domain-agnostic)
    script_file = Path(__file__).resolve().parent / "scope_train_general.py"
    spec = importlib.util.spec_from_file_location("scope_train", script_file)
    scope = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scope)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab     = scope.load_whitelist(Path(vocab_path))
    ngram_wl  = scope.load_ngram_whitelist(Path(phrase_path))
    cfg_parser = scope.load_grammar(scope.GRAMMAR_PATH)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    with open(test_split_path) as f:
        test_pairs = json.load(f)

    results = []
    for p in test_pairs:
        item = scope.format_atc(p["request"], p["response"], domain=domain)
        tok  = tokenizer(item["instruction"], return_tensors="pt",
                         max_length=512, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **tok, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        gen  = out[0, tok["input_ids"].size(1):]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        toks = text.upper().split()
        results.append({
            "request":   p["request"],
            "reference": p["response"],
            "generated": text,
            "C_tok":     scope.compute_ctok(toks, vocab),
            "C_phr":     scope.compute_cphr(toks, ngram_wl),
            "C_cfg":     scope.compute_ccfg_partial(text, cfg_parser),
        })

    summary = {
        "condition": label,
        "n_test":    len(results),
        "C_tok":     sum(r["C_tok"] for r in results) / len(results),
        "C_phr":     sum(r["C_phr"] for r in results) / len(results),
        "C_cfg":     sum(r["C_cfg"] for r in results) / len(results),
        "per_example": results,
    }

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"    C_tok={summary['C_tok']:.4f}  "
          f"C_phr={summary['C_phr']:.4f}  "
          f"C_cfg={summary['C_cfg']:.4f}")

    # Free GPU memory so subsequent conditions don't OOM
    del model
    import torch as _torch
    _torch.cuda.empty_cache()

    return summary


# ── Plotting ──────────────────────────────────────────────────────────────────

PURDUE_GOLD  = "#CFB991"
BLACK        = "#000000"
WHITE        = "#FFFFFF"
GREY_LIGHT   = "#E8E8E8"
GREY_MED     = "#AAAAAA"
GREY_DARK    = "#444444"

# Condition display order and colour coding
COND_ORDER = ["C1","C2","C3","C4","C4a","C5","C7","C8","C9","C11","C4b","C4c"]
COND_LABELS = {
    "C1":  "Vanilla",
    "C2":  "SFT",
    "C3":  "DPO",
    "C4":  "GCD",
    "C4a":  "GCD_vanilla",
    "C5":  "SCOPE-tok",
    "C7":  "SCOPE-phr",
    "C8":  "SCOPE-cfg",
    "C9":  "SCOPE-2L",
    "C11": "SCOPE-full",
    "C4b":  "SCOPE+GCD",
    "C4c":  "SCOPE+GCD_vanilla",
}
COND_COLORS = {
    "C1":  GREY_MED,
    "C2":  GREY_DARK,
    "C3":  "#A8860B",
    "C4":  "#C8860B",
    "C4a":  "#B7760B",
    "C5":  "#B8860B",
    "C7":  "#8B6914",
    "C8":  "#A0522D",
    "C9":  "#CD853F",
    "C11": PURDUE_GOLD,
    "C4b":  "#CB860B",
    "C4c":  "#A2560B",
}


def plot_training_curves(output_root, plot_dir):
    """
    Plot training curves (C_tok, C_phr, C_cfg vs epoch) for all conditions
    that have training_history.json.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load histories
    histories = {}
    for cond_id in COND_ORDER:
        hist_path = Path(output_root) / cond_id / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                histories[cond_id] = json.load(f)

    if not histories:
        print("No training histories found — skipping training curves")
        return

    metrics = [("C_tok", "$C_{\\mathrm{tok}}$ (Lexical)"),
               ("C_phr", "$C_{\\mathrm{phr}}$ (Phraseological)"),
               ("C_cfg", "$C_{\\mathrm{cfg}}$ (Syntactic)")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=WHITE)
    fig.suptitle("SCOPE Training Curves — Validation Compliance by Epoch",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, (metric, ylabel) in zip(axes, metrics):
        for cond_id, hist in histories.items():
            epochs = [h["epoch"] for h in hist]
            values = [h.get(metric, 0) for h in hist]
            ax.plot(epochs, values,
                    color=COND_COLORS.get(cond_id, GREY_MED),
                    linewidth=2.2,
                    marker="o", markersize=4,
                    label=COND_LABELS.get(cond_id, cond_id),
                    zorder=3 if cond_id == "C11" else 2)

        # Oracle ceiling
        oracle = {"C_tok": 0.879, "C_phr": 0.771, "C_cfg": 0.290}
        ax.axhline(oracle[metric], color=GREY_MED, linewidth=1,
                   linestyle="--", alpha=0.7, label="Oracle ceiling")

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out_path = plot_dir / "training_curves.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.savefig(str(out_path).replace(".pdf", ".png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_comparison_bar(all_results, plot_dir):
    """
    Grouped bar chart: C_tok, C_phr, C_cfg side by side for each condition.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Filter to conditions we have results for, in display order
    conds = [(cid, COND_LABELS.get(cid, cid), all_results[cid])
             for cid in COND_ORDER if cid in all_results and all_results[cid]]

    if not conds:
        print("No test results to plot")
        return

    labels  = [c[1] for c in conds]
    ctok    = [c[2]["C_tok"] for c in conds]
    cphr    = [c[2]["C_phr"] for c in conds]
    ccfg    = [c[2]["C_cfg"] for c in conds]
    colors  = [COND_COLORS.get(c[0], GREY_MED) for c in conds]

    x      = np.arange(len(labels))
    width  = 0.26
    offsets = [-width, 0, width]
    metric_data  = [ctok, cphr, ccfg]
    metric_names = ["$C_{\\mathrm{tok}}$", "$C_{\\mathrm{phr}}$",
                    "$C_{\\mathrm{cfg}}$"]
    hatch_list   = ["", "//", ".."]

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=WHITE)

    for off, data, mname, hatch in zip(
            offsets, metric_data, metric_names, hatch_list):
        bars = ax.bar(x + off, data, width,
                      label=mname,
                      color=[c for c in colors],
                      hatch=hatch,
                      edgecolor="white", linewidth=0.5,
                      alpha=0.92)
        # Value labels on top of each bar
        for bar, val in zip(bars, data):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=6.5, fontweight="bold", color=GREY_DARK)

    # Oracle lines
    for val, color, ls in [(0.879, "#555555", "--"),
                            (0.771, "#555555", ":"),
                            (0.290, "#555555", "-.")]:
        ax.axhline(val, color=color, linewidth=0.8, linestyle=ls, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Compliance Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("SCOPE Ablation — Test Set Compliance by Condition",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(title="Metric", fontsize=9, title_fontsize=9,
              loc="upper left", framealpha=0.9)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    plt.tight_layout()
    out_path = plot_dir / "comparison_bars.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.savefig(str(out_path).replace(".pdf", ".png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_improvement_over_sft(all_results, plot_dir):
    """
    Horizontal bar chart showing improvement of each SCOPE variant over SFT.
    """
    plot_dir = Path(plot_dir)
    if "C2" not in all_results or not all_results["C2"]:
        return

    sft = all_results["C2"]
    metrics = ["C_tok", "C_phr", "C_cfg"]
    metric_labels = ["$C_{\\mathrm{tok}}$",
                     "$C_{\\mathrm{phr}}$",
                     "$C_{\\mathrm{cfg}}$"]

    scope_conds = [c for c in COND_ORDER
                   if c not in ("C1","C2") and c in all_results
                   and all_results[c]]

    if not scope_conds:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=WHITE,
                             sharey=True)
    fig.suptitle("Improvement over SFT Baseline (Δ score)",
                 fontsize=12, fontweight="bold", y=1.02)

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        deltas = [all_results[c][metric] - sft[metric]
                  for c in scope_conds]
        labels = [COND_LABELS.get(c, c) for c in scope_conds]
        bar_colors = [COND_COLORS.get(c, GREY_MED) for c in scope_conds]

        y = np.arange(len(labels))
        bars = ax.barh(y, deltas, color=bar_colors,
                       edgecolor="white", linewidth=0.5, alpha=0.9)

        ax.axvline(0, color=GREY_DARK, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(f"Δ {mlabel}", fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linestyle=":")

        # Value labels
        for bar, val in zip(bars, deltas):
            ax.text(val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height()/2,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    out_path = plot_dir / "improvement_over_sft.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.savefig(str(out_path).replace(".pdf", ".png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_reward_curves(output_root, plot_dir):
    """
    Plot Rphr and Rcfg training rewards over epochs for SCOPE conditions.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    for cond_id in ["C7","C8","C9","C11"]:
        hist_path = Path(output_root) / cond_id / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                h = json.load(f)
                if any("Rphr" in e or "R_phr" in e for e in h):
                    histories[cond_id] = h

    if not histories:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=WHITE)

    for cond_id, hist in histories.items():
        epochs = [e["epoch"] for e in hist]
        rphr = [e.get("Rphr", e.get("R_phr", 0)) for e in hist]
        rcfg = [e.get("Rcfg", e.get("R_cfg", 0)) for e in hist]
        color = COND_COLORS.get(cond_id, GREY_MED)
        label = COND_LABELS.get(cond_id, cond_id)
        ax1.plot(epochs, rphr, color=color, marker="o",
                 markersize=4, linewidth=2, label=label)
        ax2.plot(epochs, rcfg, color=color, marker="s",
                 markersize=4, linewidth=2, label=label)

    for ax, title in [(ax1, "Mean $C_{\\mathrm{phr}}$ Reward During Training"),
                      (ax2, "Mean $C_{\\mathrm{cfg}}$ Reward During Training")]:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Reward")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, linestyle=":")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out_path = plot_dir / "reward_curves.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.savefig(str(out_path).replace(".pdf", ".png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ── LaTeX + CSV table ─────────────────────────────────────────────────────────

FULL_COND_LABELS = {
    "C1":  "Vanilla (no fine-tuning)",
    "C2":  "Standard SFT",
    "C3":  "DPO",
    "C4":  "GCD (on SFT ckpt.)",
    "C4a":  "GCD (on vanilla ckpt.)",
    "C5":  "SCOPE-tok only",
    "C6":  "SCOPE-phr (REINFORCE)",
    "C7":  "SCOPE-phr (GRPO)",
    "C8":  "SCOPE-cfg only",
    "C9":  "SCOPE-2L (tok+phr)",
    "C10": "SCOPE (REINFORCE)",
    "C11": "SCOPE-full (proposed)",
    "C4b":  "SCOPE+GCD (on SFT ckpt.)",
    "C4c":  "SCOPE+GCD_vanilla (on vanilla ckpt.)",
    "C12": "GPT-5.4 zero-shot",
    "C13": "GPT-5.4 five-shot",
}

def write_results_table(all_results, output_root):
    """Write CSV and LaTeX results table."""
    rows = []
    cond_order = ["C1","C2","C3","C4","C4a","C5","C6","C7",
                  "C8","C9","C10","C11","C4b","C4c","C12","C13"]
    for cid in cond_order:
        r = all_results.get(cid)
        if r is None:
            rows.append((cid, FULL_COND_LABELS.get(cid,""), "—","—","—"))
        else:
            rows.append((
                cid,
                FULL_COND_LABELS.get(cid, r.get("condition","")),
                f"{r['C_tok']:.4f}",
                f"{r['C_phr']:.4f}",
                f"{r['C_cfg']:.4f}",
            ))

    output_root = Path(output_root)

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = output_root / "results_table.csv"
    with open(csv_path, "w") as f:
        f.write("Condition,Method,C_tok,C_phr,C_cfg\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"  Saved: {csv_path}")

    # ── LaTeX ─────────────────────────────────────────────────────────────
    # Find best value per metric for bold formatting
    numeric_rows = [(r,c,p,g) for (cid,label,c,p,g) in rows
                    if c != "—"
                    for r in [rows.index((cid,label,c,p,g))]]
    def best_val(col_idx):
        vals = [(row[col_idx], i) for i,row in enumerate(rows) if row[col_idx] != "—"]
        if not vals: return -1, -1
        best = max(float(v) for v,_ in vals)
        return best, [i for v,i in vals if float(v)==best]

    best_ctok, best_ctok_idx = best_val(2)
    best_cphr, best_cphr_idx = best_val(3)
    best_ccfg, best_ccfg_idx = best_val(4)

    def fmt_cell(val, row_idx, best_idx):
        if val == "—": return "---"
        if isinstance(best_idx, list) and row_idx in best_idx:
            return f"\\textbf{{{val}}}"
        return val

    latex_path = output_root / "results_table.tex"
    with open(latex_path, "w") as f:
        f.write("% SCOPE Results Table — auto-generated\n")
        f.write("% Bold = best result per metric\n\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Test-set compliance results across all experimental "
                "conditions. Oracle ceilings: $\\Ctok=0.879$, $\\Cphr=0.771$, "
                "$\\Ccfg=0.290$. Best per metric in \\textbf{bold}.}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Cond.} & \\textbf{Method} & "
                "$\\Ctok$ & $\\Cphr$ & $\\Ccfg$ \\\\\n")
        f.write("\\midrule\n")

        sections = [
            ("Baselines", ["C1","C2","C3","C4","C4a"]),
            ("Single-level ablation", ["C5","C6","C7","C8"]),
            ("Two-level ablation", ["C9","C10"]),
            ("Proposed method", ["C11","C4b","C4c"]),
            ("Closed-weight inference", ["C12","C13"]),
        ]
        for sec_label, cids in sections:
            f.write(f"\\multicolumn{{5}}{{l}}"
                    f"{{\\textit{{{sec_label}}}}} \\\\\n")
            for cid in cids:
                row = next((r for r in rows if r[0]==cid), None)
                if row is None: continue
                row_idx = rows.index(row)
                c = fmt_cell(row[2], row_idx, best_ctok_idx)
                p = fmt_cell(row[3], row_idx, best_cphr_idx)
                g = fmt_cell(row[4], row_idx, best_ccfg_idx)
                star = "\\dag" if cid == "C11" else ""
                f.write(f"  {cid} & {row[1]}{star} & {c} & {p} & {g} \\\\\n")
            f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\vspace{2pt}\n")
        f.write("\\begin{flushleft}\\scriptsize\n")
        f.write("\\dag~Proposed method. "
                "C3/C4/C12/C13 require manual evaluation (see text).\n")
        f.write("\\end{flushleft}\n")
        f.write("\\end{table*}\n")

    print(f"  Saved: {latex_path}")
    return csv_path, latex_path


def print_summary_table(all_results):
    """Print ASCII summary table to stdout."""
    print("\n" + "="*65)
    print(f"{'SCOPE RESULTS SUMMARY':^65}")
    print("="*65)
    print(f"{'Cond':<6} {'Method':<28} {'C_tok':>7} {'C_phr':>7} {'C_cfg':>7}")
    print("-"*65)
    for cid in ["C1","C2","C3","C4","C4a","C5","C7","C8","C9","C11","C4b","C4c"]:
        r = all_results.get(cid)
        label = FULL_COND_LABELS.get(cid, cid)[:28]
        if r:
            print(f"{cid:<6} {label:<28} "
                  f"{r['C_tok']:>7.4f} {r['C_phr']:>7.4f} {r['C_cfg']:>7.4f}")
        else:
            print(f"{cid:<6} {label:<28} {'—':>7} {'—':>7} {'—':>7}")
    print("="*65)
    # Oracle
    print(f"{'':6} {'Oracle ceiling':<28} "
          f"{'0.8790':>7} {'0.7710':>7} {'0.2900':>7}")
    print("="*65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run all SCOPE conditions")
    parser.add_argument("--data",        default="atc_pairs.json")
    parser.add_argument("--test_data",   default="atc_test.json",
                        help="Test split for final evaluation")
    parser.add_argument("--model",       default="gpt2-large")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--M_samples",   type=int,   default=4)
    parser.add_argument("--max_new_tok", type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output_root", default="results")
    parser.add_argument("--plots_dir",   default="plots")
    parser.add_argument("--script",      default="scope_train_general.py")
    parser.add_argument("--gcd_script",  default="evaluate_gcd_general.py",
                        help="Path to evaluate_gcd_general.py")
    parser.add_argument("--grammar",     default="G_ATC.lark",
                        help="Lark grammar file for GCD (C4)")
    parser.add_argument("--vocab_path",  default="vocab_ATC.json",
                        help="Vocabulary whitelist JSON (V_domain)")
    parser.add_argument("--phrase_path", default="ngram_whitelist_ATC.json",
                        help="N-gram whitelist JSON (P_domain)")
    parser.add_argument("--domain",      default="atc",
                        choices=["atc", "smcp"],
                        help="Domain for system prompt (atc or smcp)")
    parser.add_argument("--gad_script",  default="evaluate_gad.py",
                        help="Path to evaluate_gad.py (Park et al. NeurIPS 2024)")
    parser.add_argument("--gad_grammar", default="G_ATC.ebnf",
                        help="GBNF grammar file for GAD (not Lark format)")
    parser.add_argument("--dpo_beta",    type=float, default=0.1,
                        help="DPO beta hyperparameter (C3)")
    parser.add_argument("--dpo_ref",     default="",
                        help="Path to DPO reference model (default: base model)")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip conditions that already have DONE flag")
    parser.add_argument("--eval_only",   action="store_true",
                        help="Skip training, only evaluate and plot")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Print commands without running")
    parser.add_argument("--conditions",  nargs="+",
                        help="Only run specific conditions e.g. --conditions C1 C2 C11")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plots_dir)
    script_path = Path(args.script).resolve()

    if not script_path.exists():
        print(f"ERROR: training script not found at {script_path}")
        sys.exit(1)

    # Use full dataset for training, test split for evaluation
    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"WARNING: test split {test_path} not found — using full dataset")
        test_path = Path(args.data)

    BASE, conditions = make_conditions(args)
    start_time = time.time()

    print(f"\nSCOPE Experimental Runner")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output:  {output_root.resolve()}")
    print(f"Conditions to run: {len(conditions)}")

    gcd_script_path = Path(args.gcd_script).resolve()
    grammar_path    = Path(args.grammar).resolve()

    # ── Training / Evaluation ─────────────────────────────────────────────
    if not args.eval_only:
        for cond_id, label, description, kind, extra_args in conditions:
            if args.conditions and cond_id not in args.conditions:
                continue

            if kind == "gcd":
                # extra_args is a dict: {"ckpt_source": "C2"}, {"ckpt_source": "C1"} or {"ckpt_source": "C11"}
                src_cond = (extra_args or {}).get("ckpt_source", "C2")
                ckpt_path = output_root / src_cond / "best"
                run_gcd_condition(
                    cond_id, label, description,
                    output_root, gcd_script_path,
                    ckpt_path, test_path, grammar_path,
                    args.max_new_tok, dry_run=args.dry_run,
                    vocab_path=args.vocab_path,
                    phrase_path=args.phrase_path,
                    domain=args.domain,
                )
            elif kind == "gad":
                # GAD (ASAp) — Park et al. NeurIPS 2024
                src = (extra_args or {}).get("ckpt_source", "C2")
                # If src is a condition ID, resolve to checkpoint path
                if src.startswith("C"):
                    ckpt_path = output_root / src / "best"
                else:
                    ckpt_path = Path(src)  # e.g. "gpt2-large"
                gad_script_path  = Path(args.gad_script).resolve()
                gad_grammar_path = Path(args.gad_grammar).resolve()
                # Warn if user accidentally passed the Lark grammar instead of GBNF
                if str(gad_grammar_path).endswith(".lark"):
                    print(f"  ⚠  WARNING: --gad_grammar points to a .lark file.")
                    print(f"     GAD requires GBNF format (.ebnf). Use G_ATC_v2.ebnf.")
                    print(f"     Skipping {cond_id}.")
                    (output_root / cond_id).mkdir(parents=True, exist_ok=True)
                    continue
                run_gad_condition(
                    cond_id, label, description,
                    output_root, gad_script_path,
                    ckpt_path, test_path, gad_grammar_path,
                    args.max_new_tok, dry_run=args.dry_run
                )
            elif kind == "manual":
                _print_header(cond_id, label, description)
                print(f"  ⚠  Manual condition — skipping automated run")
                print(f"     Save results to {output_root/cond_id}/test_results.json")
                (output_root / cond_id).mkdir(parents=True, exist_ok=True)
            else:
                # kind == "train"
                run_condition(
                    cond_id, label, description, kind, extra_args,
                    BASE, output_root, script_path,
                    dry_run=args.dry_run
                )

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n\nEvaluating all conditions on test set...")
    all_results = {}
    for cond_id, label, description, kind, extra_args in conditions:
        out_dir = output_root / cond_id
        if not out_dir.exists():
            continue
        r = evaluate_condition(
            cond_id, label, out_dir,
            args.data, script_path, test_path,
            vocab_path=args.vocab_path,
            phrase_path=args.phrase_path,
            domain=args.domain,
        )
        if r:
            all_results[cond_id] = r

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── Write tables ──────────────────────────────────────────────────────
    print("\nGenerating results tables...")
    write_results_table(all_results, output_root)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_training_curves(output_root, plot_dir)
    plot_comparison_bar(all_results, plot_dir)
    plot_improvement_over_sft(all_results, plot_dir)
    plot_reward_curves(output_root, plot_dir)

    elapsed = (time.time() - start_time) / 60
    print(f"\n✓ All done in {elapsed:.1f} min")
    print(f"  Tables: {output_root}/results_table.{{csv,tex}}")
    print(f"  Plots:  {plot_dir}/")


if __name__ == "__main__":
    main()
