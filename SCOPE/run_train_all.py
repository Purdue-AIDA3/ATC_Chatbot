#!/usr/bin/env python3
"""
run_train_all.py — Unified SCOPE experiment runner
====================================================
Runs training, evaluation, and/or IFEval for any combination of:
  - Domains:     atc, smcp (maritime)
  - Models:      gpt2, llama, qwen
  - Conditions:  C1, C2, C3, C4, C11 (and any others defined in the runners)
  - Tasks:       train, ifeval, both

Usage examples
--------------
# Full ATC run — all models, all conditions:
python run_train_all.py --domain atc --models gpt2 llama qwen --conditions C2 C3 C11 C4

# Maritime only, GPT-2, two conditions:
python run_train_all.py --domain smcp --models gpt2 --conditions C2 C11

# ATC Llama + Qwen + IFEval afterwards:
python run_train_all.py --domain atc --models llama qwen --conditions C2 C11 --tasks train ifeval

# IFEval only on already-trained models:
python run_train_all.py --domain atc --models gpt2 llama qwen --conditions C2 C11 --tasks ifeval

# Colab path:
python run_train_all.py --domain smcp --scope_dir /content/drive/MyDrive/ATC_Chatbot/SCOPE

# Gilbreth (default path):
python run_train_all.py --domain smcp
"""

import argparse
import json
import subprocess
import sys
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified SCOPE runner — ATC/SMCP × GPT-2/Llama/Qwen × train/IFEval"
    )
    parser.add_argument("--scope_dir", default="/scratch/gilbreth/oawoyera/scope",
                        help="Root directory containing all SCOPE scripts and data files. "
                             "For Colab: /content/drive/MyDrive/ATC_Chatbot/SCOPE")
    parser.add_argument("--domain", default="atc", choices=["atc", "smcp"],
                        help="Domain to run: atc (ATC/ICAO) or smcp (maritime/IMO)")
    parser.add_argument("--models", nargs="+", default=["gpt2", "llama", "qwen"],
                        choices=["gpt2", "llama", "qwen"],
                        help="Models to run")
    parser.add_argument("--conditions", nargs="+", default=["C2", "C3", "C11", "C4"],
                        help="Conditions to run (e.g. C2 C3 C11 C4)")
    parser.add_argument("--tasks", nargs="+", default=["train"],
                        choices=["train", "ifeval"],
                        help="Tasks to run: train, ifeval, or both")
    parser.add_argument("--output_root", default=None,
                        help="Override output directory "
                             "(default: <scope_dir>/results_<domain>)")
    parser.add_argument("--ifeval_conditions", nargs="+", default=["C2", "C11"],
                        help="Which conditions to run IFEval on (default: C2 C11)")
    parser.add_argument("--ifeval_batch", type=int, default=8,
                        help="Batch size for IFEval (default: 8)")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DOMAIN CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

def build_domain_config(args):
    """Return all domain-specific paths and args."""
    S = Path(args.scope_dir)

    if args.domain == "atc":
        return {
            "data":        str(S / "atc_pairs.json"),
            "test_data":   str(S / "atc_test.json"),
            "grammar":     str(S / "G_ATC.lark"),
            "vocab_path":  str(S / "vocab_ATC.json"),
            "phrase_path": str(S / "ngram_whitelist_ATC.json"),
            "domain":      "atc",
            "label":       "ATC",
        }
    else:  # smcp
        return {
            "data":        str(S / "smcp_pairs.json"),
            "test_data":   str(S / "smcp_test.json"),
            "grammar":     str(S / "G_SMCP.lark"),
            "vocab_path":  str(S / "vocab_SMCP.json"),
            "phrase_path": str(S / "ngram_whitelist_SMCP.json"),
            "domain":      "smcp",
            "label":       "Maritime (SMCP)",
        }


def domain_cli_args(dcfg):
    """Flat CLI arg list from domain config."""
    return [
        "--data",        dcfg["data"],
        "--test_data",   dcfg["test_data"],
        "--domain",      dcfg["domain"],
        "--grammar",     dcfg["grammar"],
        "--vocab_path",  dcfg["vocab_path"],
        "--phrase_path", dcfg["phrase_path"],
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RESULT PATHS
# ═══════════════════════════════════════════════════════════════════════════════

def result_root(args):
    S = Path(args.scope_dir)
    if args.output_root:
        return Path(args.output_root)
    suffix = "atc" if args.domain == "atc" else "maritime"
    return S / f"results_{suffix}"


def model_result_dir(results_root, model):
    """Where a model's condition results live."""
    if model == "gpt2":
        return results_root / "gpt"
    return results_root / model


def checkpoint_path(results_root, model, condition):
    return model_result_dir(results_root, model) / condition / "best"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def run_training(args, dcfg, results_root):
    S = Path(args.scope_dir)
    domain_args = domain_cli_args(dcfg)

    print(f"\n{'#'*65}")
    print(f"  TRAINING — {dcfg['label']} | {', '.join(args.models)}")
    print(f"  Conditions: {', '.join(args.conditions)}")
    print(f"{'#'*65}")

    for model in tqdm(args.models, desc="Models", position=0):
        for condition in tqdm(args.conditions,
                              desc=f"{model} conditions", position=1, leave=False):

            print(f"\n{'='*60}")
            print(f"  {model.upper()} / {condition} / {dcfg['domain'].upper()}")
            print(f"{'='*60}")

            if model == "gpt2":
                cmd = [
                    sys.executable,
                    str(S / "run_all_conditions_general.py"),
                    "--model",       "gpt2-large",
                    "--script",      str(S / "scope_train_general.py"),
                    "--gcd_script",  str(S / "evaluate_gcd_general.py"),
                    "--output_root", str(model_result_dir(results_root, model)),
                    "--conditions",  condition,
                ] + domain_args

            else:  # llama or qwen
                cmd = [
                    sys.executable,
                    str(S / "run_new_models_general.py"),
                    "--models",       model,
                    "--train_script", str(S / "scope_train_general.py"),
                    "--gcd_script",   str(S / "evaluate_gcd_general.py"),
                    "--output_root",  str(results_root),
                    "--conditions",   condition,
                ] + domain_args

            result = subprocess.run(cmd, check=False)
            status = "✓ done" if result.returncode == 0 else f"✗ FAILED (exit {result.returncode})"
            print(f"\n  {model.upper()} / {condition}: {status}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. IFEVAL
# ═══════════════════════════════════════════════════════════════════════════════

IFEVAL_CONDITIONS = {
    # condition_id → (label_suffix, needs_trained_checkpoint)
    "C1":  ("vanilla",  False),   # base model — no checkpoint
    "C2":  ("sft",      True),
    "C3":  ("dpo",      True),
    "C11": ("scope",    True),
}

# Base model IDs per model key
BASE_MODEL_IDS = {
    "gpt2":  "gpt2-large",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
}

def run_ifeval(args, dcfg, results_root):
    S = Path(args.scope_dir)
    domain_tag = dcfg["domain"]
    ifeval_dir = S / f"results_ifeval_{domain_tag}"
    ifeval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*65}")
    print(f"  IFEVAL — {dcfg['label']} | {', '.join(args.models)}")
    print(f"  Conditions: {', '.join(args.ifeval_conditions)}")
    print(f"{'#'*65}")

    # Install lm-eval if needed
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "lm-eval", "--quiet"],
        check=False
    )

    completed = {}

    for model in args.models:
        for condition in args.ifeval_conditions:
            cond_info = IFEVAL_CONDITIONS.get(condition)
            if cond_info is None:
                print(f"  WARNING: IFEval not configured for condition {condition} — skipping")
                continue

            suffix, needs_ckpt = cond_info
            label = f"{model}_{condition}_{suffix}_{domain_tag}"
            out_file = ifeval_dir / f"{label}_ifeval.json"

            if out_file.exists():
                print(f"  ✓ {label} — already done, skipping")
                completed[label] = out_file
                continue

            # Resolve model path
            if not needs_ckpt:
                model_path = BASE_MODEL_IDS[model]
            else:
                ckpt = checkpoint_path(results_root, model, condition)
                if not ckpt.exists():
                    print(f"  ⚠ {label} — checkpoint not found at {ckpt}, skipping")
                    continue
                model_path = str(ckpt)

            print(f"\n{'='*60}")
            print(f"  IFEval: {label}")
            print(f"  Model:  {model_path}")
            print(f"{'='*60}")

            cmd = [
                sys.executable, "-m", "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={model_path},dtype=bfloat16",
                "--tasks", "ifeval",
                "--device", "cuda",
                "--batch_size", str(args.ifeval_batch),
                "--output_path", str(out_file),
                "--log_samples",
            ]

            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                print(f"  ✓ Saved: {out_file}")
                completed[label] = out_file
            else:
                print(f"  ✗ IFEval FAILED for {label}")

    # Print summary
    _print_ifeval_summary(completed, domain_tag)


def _print_ifeval_summary(completed, domain_tag):
    print(f"\n{'='*60}")
    print(f"  IFEVAL SUMMARY — {domain_tag.upper()}")
    print(f"{'='*60}")
    print(f"  {'Condition':<30} {'Prompt Acc':>12} {'Instr Acc':>12}")
    print(f"  {'-'*56}")

    for label, out_file in sorted(completed.items()):
        try:
            with open(out_file) as f:
                data = json.load(f)
            res = data.get("results", {}).get("ifeval", {})
            pa  = res.get("prompt_level_strict_acc,none",
                          res.get("prompt_level_strict_acc", None))
            ia  = res.get("inst_level_strict_acc,none",
                          res.get("inst_level_strict_acc", None))
            pa_str = f"{pa*100:.1f}%" if pa is not None else "—"
            ia_str = f"{ia*100:.1f}%" if ia is not None else "—"
        except Exception:
            pa_str = ia_str = "error"
        print(f"  {label:<30} {pa_str:>12} {ia_str:>12}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COLLECT RESULTS & PLOT
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_KEYS  = ["C_tok", "C_phr", "C_cfg"]
MODEL_COLORS = {"gpt2": "#4C72B0", "llama": "#DD8452", "qwen": "#55A868"}

def collect_and_plot(args, dcfg, results_root):
    print(f"\n{'#'*65}")
    print(f"  RESULTS — {dcfg['label']}")
    print(f"{'#'*65}")

    all_results = {}
    for model in args.models:
        all_results[model] = {}
        for condition in args.conditions:
            path = model_result_dir(results_root, model) / condition / "test_results.json"
            if path.exists():
                with open(path) as f:
                    r = json.load(f)
                all_results[model][condition] = {k: r.get(k, 0.0) for k in METRIC_KEYS}
            else:
                all_results[model][condition] = None

    # Print table
    print(f"\n{'='*70}")
    print(f"  {'Model':<10} {'Condition':<10} {'C_tok':>8} {'C_phr':>8} {'C_cfg':>8}")
    print(f"  {'-'*60}")
    for model in args.models:
        for condition in args.conditions:
            r = all_results[model].get(condition)
            if r:
                print(f"  {model:<10} {condition:<10} "
                      f"{r['C_tok']:>8.4f} {r['C_phr']:>8.4f} {r['C_cfg']:>8.4f}")
            else:
                print(f"  {model:<10} {condition:<10} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    print(f"  {'='*60}")

    # Grouped bar charts per metric
    x         = np.arange(len(args.conditions))
    bar_width  = 0.25
    n_models   = len(args.models)
    offsets    = np.linspace(-bar_width, bar_width, n_models)

    for metric in METRIC_KEYS:
        fig, ax = plt.subplots(figsize=(max(8, len(args.conditions) * 2), 5))
        for i, model in enumerate(args.models):
            values = [
                (all_results[model].get(c) or {}).get(metric, 0.0)
                for c in args.conditions
            ]
            ax.bar(x + offsets[i], values, width=bar_width,
                   label=model.upper(),
                   color=MODEL_COLORS.get(model, f"C{i}"),
                   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(args.conditions)
        ax.set_xlabel("Condition")
        ax.set_ylabel(metric)
        ax.set_title(f"{dcfg['label']} — {metric} by Model and Condition")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        out = results_root / f"{dcfg['domain']}_{metric}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")

    # C11 compliance profile if C11 was run
    if "C11" in args.conditions:
        fig, ax = plt.subplots(figsize=(7, 5))
        for model in args.models:
            r = all_results[model].get("C11")
            if r:
                ax.plot(METRIC_KEYS, [r[m] for m in METRIC_KEYS],
                        marker="o", label=model.upper(),
                        color=MODEL_COLORS.get(model), linewidth=2)
        ax.set_xlabel("Compliance Metric (coarse → fine)")
        ax.set_ylabel("Score")
        ax.set_title(f"SCOPE-full (C11) — Compliance Profile ({dcfg['label']})")
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        out = results_root / f"{dcfg['domain']}_C11_profile.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    dcfg    = build_domain_config(args)
    rroot   = result_root(args)
    rroot.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*65}")
    print(f"  SCOPE — Unified Runner")
    print(f"  Domain:     {dcfg['label']}")
    print(f"  Models:     {', '.join(args.models)}")
    print(f"  Conditions: {', '.join(args.conditions)}")
    print(f"  Tasks:      {', '.join(args.tasks)}")
    print(f"  Output:     {rroot}")
    print(f"{'#'*65}")

    if "train" in args.tasks:
        run_training(args, dcfg, rroot)
        collect_and_plot(args, dcfg, rroot)

    if "ifeval" in args.tasks:
        run_ifeval(args, dcfg, rroot)

    print(f"\n{'#'*65}")
    print(f"  All tasks complete.")
    print(f"{'#'*65}\n")


if __name__ == "__main__":
    main()
