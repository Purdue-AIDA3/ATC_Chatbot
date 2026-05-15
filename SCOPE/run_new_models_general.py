#!/usr/bin/env python3
"""
run_new_models_general.py — Run SCOPE on Llama-3.1-8B and Qwen2.5-7B
Domain-agnostic: ATC (ICAO) and maritime (SMCP) via --domain.

Conditions per model:
  C2:  SFT baseline
  C3:  DPO
  C4:  GCD on SFT checkpoint
  C11: SCOPE-full (proposed)

Usage:
  python run_new_models_general.py \
    --models llama qwen --conditions C2 C3 C11 C4 \
    --domain smcp \
    --grammar G_SMCP.lark \
    --vocab_path vocab_SMCP.json \
    --phrase_path ngram_whitelist_SMCP.json \
    --data smcp_pairs.json --test_data smcp_test.json \
    --train_script scope_train_general.py \
    --gcd_script evaluate_gcd_general.py \
    --output_root results_maritime

  python run_new_models_general.py \
    --models llama qwen \
    --conditions C2 C3 C11 C4 \
    --domain atc \
    --output_root results_new_models/
"""

import os, sys, subprocess, time, json, argparse
from pathlib import Path

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "llama": {
        "model_id":  "meta-llama/Llama-3.1-8B-Instruct",
        "shortname": "Llama-3.1-8B",
        "lr":        "1e-5",
        "epochs":    "3",
        "batch":     "4",
        "grad_accum":"4",
        "M_samples": "2",
    },
    "qwen": {
        "model_id":  "Qwen/Qwen2.5-7B-Instruct",
        "shortname": "Qwen2.5-7B",
        "lr":        "1e-5",
        "epochs":    "3",
        "batch":     "4",
        "grad_accum":"4",
        "M_samples": "2",
    },
}

# ── Condition definitions ─────────────────────────────────────────────────────
CONDITIONS = {
    "C2": {
        "label": "Standard SFT",
        "kind":  "train",
        "extra": ["--lambda_ce", "1.0", "--no_ltok", "--no_lphr", "--no_lcfg"],   # pure L_CE only (matches GPT-2 C2)
    },
    "C3": {
        "label": "DPO",
        "kind":  "train",
        "extra": ["--lambda_ce", "1.0", "--dpo", "--dpo_beta", "0.1",
                  "--no_ltok", "--no_lphr", "--no_lcfg"],   # DPO only (matches GPT-2 C3)
    },
    "C11": {
        "label": "SCOPE-full (proposed)",
        "kind":  "train",
        "extra": ["--lambda_ce", "0.5"],
    },
    "C4": {
        "label": "GCD on SFT",
        "kind":  "gcd",
        "source": "C2",   # apply GCD to C2/best
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def hf_login():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN not set. Add it to Colab Secrets.")
        sys.exit(1)
    try:
        from huggingface_hub import login
        login(token=token)
        print("✓ HuggingFace authenticated")
    except Exception as e:
        print(f"WARNING: HF login issue: {e} — proceeding")

def run_subprocess(cmd, log_path, label):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        proc.wait()
    elapsed = (time.time() - t0) / 60
    ok = proc.returncode == 0
    print(f"\n  {'✓ Complete' if ok else '✗ FAILED'} ({elapsed:.1f} min)")
    return ok

def load_metrics(results_path):
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None

def print_metrics(cond_id, label, r):
    if r:
        print(f"    {cond_id} {label}: "
              f"C_tok={r.get('C_tok',0):.4f}  "
              f"C_phr={r.get('C_phr',0):.4f}  "
              f"C_cfg={r.get('C_cfg',0):.4f}")

# ── Run one condition for one model ──────────────────────────────────────────
def run_condition(cond_id, model_key, model_info, args, out_root):
    cond      = CONDITIONS[cond_id]
    model_dir = out_root / model_key
    cond_dir  = model_dir / cond_id
    done_flag = cond_dir / "DONE"
    label     = f"{model_info['shortname']} / {cond['label']}"
    py        = sys.executable

    if done_flag.exists():
        print(f"\n  ✓ {label} — already complete, skipping")
        return load_metrics(cond_dir / "test_results.json")

    if cond["kind"] == "gcd":
        # GCD: evaluate_gcd_general.py on the SFT checkpoint
        src_ckpt = model_dir / cond["source"] / "best"
        if not src_ckpt.exists():
            print(f"\n  ✗ {label} — source checkpoint {src_ckpt} not found")
            print(f"    Run {cond['source']} first.")
            return None
        cmd = [
            py, args.gcd_script,
            "--model",   str(src_ckpt),
            "--data",    args.test_data,
            "--grammar", args.grammar,
            "--output",  str(cond_dir),
            "--vocab",   args.vocab_path,
            "--phrase",  args.phrase_path,
            "--domain",  args.domain,
        ]

    else:  # train condition
        common = [
            "--model",      model_info["model_id"],
            "--data",       args.data,
            "--test_data",  args.test_data,
            "--output",     str(cond_dir),
            "--epochs",     model_info["epochs"],
            "--batch_size", model_info["batch"],
            "--grad_accum", model_info["grad_accum"],
            "--lr",         model_info["lr"],
            "--M_samples",  model_info["M_samples"],
            "--max_new_tok","64",
            "--seed",       "42",
            "--lambda_tok", "1.0",
            "--lambda_phr", "0.5",
            "--lambda_cfg", "0.3",
            "--vocab_path", args.vocab_path,
            "--phrase_path",args.phrase_path,
            "--grammar",    args.grammar,
            "--domain",     args.domain,
            "--use_chat_template",
            "--gradient_checkpointing",
        ]
        cmd = [py, args.train_script] + common + cond["extra"]

    ok = run_subprocess(cmd, cond_dir / "run.log", label)
    if ok:
        done_flag.touch()   # mark complete so --resume skips this
    r  = load_metrics(cond_dir / "test_results.json") if ok else None
    print_metrics(cond_id, cond["label"], r)
    return r

# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(all_results, gpt2_results=None):
    print(f"\n\n{'='*78}")
    print(f"{'MULTI-MODEL RESULTS SUMMARY':^78}")
    print(f"{'='*78}")
    print(f"{'Model':<16} {'Method':<25} {'C_tok':>6} {'C_phr':>6} {'C_cfg':>6}")
    print("-" * 78)

    # GPT-2 reference rows
    if gpt2_results:
        for cid, r in gpt2_results.items():
            label = CONDITIONS.get(cid, {}).get("label", cid)
            print(f"  {'GPT-2 Large':<14} {label:<25} "
                  f"{r.get('C_tok',0):>6.4f} {r.get('C_phr',0):>6.4f} "
                  f"{r.get('C_cfg',0):>6.4f}")
        print()

    for model_key, cond_results in all_results.items():
        shortname = MODEL_REGISTRY[model_key]["shortname"]
        for cid, r in cond_results.items():
            if r is None:
                continue
            label = CONDITIONS.get(cid, {}).get("label", cid)
            marker = " ◀" if cid == "C11" else ""
            print(f"  {shortname:<14} {label:<25} "
                  f"{r.get('C_tok',0):>6.4f} {r.get('C_phr',0):>6.4f} "
                  f"{r.get('C_cfg',0):>6.4f}{marker}")
        print()

    print("=" * 78)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",       nargs="+", default=["llama", "qwen"],
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--conditions",   nargs="+", default=["C2","C3","C11","C4"])
    parser.add_argument("--output_root",  default="results_new_models")
    parser.add_argument("--data",         default="atc_pairs.json")
    parser.add_argument("--test_data",    default="atc_test.json")
    parser.add_argument("--train_script", default="scope_train_general.py")
    parser.add_argument("--gcd_script",   default="evaluate_gcd_general.py")
    parser.add_argument("--grammar",      default="G_ATC.lark")
    parser.add_argument("--vocab_path",   default="vocab_ATC.json",
                        help="Vocabulary whitelist JSON (V_domain)")
    parser.add_argument("--phrase_path",  default="ngram_whitelist_ATC.json",
                        help="N-gram whitelist JSON (P_domain)")
    parser.add_argument("--domain",       default="atc",
                        choices=["atc", "smcp"],
                        help="Domain for system prompt")
    parser.add_argument("--gpt2_results", default="results2",
                        help="Path to GPT-2 results for comparison table")
    args = parser.parse_args()

    hf_login()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load GPT-2 reference results if available
    gpt2_results = {}
    gpt2_dir = Path(args.gpt2_results)
    for cid in args.conditions:
        p = gpt2_dir / cid / "test_results.json"
        if p.exists():
            with open(p) as f:
                gpt2_results[cid] = json.load(f)

    all_results = {}
    for model_key in args.models:
        model_info = MODEL_REGISTRY[model_key]
        print(f"\n\n{'#'*72}")
        print(f"# MODEL: {model_info['shortname']} ({model_info['model_id']})")
        print(f"{'#'*72}")

        cond_results = {}
        # Ensure C2 runs before C4 (GCD depends on C2 checkpoint)
        ordered_conditions = sorted(
            args.conditions,
            key=lambda c: 99 if c == "C4" else 0
        )
        for cond_id in ordered_conditions:
            # GCD must come after C2 — reorder if needed
            if cond_id == "C4" and "C2" not in cond_results:
                print(f"  Skipping C4 (GCD) — C2 not yet complete for {model_key}")
                continue
            r = run_condition(cond_id, model_key, model_info, args, out_root)
            cond_results[cond_id] = r

        # Run C4 after C2 if C2 just completed
        if "C4" in args.conditions and "C4" not in cond_results:
            r = run_condition("C4", model_key, model_info, args, out_root)
            cond_results["C4"] = r

        all_results[model_key] = cond_results

    print_summary(all_results, gpt2_results if gpt2_results else None)

    # Save combined results JSON
    summary_path = out_root / "multi_model_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            k: {cid: r for cid, r in v.items() if r}
            for k, v in all_results.items()
        }, f, indent=2)
    print(f"\nSaved: {summary_path}")

if __name__ == "__main__":
    main()
