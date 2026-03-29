"""
generate_charts.py
Run this script from the repo root to regenerate the result SVGs in assets/.

    python generate_charts.py

Requires: matplotlib
    pip install matplotlib
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Output directory ──────────────────────────────────────────────────────────
ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
MODELS = ["GPT-2", "Llama-2", "Qwen-3"]

DATA = {
    "BERTScore": {
        "Vanilla":                      [0.772, 0.785, 0.795],
        "Standard fine-tuning":         [0.787, 0.788, 0.795],
        "Grammar-informed fine-tuning": [0.809, 0.814, 0.794],
    },
    "Semantic Similarity": {
        "Vanilla":                      [0.179, 0.241, 0.325],
        "Standard fine-tuning":         [0.187, 0.148, 0.302],
        "Grammar-informed fine-tuning": [0.296, 0.319, 0.295],
    },
    "ATC Token Compliance": {
        "Vanilla":                      [0.292, 0.278, 0.171],
        "Standard fine-tuning":         [0.221, 0.298, 0.313],
        "Grammar-informed fine-tuning": [0.611, 0.662, 0.514],
    },
}

Y_LIMITS = {
    "BERTScore":            (0.76, 0.83),
    "Semantic Similarity":  (0.00, 0.45),
    "ATC Token Compliance": (0.00, 0.80),
}

FILENAMES = {
    "BERTScore":            "bertscore.svg",
    "Semantic Similarity":  "semantic_similarity.svg",
    "ATC Token Compliance": "atc_compliance.svg",
}

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "Vanilla":                      "#4F46E5",   # indigo
    "Standard fine-tuning":         "#B45309",   # amber
    "Grammar-informed fine-tuning": "#059669",   # emerald
}

EDGE_COLORS = {
    "Vanilla":                      "#3730A3",
    "Standard fine-tuning":         "#92400E",
    "Grammar-informed fine-tuning": "#047857",
}

BAR_WIDTH  = 0.22
GROUP_GAP  = 0.88       # distance between group centres
FONT_FAMILY = "DejaVu Sans"

plt.rcParams.update({
    "font.family":        FONT_FAMILY,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.color":         "#e5e7eb",
    "grid.linewidth":     0.8,
    "grid.linestyle":     "-",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

VARIANTS = list(COLORS.keys())
N_GROUPS  = len(MODELS)
N_BARS    = len(VARIANTS)

# Offsets so bars are centred on each group tick
offsets = np.linspace(
    -(N_BARS - 1) / 2 * BAR_WIDTH,
     (N_BARS - 1) / 2 * BAR_WIDTH,
    N_BARS,
)

x = np.arange(N_GROUPS) * GROUP_GAP


# ── Chart generator ───────────────────────────────────────────────────────────
def make_chart(metric: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.6))

    variant_data = DATA[metric]
    y_min, y_max = Y_LIMITS[metric]

    for i, variant in enumerate(VARIANTS):
        values = variant_data[variant]
        bars = ax.bar(
            x + offsets[i],
            values,
            width=BAR_WIDTH * 0.92,
            color=COLORS[variant],
            edgecolor=EDGE_COLORS[variant],
            linewidth=0.8,
            label=variant,
            zorder=3,
        )
        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (y_max - y_min) * 0.012,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7.5,
                color="#374151",
            )

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11, color="#374151")
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.12)
    ax.yaxis.set_tick_params(labelsize=9, labelcolor="#6b7280")
    ax.set_ylabel(metric, fontsize=10, color="#374151", labelpad=8)

    # Spine styling
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")

    # Legend (custom patches, placed above plot)
    patches = [
        mpatches.Patch(facecolor=COLORS[v], edgecolor=EDGE_COLORS[v], linewidth=0.8, label=v)
        for v in VARIANTS
    ]
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handlelength=1.2,
        handleheight=0.9,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(ASSETS_DIR, FILENAMES[metric])
    fig.savefig(out_path, format="svg", bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating charts...")
    for metric in DATA:
        make_chart(metric)
    print("Done. Charts saved to assets/")
