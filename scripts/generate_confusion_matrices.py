"""
Confusion matrices for all 5 thesis models on the DCASE25 test set.
Saves one PNG per model to assets/confusion_matrix_<model_name>.png

Outputs:
  assets/confusion_matrix_xlstm_balanced.png
  assets/confusion_matrix_xlstm_inverted.png
  assets/confusion_matrix_mamba_32state.png
  assets/confusion_matrix_gru_baseline.png
  assets/confusion_matrix_gru_embed34.png
"""

import os
import re
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _eval_utils import MODELS, CLASSES, load_model, run_inference

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

FULL_LABELS = [c.replace("_", " ").title() for c in CLASSES]


def safe_filename(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


for model_name, cfg in MODELS.items():
    print(f"Loading {model_name} …")
    model = load_model(cfg)
    preds, labels = run_inference(model, DEVICE)

    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=list(range(len(CLASSES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    macro = cm.diagonal().sum() / cm.sum() * 100
    print(f"  Macro accuracy: {macro:.2f}%")

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor("white")

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    n = len(CLASSES)
    for r in range(n):
        for c in range(n):
            val = cm_norm[r, c]
            raw = cm[r, c]
            text_color = "white" if val > 0.55 else "black"
            ax.text(c, r, f"{val:.2f}\n({raw})",
                    ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(FULL_LABELS, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(FULL_LABELS, fontsize=9)
    ax.set_xlabel("Predicted Scene", fontsize=11, labelpad=10)
    ax.set_ylabel("True Scene", fontsize=11)
    ax.set_title(
        f"{model_name}\nDCASE25 Test Set — Macro Accuracy: {macro:.2f}%",
        fontsize=12, fontweight="bold", pad=14,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised accuracy", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    )

    plt.tight_layout()
    fname = f"confusion_matrix_{safe_filename(model_name)}.png"
    out   = os.path.join(ASSETS_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}\n")

print("All confusion matrices saved.")
