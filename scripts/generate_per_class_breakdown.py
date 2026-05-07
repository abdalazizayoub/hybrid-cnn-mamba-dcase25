"""
Per-class accuracy breakdown for all 5 thesis models.
Runs full inference on the DCASE25 test set, reports per-scene accuracy, and
saves a grouped bar chart + CSV to assets/.

Outputs:
  assets/per_class_breakdown.png
  assets/per_class_breakdown.csv
"""

import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _eval_utils import MODELS, CLASSES, load_model, run_inference

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

COLORS = [
    "#E63946", "#FF6B35", "#2196F3", "#4CAF50", "#9C27B0",
]

results = {}

for model_name, cfg in MODELS.items():
    print(f"Loading {model_name} …")
    model = load_model(cfg)
    preds, labels = run_inference(model, DEVICE)

    per_class_acc = {}
    for cls_idx, cls_name in enumerate(CLASSES):
        mask     = labels == cls_idx
        n_total  = mask.sum().item()
        n_correct = (preds[mask] == cls_idx).sum().item()
        per_class_acc[cls_name] = (n_correct / n_total * 100) if n_total > 0 else 0.0

    macro = np.mean(list(per_class_acc.values()))
    per_class_acc["__macro__"] = macro
    results[model_name] = per_class_acc
    print(f"  Macro acc: {macro:.2f}%")
    for c, v in per_class_acc.items():
        if not c.startswith("__"):
            print(f"    {c:<25} {v:.1f}%")
    print()

df = pd.DataFrame(results).T
df.index.name = "model"
df.to_csv(os.path.join(ASSETS_DIR, "per_class_breakdown.csv"))
print(f"CSV saved → {os.path.join(ASSETS_DIR, 'per_class_breakdown.csv')}\n")

# --- Plot -----------------------------------------------------------------
scene_names = [c.replace("_", " ").title() for c in CLASSES]
n_models  = len(MODELS)
n_classes = len(CLASSES)
x = np.arange(n_classes)
bar_w = 0.15

fig, ax = plt.subplots(figsize=(16, 6))

for i, (model_name, color) in enumerate(zip(MODELS.keys(), COLORS)):
    vals = [results[model_name][c] for c in CLASSES]
    offset = (i - n_models / 2 + 0.5) * bar_w
    bars = ax.bar(x + offset, vals, bar_w, label=model_name,
                  color=color, alpha=0.85, edgecolor="white", linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels(scene_names, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title(
    "Per-Class (Scene) Accuracy — All Models on DCASE25 Test Set",
    fontsize=12, fontweight="bold",
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylim(0, 105)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

macro_vals = [results[m]["__macro__"] for m in MODELS.keys()]
summary = "  ".join(f"{m}: {v:.1f}%" for m, v in zip(MODELS.keys(), macro_vals))
ax.set_xlabel(f"Acoustic Scene\n\nMacro accuracy — {summary}", fontsize=9)

plt.tight_layout()
out_png = os.path.join(ASSETS_DIR, "per_class_breakdown.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved → {out_png}")
