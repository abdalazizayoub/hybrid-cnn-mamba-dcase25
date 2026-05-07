"""
Pull real per-epoch training curves from the wandb API and plot:
  - Left panel:  Validation accuracy (macro avg) vs epoch
  - Right panel: Training loss vs epoch

Outputs: assets/training_curves.png
"""

import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

WANDB_PROJECT = "DCASE25_Hybrid_Architecture"

# Display name → wandb run ID
RUNS = {
    "xLSTM Balanced":   "u98gl5ud",
    "xLSTM Inverted":   "u57xp03c",
    "Mamba (32-state)": "qrtvtccy",
    "GRU Baseline":     "3q8cq147",
    "GRU (embed-34)":   "cxee52yq",
}

COLORS = {
    "xLSTM Balanced":   "#E63946",
    "xLSTM Inverted":   "#FF6B35",
    "Mamba (32-state)": "#2196F3",
    "GRU Baseline":     "#4CAF50",
    "GRU (embed-34)":   "#9C27B0",
}


def fetch_curves(run_id):
    """Return (val_epochs, val_accs, train_epochs, train_losses) from wandb history."""
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")

    val_epochs, val_accs = [], []
    train_epochs, train_losses = [], []

    for row in run.scan_history():
        epoch = row.get("epoch")
        if epoch is None:
            continue
        epoch = int(epoch)

        val_acc = row.get("val/macro_avg_acc")
        if val_acc is not None:
            val_epochs.append(epoch)
            val_accs.append(float(val_acc) * 100)

        train_loss = row.get("train/loss_epoch")
        if train_loss is not None:
            train_epochs.append(epoch)
            train_losses.append(float(train_loss))

    # de-duplicate: keep last entry per epoch (final value logged that epoch)
    def dedup(epochs, vals):
        seen = {}
        for e, v in zip(epochs, vals):
            seen[e] = v
        pairs = sorted(seen.items())
        return [p[0] for p in pairs], [p[1] for p in pairs]

    val_epochs,   val_accs     = dedup(val_epochs,   val_accs)
    train_epochs, train_losses = dedup(train_epochs, train_losses)
    return val_epochs, val_accs, train_epochs, train_losses


print("Fetching wandb history …")
data = {}
for model_name, run_id in RUNS.items():
    print(f"  {model_name} ({run_id}) …", end=" ", flush=True)
    ve, va, te, tl = fetch_curves(run_id)
    data[model_name] = (ve, va, te, tl)
    best = max(va) if va else 0
    print(f"{len(ve)} val epochs, best val acc = {best:.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("white")
for ax in (ax_acc, ax_loss):
    ax.set_facecolor("#f9f9f9")

for model_name, (ve, va, te, tl) in data.items():
    color = COLORS[model_name]

    # ── Val accuracy ──────────────────────────────────────────────
    if va:
        best_acc = max(va)
        best_ep  = ve[va.index(best_acc)]
        ax_acc.plot(ve, va, color=color, linewidth=1.8, label=f"{model_name}  ({best_acc:.1f}%)")
        ax_acc.plot(best_ep, best_acc, "*", color=color, markersize=11,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=5)

    # ── Train loss ────────────────────────────────────────────────
    if tl:
        ax_loss.plot(te, tl, color=color, linewidth=1.8, label=model_name)

# ── Val accuracy axes ──────────────────────────────────────────────────────
ax_acc.set_xlabel("Epoch", fontsize=11)
ax_acc.set_ylabel("Validation Accuracy (%)", fontsize=11)
ax_acc.set_title("Validation Accuracy vs Epoch", fontsize=12, fontweight="bold")
ax_acc.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax_acc.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
ax_acc.set_xlim(left=0)
ax_acc.legend(fontsize=9, loc="lower right", framealpha=0.92,
              title="Model (best val acc ★)", title_fontsize=8.5)
ax_acc.annotate("★ = best checkpoint", xy=(0.01, 0.02),
                xycoords="axes fraction", fontsize=8, color="grey")

# ── Train loss axes ────────────────────────────────────────────────────────
ax_loss.set_xlabel("Epoch", fontsize=11)
ax_loss.set_ylabel("Training Loss (cross-entropy)", fontsize=11)
ax_loss.set_title("Training Loss vs Epoch", fontsize=12, fontweight="bold")
ax_loss.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
ax_loss.set_xlim(left=0)
ax_loss.legend(fontsize=9, loc="upper right", framealpha=0.92)

fig.suptitle(
    "DCASE25 Hybrid Models — Real Training Curves (source: Weights & Biases)",
    fontsize=13, fontweight="bold", y=1.02,
)
plt.tight_layout()

out = os.path.join(ASSETS_DIR, "training_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out}")
