"""
Shared model-loading + inference utilities for the three thesis evaluation scripts.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, ROOT)

# Pre-cache the installed mamba_ssm so that hybrid_net.py's sys.path injection
# of AUM/vim-mamba_ssm does not shadow the working installed version.
import mamba_ssm  # noqa: F401  — must come before any models.hybrid_net import

from dataset.dcase25 import get_test_set

CLASSES = [
    'airport', 'bus', 'metro', 'metro_station', 'park',
    'public_square', 'shopping_mall', 'street_pedestrian',
    'street_traffic', 'tram',
]

CKPT_BASE = "/home/abdalaziz-ayoub/Thesis_Hybrid_CNN_Mamba/checkpoints"

MODELS = {
    "xLSTM Balanced": {
        "ckpt":      os.path.join(CKPT_BASE, "xLSTM_2Block_Balanced/best-epoch=23-val_acc=0.48.ckpt"),
        "engine":    "xlstm",
        "slstm_at":  [1],
        "embed_dim": 32,
        "depth":     2,
        "n_mels":    256,
    },
    "xLSTM Inverted": {
        "ckpt":      os.path.join(CKPT_BASE, "xLSTM_Inverted_Hybrid/best-epoch=89-val_acc=0.49.ckpt"),
        "engine":    "xlstm",
        "slstm_at":  [0],
        "embed_dim": 32,
        "depth":     2,
        "n_mels":    256,
    },
    "Mamba (32-state)": {
        "ckpt":      os.path.join(CKPT_BASE, "Hybrid_256mels_32state/best-student-epoch=84-val_acc=0.48.ckpt"),
        "engine":    "mamba",
        "embed_dim": 28,
        "depth":     2,
        "n_mels":    256,
    },
    "GRU Baseline": {
        "ckpt":      os.path.join(CKPT_BASE, "Baseline_GRU_Depth2_Embed28/best-student-epoch=83-val_acc=0.47.ckpt"),
        "engine":    "gru",
        "embed_dim": 28,
        "depth":     2,
        "n_mels":    256,
    },
    "GRU (embed-34)": {
        "ckpt":      os.path.join(CKPT_BASE, "GRU_Depth2_Embe34/best-student-epoch=72-val_acc=0.49.ckpt"),
        "engine":    "gru",
        "embed_dim": 34,
        "depth":     2,
        "n_mels":    256,
    },
}


def load_model(cfg):
    engine = cfg["engine"]
    kwargs = dict(
        n_classes=10,
        n_mels=cfg["n_mels"],
        target_length=33,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        patch_size=4,
        d_state=32,
    )

    if engine == "xlstm":
        from models.hybrid_xlstm import get_model
        model = get_model(**kwargs, slstm_at=cfg.get("slstm_at", [1]))
    elif engine == "gru":
        from models.hybrid_gru import get_model
        model = get_model(**kwargs)
    else:
        from models.hybrid_net import get_model
        model = get_model(**kwargs)

    ckpt = torch.load(cfg["ckpt"], map_location="cpu")
    state = {k.replace("student.", "", 1): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, device, batch_size=64, num_workers=4):
    ds = get_test_set()
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    all_preds  = []
    all_labels = []

    model = model.to(device)
    for x, _, labels, _, _ in dl:
        x, labels = x.to(device), labels.to(device)
        logits = model(x)
        preds  = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_preds), torch.cat(all_labels)
