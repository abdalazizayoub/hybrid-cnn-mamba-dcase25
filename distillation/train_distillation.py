import os
import sys
import math
import argparse
from argparse import Namespace
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torchaudio

# ==================================================================
# 1. BULLETPROOF IMPORTS (Avoiding Namespace Collisions)
# ==================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# A. Dataset & Helpers (Safe)
from dataset.dcase25 import get_training_set, get_test_set
from helpers import complexity

# B. EfficientAT Teacher (Loads from Knoweledge_Distiliation/models)
from models.mn.model import get_model as get_mn

# C. Mamba Student (Loads from dcase2025_task1_audio_mamba/models)
# We use importlib to explicitly load the parent's net.py to avoid 
# colliding with the EfficientAT 'models' folder!
net_path = os.path.join(parent_dir, "models", "net.py")
spec = importlib.util.spec_from_file_location("mamba_net", net_path)
mamba_net = importlib.util.module_from_spec(spec)
sys.modules["mamba_net"] = mamba_net
spec.loader.exec_module(mamba_net)
get_mamba_student = mamba_net.get_model

# ==================================================================
# 2. Helper Functions & KD Loss
# ==================================================================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

class KDLoss(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels, alpha):
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(student_log_probs, soft_targets) * (self.temperature ** 2)
        hard_loss = F.cross_entropy(student_logits, labels)
        return (alpha * soft_loss) + ((1.0 - alpha) * hard_loss)

# ==================================================================
# 3. Lightning Module
# ==================================================================
class DistillationModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        if isinstance(config, dict):
            config = Namespace(**config)
        self.save_hyperparameters(config)
        self.config = config

        # --- Loss & Alpha Setup ---
        self.loss_fn = KDLoss(temperature=config.temperature)
        self.current_alpha = config.start_alpha  

        # --- EfficientAT Teacher Setup ---
        print(f"Loading Fine-Tuned EfficientAT Teacher from: {config.teacher_ckpt}")
        self.teacher_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=1024, win_length=800, hop_length=320, n_mels=128, f_min=0.0, f_max=None
        )
        
        # Load the exact width_mult used during teacher training!
        self.teacher_net = get_mn(num_classes=config.n_classes, pretrained_name="mn10_as", width_mult=1.0)
        
        # Strip Lightning 'net.' prefix to load raw weights into the CNN
        ckpt = torch.load(config.teacher_ckpt, map_location="cpu")
        new_state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith("net."):
                new_state_dict[k.replace("net.", "")] = v
        self.teacher_net.load_state_dict(new_state_dict, strict=True)
        
        # Freeze Teacher
        self.teacher_net.eval()
        for param in self.teacher_net.parameters():
            param.requires_grad = False

        # --- Mamba Student Setup ---
        self.student_mel = torch.nn.Sequential(
            torchaudio.transforms.Resample(orig_freq=config.orig_sample_rate, new_freq=config.sample_rate),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate, n_fft=config.n_fft,
                win_length=config.window_length, hop_length=config.hop_length,
                n_mels=config.n_mels, f_min=config.f_min, f_max=config.f_max
            )
        )
        p_size_s = tuple(int(x) for x in config.patch_size.split(',')) if isinstance(config.patch_size, str) else tuple(config.patch_size)
        self.student = get_mamba_student(
            n_classes=config.n_classes, n_mels=config.n_mels, target_length=config.target_length,
            embed_dim=config.embed_dim, depth=config.depth, patch_size=p_size_s, d_state=config.d_state
        )

    def get_teacher_logits(self, x):
        """Processes audio into Base-10 dB Spectrogram for EfficientAT"""
        with torch.no_grad():
            with torch.autocast(device_type=x.device.type, enabled=False):
                t_features = x.to(torch.float32)
                t_features = self.teacher_mel(t_features)
                t_features = 10.0 * torch.log10(t_features + 1e-8)
            t_features = t_features.unsqueeze(1)
            t_logits, _ = self.teacher_net(t_features)
            return t_logits

    def get_student_logits(self, x):
        """Processes audio into Natural Log Spectrogram for Mamba SSM"""
        with torch.autocast(device_type=x.device.type, enabled=False):
            s_features = x.to(torch.float32)
            s_features = self.student_mel(s_features)
            s_features = torch.log(s_features + 1e-8) 
        s_features = s_features.unsqueeze(1)
        return self.student(s_features)

    def on_train_epoch_start(self):
        # Alpha Annealing Logic
        progress = self.current_epoch / max(1, self.config.n_epochs)
        alpha_range = self.config.start_alpha - self.config.end_alpha
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        self.current_alpha = self.config.end_alpha + (alpha_range * cosine_decay)
        self.log("alpha", self.current_alpha, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        x, _, labels, _, _ = batch  # Matching your dcase25.py dataset outputs
        
        if x.dim() > 2:
            x = x.squeeze()
            
        t_logits = self.get_teacher_logits(x)
        s_logits = self.get_student_logits(x)
        
        loss = self.loss_fn(s_logits, t_logits, labels, self.current_alpha)
        acc = (s_logits.argmax(dim=-1) == labels).float().mean()
        
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, labels, _, _ = batch
        
        if x.dim() > 2:
            x = x.squeeze()
            
        s_logits = self.get_student_logits(x)
        loss = F.cross_entropy(s_logits, labels)
        acc = (s_logits.argmax(dim=-1) == labels).float().mean()
        
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/acc", acc, sync_dist=True)
        self.log("val/macro_avg_acc", acc, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }

# ==================================================================
# 4. Main Execution
# ==================================================================
def train(config):
    pl.seed_everything(42)
    wandb_logger = WandbLogger(project=config.project_name, name=config.experiment_name, config=vars(config))

    pl_module = DistillationModule(config)

    # Output Complexity constraints to ensure Mamba stays under limits
    t_macs, t_bytes = complexity.get_torch_macs_memory(pl_module.teacher_net, input_size=(1, 1, 128, 101))
    s_macs, s_bytes = complexity.get_torch_macs_memory(pl_module.student, input_size=(1, 1, config.n_mels, config.target_length))
    
    print(f"\n--- Complexity Report ---")
    print(f"Student Size: {s_bytes/1024:.2f} KB (Must be < 128.00 KB)")
    print(f"Teacher Size: {t_bytes/1024/1024:.2f} MB")
    print(f"-------------------------\n")

    wandb_logger.experiment.config.update({
        "Student_Size_Bytes": s_bytes, "Teacher_Size_Bytes": t_bytes,
        "Student_MACs": s_macs, "Teacher_MACs": t_macs
    })

    # Data Loaders using dataset/dcase25.py
    roll_samples = int(32000 * config.roll_sec)
    train_dl = DataLoader(get_training_set(split=config.subset, roll=roll_samples), batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
    val_dl = DataLoader(get_test_set(), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/macro_avg_acc", mode="max", save_top_k=1,
        dirpath=f"checkpoints/{config.experiment_name}",
        filename="best-student-epoch={epoch:02d}-val_acc={val/macro_avg_acc:.2f}",
        auto_insert_metric_name=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=config.n_epochs, accelerator="gpu", devices=config.devices,
        logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], precision=config.precision
    )

    trainer.fit(pl_module, train_dl, val_dl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Core Distillation Info
    parser.add_argument("--project_name", type=str, default="DCASE25_Distillation")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--teacher_ckpt", type=str, required=True) # E.g., checkpoints/best-effat.ckpt
    
    # Training Params
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--subset", type=int, default=25) 
    parser.add_argument("--roll_sec", type=float, default=0.1)
    
    # Audio Params
    parser.add_argument("--orig_sample_rate", type=int, default=44100)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--window_length", type=int, default=800)
    parser.add_argument("--hop_length", type=int, default=320)
    parser.add_argument("--f_min", type=int, default=0)
    parser.add_argument("--f_max", type=int, default=None)
    parser.add_argument("--target_length", type=int, default=101)
    parser.add_argument("--n_classes", type=int, default=10)

    # KD Alpha Params (Start relying heavily on teacher, finish relying on labels)
    parser.add_argument("--start_alpha", type=float, default=0.90)
    parser.add_argument("--end_alpha", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=2.0)

    # --- Student Architecture (Mamba) Params ---
    parser.add_argument("--embed_dim", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--d_state", type=int, default=32)
    parser.add_argument("--patch_size", type=str, default="16,16")

    args = parser.parse_args()
    train(args)