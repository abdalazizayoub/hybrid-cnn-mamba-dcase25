import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import transformers

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models.hybrid_net import get_model as get_student_model

# ==========================================
# 1. ESC-50 DATALOADER (Mimics DCASE Output)
# ==========================================
class ESC50PretrainDataset(Dataset):
    def __init__(self, data_dir="data/esc50", roll_sec=0.1):
        self.data_dir = data_dir
        self.df = pd.read_csv(os.path.join(data_dir, "meta.csv"))
        self.roll_samples = int(44100 * roll_sec)
        
        # Matches the expected DCASE Hybrid model shapes
        self.mel_transform = T.MelSpectrogram(
            sample_rate=44100, n_fft=1024, win_length=800, hop_length=320, n_mels=256
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        label = row['target']
        
        audio_path = os.path.join(self.data_dir, filename)
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample just in case
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
            
        # Random audio slice (Data Augmentation)
        if waveform.shape[1] > self.roll_samples:
            start = torch.randint(0, waveform.shape[1] - self.roll_samples, (1,)).item()
            waveform = waveform[:, start:start + self.roll_samples]
            
        # Convert to Mel Spectrogram
        mel = self.mel_transform(waveform)
        mel = 10.0 * torch.log10(mel + 1e-8)
        
        # Force strict shape matching [1, 256, 33] for the Mamba layers
        target_frames = 33
        if mel.shape[-1] < target_frames:
            mel = F.pad(mel, (0, target_frames - mel.shape[-1]))
        elif mel.shape[-1] > target_frames:
            mel = mel[:, :, :target_frames]
            
        mel = mel.unsqueeze(0) # Add channel dim
        
        # Return Fake DCASE Tuple
        return mel, filename, label, "a", "none"


# ==========================================
# 2. LIGHTNING MODULE
# ==========================================
class DirectStudentModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.student = get_student_model(
            n_classes=config.n_classes, n_mels=256, target_length=33,   
            embed_dim=config.embed_dim, depth=config.depth, patch_size=config.patch_size
        )

        self.freq_mask = T.FrequencyMasking(freq_mask_param=48)
        self.time_mask = T.TimeMasking(time_mask_param=15) 
        self.mixup_alpha = 0.3 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def training_step(self, batch, batch_idx):
        x, _, labels, _, _ = batch
        x = self.freq_mask(x)
        x = self.time_mask(x)
        
        if torch.rand(1).item() < 0.5:
            lam = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha).sample().to(x.device)
            index = torch.randperm(x.size(0)).to(x.device)
            x = lam * x + (1 - lam) * x[index]
            y_hat = self.student(x)
            loss1 = F.cross_entropy(y_hat, labels)
            loss2 = F.cross_entropy(y_hat, labels[index])
            loss = lam * loss1 + (1 - lam) * loss2
        else:
            y_hat = self.student(x)
            loss = F.cross_entropy(y_hat, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, labels, _, _ = batch
        y_hat = self.student(x)
        acc = (y_hat.argmax(dim=-1) == labels).float().mean()
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train(config):
    wandb_logger = WandbLogger(project=config.project_name, config=vars(config), name=config.experiment_name)
    ckpt_dir = os.path.join(current_dir, "checkpoints", config.experiment_name)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/acc", mode="max", save_top_k=1, save_last=True,
        dirpath=ckpt_dir, filename='best-pretrain-epoch={epoch:02d}-val_acc={val/acc:.2f}'
    )

    dataset = ESC50PretrainDataset(roll_sec=config.roll_sec)
    
    # 90/10 Split for quick pre-training evaluation
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_dl = DataLoader(train_set, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_set, num_workers=config.num_workers, batch_size=config.batch_size)

    pl_module = DirectStudentModule(config)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs, logger=wandb_logger, accelerator="gpu", devices=config.devices,
        precision=config.precision, callbacks=[checkpoint_callback]
    )

    trainer.fit(pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="DCASE25_Hybrid_PreTrain")
    parser.add_argument("--experiment_name", type=str, default="PreTrain_ESC50")
    
    # Same Architecture!
    parser.add_argument("--embed_dim",  type=int, default=28) 
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=4)
    
    # ESC-50 has exactly 50 classes
    parser.add_argument("--n_classes", type=int, default=50)
    
    parser.add_argument("--lr", type=float, default=0.001)              
    parser.add_argument("--warmup_steps", type=int, default=1000)       
    parser.add_argument("--n_epochs", type=int, default=100) 
    parser.add_argument("--weight_decay", type=float, default=1e-3) 
    parser.add_argument("--batch_size", type=int, default=128)          
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--roll_sec", type=float, default=0.1)          
    
    args = parser.parse_args()
    train(args)