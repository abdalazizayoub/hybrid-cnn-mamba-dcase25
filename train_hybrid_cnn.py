import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import transformers

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dataset.dcase25 import get_training_set, get_test_set
from models.hybrid_gru import get_model as get_gru_model
from models.hybrid_xlstm import get_model as get_xlstm_model
from helpers.complexity import get_torch_macs_memory

class DirectStudentModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        model_kwargs = {
            'n_classes': config.n_classes,
            'n_mels': config.n_mels,         
            'target_length': 33,   
            'embed_dim': config.embed_dim,   
            'depth': config.depth,           
            'patch_size': config.patch_size,
            'd_state': config.d_state,
            'd_conv': config.d_conv,
            'slstm_at': config.slstm_at # Pass the architecture recipe
        }
        
        if config.sequence_engine.lower() == "gru":
            self.student = get_gru_model(**model_kwargs)
        elif config.sequence_engine.lower() == "xlstm":
            self.student = get_xlstm_model(**model_kwargs)
        else:
            raise ValueError(f"Unknown engine: {config.sequence_engine}")

        # Regularization Tools
        self.freq_mask = T.FrequencyMasking(freq_mask_param=24) 
        self.time_mask = T.TimeMasking(time_mask_param=10)
        self.mixup_alpha = 0.3 
        
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park',
                          'public_square', 'shopping_mall', 'street_pedestrian',
                          'street_traffic', 'tram']
        self.validation_step_outputs = []

    def on_train_start(self):
        """Logs complexity and xLSTM structure details for the thesis report."""
        input_shape = (1, 1, self.config.n_mels, 33)
        macs, current_bytes = get_torch_macs_memory(self.student, input_shape)
        mmacs = macs / 1_000_000.0
        fp16_kb = (current_bytes / 2.0) / 1024.0
        
        # Calculate xLSTM block distribution
        num_slstm = 0
        num_mlstm = 0
        if self.config.sequence_engine.lower() == "xlstm":
            num_slstm = len([i for i in self.config.slstm_at if i < self.config.depth])
            num_mlstm = self.config.depth - num_slstm

        print("\n" + "="*60)
        print(" DCASE 2025 TASK 1 COMPLEXITY REPORT 🚀")
        print(f"Sequence Engine       : {self.config.sequence_engine.upper()}")
        if self.config.sequence_engine.lower() == "xlstm":
            print(f"xLSTM Structure       : {num_mlstm}x mLSTM | {num_slstm}x sLSTM")
        print(f"FP16 Inference Size   : {fp16_kb:.2f} KB (Limit: 128.00 KB)")
        print(f"Computational MACs    : {mmacs:.2f} Million")
        print("="*60 + "\n")
        
        if self.logger and hasattr(self.logger.experiment, 'config'):
            self.logger.experiment.config.update({
                "Model_MACs_Millions": mmacs,
                "Model_Size_FP16_KB": fp16_kb,
                "xlstm_mlstm_count": num_mlstm,
                "xlstm_slstm_count": num_slstm,
                "xlstm_recipe": str(self.config.slstm_at)
            })

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def training_step(self, batch, batch_idx):
        x, _, labels, _, _ = batch
        x = self.freq_mask(self.time_mask(x))
        
        # MixUp Augmentation
        if torch.rand(1).item() < 0.5:
            lam = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha).sample().to(x.device)
            index = torch.randperm(x.size(0)).to(x.device)
            x = lam * x + (1 - lam) * x[index]
            y_hat = self.student(x)
            loss = lam * F.cross_entropy(y_hat, labels, label_smoothing=0.1) + \
                   (1 - lam) * F.cross_entropy(y_hat, labels[index], label_smoothing=0.1)
        else:
            y_hat = self.student(x)
            loss = F.cross_entropy(y_hat, labels, label_smoothing=0.1)

        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=x.size(0), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, labels, _, _ = batch
        y_hat = self.student(x)
        loss = F.cross_entropy(y_hat, labels)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == labels).float().mean()
        
        self.validation_step_outputs.append({"loss": loss, "acc": acc})
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in self.validation_step_outputs]).mean()
        self.log("val/loss", avg_loss, sync_dist=True, prog_bar=True)
        self.log("val/macro_avg_acc", avg_acc, sync_dist=True, prog_bar=True)
        self.validation_step_outputs.clear()

def train(config):
    wandb_logger = WandbLogger(project=config.project_name, config=vars(config), name=config.experiment_name)
    ckpt_dir = os.path.join(current_dir, "checkpoints", config.experiment_name)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/macro_avg_acc", mode="max", save_top_k=1, save_last=True,
        dirpath=ckpt_dir, filename='best-epoch={epoch:02d}-val_acc={val/macro_avg_acc:.2f}'
    )
    
    train_dl = DataLoader(get_training_set(split=config.subset, roll=int(44100 * config.roll_sec)), 
                          num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(get_test_set(), num_workers=config.num_workers, batch_size=config.batch_size)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs, logger=wandb_logger, accelerator="gpu", devices=1,
        precision=config.precision, callbacks=[checkpoint_callback, LearningRateMonitor('step')],
        gradient_clip_val=1.0
    )
    trainer.fit(DirectStudentModule(config), train_dl, val_dl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="DCASE25_Final_Benchmark")
    parser.add_argument("--experiment_name", type=str, default="xLSTM_2Block_Balanced")
    parser.add_argument("--sequence_engine", type=str, default="xlstm", choices=['gru', 'xlstm'])
    
    # Architecture params
    parser.add_argument("--n_mels", type=int, default=256) 
    parser.add_argument("--embed_dim", type=int, default=32) 
    parser.add_argument("--depth", type=int, default=2) 
    parser.add_argument("--slstm_at", type=int, nargs='+', default=[1], help="Layer indices for sLSTM blocks")
    
    # Optimization
    parser.add_argument("--lr", type=float, default=0.0005) 
    parser.add_argument("--weight_decay", type=float, default=0.05) # Strong regularization
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    
    # Environment
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--subset", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--roll_sec", type=float, default=0.1)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=4) # Legacy placeholders
    parser.add_argument("--d_state", type=int, default=32)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)

    train(parser.parse_args())