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
from models.hybrid_net import get_model as get_student_model
from helpers.complexity import get_torch_macs_memory , get_model_size_bytes


class DirectStudentModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # 1. The Student Model (Hybrid CNN-Mamba)
        self.student = get_student_model(
            n_classes=config.n_classes,
            n_mels=256,         
            target_length=33,   
            embed_dim=config.embed_dim,
            depth=config.depth,
            patch_size=config.patch_size
        )

        # 2. SNTL-NTU Augmentation: SpecAugment (Frequency Masking)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=48)
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park',
                          'public_square', 'shopping_mall', 'street_pedestrian',
                          'street_traffic', 'tram']
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}
        self.validation_step_outputs = []

    def on_train_start(self):
        """Calculates and logs the exact architecture byte size and MACs before training begins."""
        input_shape = (1, 1, 256, 33)
        macs, current_bytes = get_torch_macs_memory(self.student, input_shape)
        mmacs = macs / 1_000_000.0
        current_kb = current_bytes / 1024.0
        is_fp32 = next(self.student.parameters()).dtype == torch.float32
        fp16_bytes = current_bytes / 2.0 if is_fp32 else current_bytes
        fp16_kb = fp16_bytes / 1024.0
        max_bytes = 128000
        max_kb = max_bytes / 1024.0
        print("\n" + "="*60)
        print(" DCASE 2025 TASK 1 COMPLEXITY REPORT 🚀")
        print(f"Current FP32 Size     : {current_kb:.2f} KB")
        print(f"FP16 Inference Size   : {fp16_kb:.2f} KB (Limit: {max_kb:.2f} KB)")
        
        compliant = True
        if fp16_bytes > max_bytes:
            print(" WARNING: Model SIZE is OVER the 128,000 bytes limit!")
            compliant = False
            
        if compliant:
            print("STATUS: Fully compliant with all DCASE constraints.")
        print("="*60 + "\n")
        if self.logger and hasattr(self.logger.experiment, 'config'):
            self.logger.experiment.config.update({
                "Model_MACs_Millions": mmacs,
                "Model_Size_FP16_KB": fp16_kb,
                "DCASE_Compliant": compliant
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
        x_aug = self.freq_mask(x)
        y_hat = self.student(x_aug)
        loss = F.cross_entropy(y_hat, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=x.size(0))
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=x.size(0))
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, files, labels, devices, _ = batch
        y_hat = self.student(x)

        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct_per_sample.sum(),
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }

        for i, d in enumerate(devices):
            results[f"devloss.{d}"] = results.get(f"devloss.{d}", torch.as_tensor(0., device=self.device)) + samples_loss[i]
            results[f"devcnt.{d}"] = results.get(f"devcnt.{d}", torch.as_tensor(0., device=self.device)) + 1
            results[f"devn_correct.{d}"] = results.get(f"devn_correct.{d}", torch.as_tensor(0., device=self.device)) + n_correct_per_sample[i]

        self.validation_step_outputs.append({k: v.cpu() for k, v in results.items()})
        return samples_loss.mean()

    def on_validation_epoch_end(self):
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs["loss"].mean()
        acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()
        logs = {"acc": acc, "loss": avg_loss}

        for d in self.device_ids:
            dev_cnt = outputs.get(f"devcnt.{d}", torch.as_tensor(0.)).sum()
            if dev_cnt > 0:
                logs[f"acc.{d}"] = outputs.get(f"devn_correct.{d}", torch.as_tensor(0.)).sum() / dev_cnt
            
            grp = self.device_groups[d]
            logs[f"acc.{grp}"] = logs.get(f"acc.{grp}", 0.) + outputs.get(f"devn_correct.{d}", torch.as_tensor(0.)).sum()
            logs[f"count.{grp}"] = logs.get(f"count.{grp}", 0.) + dev_cnt

        for grp in set(self.device_groups.values()):
            if logs.get(f"count.{grp}", 0) > 0:
                logs[f"acc.{grp}"] /= logs[f"count.{grp}"]

        macro_acc = outputs["n_correct"].sum() / outputs["n_pred"].sum() 
        
        logs.pop("loss", None)
        
        self.log_dict({f"val/{k}": v for k, v in logs.items()}, sync_dist=True)
        self.log("val/loss", avg_loss, sync_dist=True, prog_bar=True)
        self.log("val/macro_avg_acc", macro_acc, sync_dist=True, prog_bar=True)
        
        self.validation_step_outputs.clear()


def train(config):
    wandb_logger = WandbLogger(project=config.project_name, config=vars(config), name=config.experiment_name)
    ckpt_dir = os.path.join(current_dir, "checkpoints", config.experiment_name)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/macro_avg_acc", mode="max", save_top_k=1, save_last=True,
        dirpath=ckpt_dir,
        filename='best-student-epoch={epoch:02d}-val_acc={val/macro_avg_acc:.2f}',
        auto_insert_metric_name=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    roll_samples = int(44100 * config.roll_sec)
    train_dl = DataLoader(
        get_training_set(split=config.subset, roll=roll_samples), 
        num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    val_dl = DataLoader(get_test_set(), num_workers=config.num_workers, batch_size=config.batch_size)

    pl_module = DirectStudentModule(config)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs, logger=wandb_logger, accelerator="gpu", devices=config.devices,
        precision=config.precision, check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0  
    )

    trainer.fit(pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Direct Student Training (SNTL-NTU style)')
    parser.add_argument("--project_name", type=str, default="DCASE25_Hybrid_Direct")
    parser.add_argument("--experiment_name", type=str, default="Hybrid_SNTL_Baseline")
    
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=4)
    
    parser.add_argument("--lr", type=float, default=0.001)              
    parser.add_argument("--warmup_steps", type=int, default=1000)       
    parser.add_argument("--n_epochs", type=int, default=150)            
    parser.add_argument("--batch_size", type=int, default=256)          
    
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--subset", type=int, default=25)              
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4) 
    parser.add_argument("--roll_sec", type=float, default=0.1)          
    
    args = parser.parse_args()
    train(args)