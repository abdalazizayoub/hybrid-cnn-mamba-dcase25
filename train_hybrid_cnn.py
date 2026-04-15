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
        
        # Logging configurations
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park',
                          'public_square', 'shopping_mall', 'street_pedestrian',
                          'street_traffic', 'tram']
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}
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
        x, _, labels, devices, _ = batch
        
        y_hat = self.student(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct_per_sample.sum(),
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }

        # Track metrics per device
        for i, d in enumerate(devices):
            results[f"devloss.{d}"] = results.get(f"devloss.{d}", torch.as_tensor(0., device=self.device)) + samples_loss[i]
            results[f"devcnt.{d}"] = results.get(f"devcnt.{d}", torch.as_tensor(0., device=self.device)) + 1
            results[f"devn_correct.{d}"] = results.get(f"devn_correct.{d}", torch.as_tensor(0., device=self.device)) + n_correct_per_sample[i]

        # Track metrics per class
        for i, lbl_index in enumerate(labels):
            lbl_name = self.label_ids[lbl_index]
            results[f"lblloss.{lbl_name}"] = results.get(f"lblloss.{lbl_name}", torch.as_tensor(0., device=self.device)) + samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] = results.get(f"lbln_correct.{lbl_name}", torch.as_tensor(0., device=self.device)) + n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] = results.get(f"lblcnt.{lbl_name}", torch.as_tensor(0., device=self.device)) + 1

        self.validation_step_outputs.append({k: v.cpu() for k, v in results.items()})
        return samples_loss.mean()

    def on_validation_epoch_end(self):
        outputs = {}
        for step_output in self.validation_step_outputs:
            for k, v in step_output.items():
                if k not in outputs:
                    outputs[k] = []
                outputs[k].append(v)
                
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs["loss"].mean()
        acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()
        logs = {"acc": acc, "loss": avg_loss}

        # Compute Device Accuracies
        for d in self.device_ids:
            dev_cnt = outputs.get(f"devcnt.{d}", torch.as_tensor([0.])).sum()
            if dev_cnt > 0:
                logs[f"acc.{d}"] = outputs.get(f"devn_correct.{d}", torch.as_tensor([0.])).sum() / dev_cnt
            
            grp = self.device_groups[d]
            logs[f"acc.{grp}"] = logs.get(f"acc.{grp}", 0.) + outputs.get(f"devn_correct.{d}", torch.as_tensor([0.])).sum()
            logs[f"count.{grp}"] = logs.get(f"count.{grp}", 0.) + dev_cnt

        # Compute Device Group Accuracies (real, seen, unseen)
        for grp in set(self.device_groups.values()):
            if logs.get(f"count.{grp}", 0) > 0:
                logs[f"acc.{grp}"] /= logs[f"count.{grp}"]

        # Compute Class Accuracies & Macro Average
        label_accs = []
        for lbl in self.label_ids:
            denom = outputs.get(f"lblcnt.{lbl}", torch.as_tensor([0.])).sum()
            if denom > 0:
                l_acc = outputs.get(f"lbln_correct.{lbl}", torch.as_tensor([0.])).sum() / denom
                logs[f"acc.{lbl}"] = l_acc
                label_accs.append(l_acc)

        if label_accs:
            logs["macro_avg_acc"] = torch.mean(torch.stack(label_accs))
        else:
            logs["macro_avg_acc"] = acc

        macro_acc = logs.pop("macro_avg_acc", 0.0) 
        logs.pop("loss", None)
        
        self.log_dict({f"val/{k}": v for k, v in logs.items()}, sync_dist=True)
        self.log("val/loss", avg_loss, sync_dist=True, prog_bar=True)
        self.log("val/macro_avg_acc", macro_acc, sync_dist=True, prog_bar=True)
        
        self.validation_step_outputs.clear()

def train(config):
    wandb_logger = WandbLogger(project=config.project_name, config=vars(config), name=config.experiment_name)
    ckpt_dir = os.path.join(current_dir, "checkpoints", config.experiment_name)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/macro_avg_acc", mode="max", save_top_k=1, save_last=True,
        dirpath=ckpt_dir, filename='best-epoch={epoch:02d}-val_acc={val/macro_avg_acc:.2f}',
        auto_insert_metric_name=False
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
    # UPDATED: Corrected WandB project name to match the rest of your experiment space
    parser.add_argument("--project_name", type=str, default="DCASE25_Hybrid_Architecture")
    parser.add_argument("--experiment_name", type=str, default="xLSTM_2Block_Balanced")
    parser.add_argument("--sequence_engine", type=str, default="xlstm", choices=['gru', 'xlstm'])
    
    # Architecture params
    parser.add_argument("--n_mels", type=int, default=256) 
    parser.add_argument("--embed_dim", type=int, default=32) 
    parser.add_argument("--depth", type=int, default=2) 
    parser.add_argument("--slstm_at", type=int, nargs='+', default=[1], help="Layer indices for sLSTM blocks")
    
    # Optimization
    parser.add_argument("--lr", type=float, default=0.0005) 
    parser.add_argument("--weight_decay", type=float, default=0.05) 
    parser.add_argument("--n_epochs", type=int, default=150)