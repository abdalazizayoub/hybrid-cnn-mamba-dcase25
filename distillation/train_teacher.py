import argparse
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import transformers
import wandb
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dcase25 import get_training_set, get_test_set
from helpers.init import worker_init_fn
from helpers import complexity

# --- EFFICIENT-AT IMPORT ---
from models.mn.model import get_model as get_mn

class EfficientATTeacherModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config 

        # -------- THE 4.8M PARAMETER SETUP --------
        # Using the perfectly balanced mn10_as with its exact width_mult
        self.net = get_mn(
            num_classes=config.n_classes, 
            pretrained_name="mn10_as",
            width_mult=1.0  
        )
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, 
            n_fft=1024, 
            win_length=800, 
            hop_length=320, 
            n_mels=128, 
            f_min=0.0, 
            f_max=None
        )
        
        # --- SPECAUGMENT ---
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=24)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

        # -------- Metadata for Logging --------
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = [
            'airport', 'bus', 'metro', 'metro_station', 'park',
            'public_square', 'shopping_mall', 'street_pedestrian',
            'street_traffic', 'tram'
        ]
        self.device_groups = {
            'a': "real", 'b': "real", 'c': "real",
            's1': "seen", 's2': "seen", 's3': "seen",
            's4': "unseen", 's5': "unseen", 's6': "unseen"
        }

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x, mixup_lam=None, mixup_index=None, augment=False):
        if x.dim() == 4:
            x = x.squeeze(1).squeeze(1)
        elif x.dim() == 3:
            x = x.squeeze(1)
            
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.float32)
            
            # 1. Convert to Spectrogram
            x = self.mel_transform(x)
            
            # 2. Convert to Decibels (Base-10) safely!
            x = 10.0 * torch.log10(x + 1e-8)
            
            # 3. Apply SpecAugment (Only when augment=True)
            if augment:
                x = self.freq_mask(x)
                x = self.time_mask(x)
                
        # 4. Add Channel Dimension for the CNN
        x = x.unsqueeze(1)
        
        # 5. Mixup the Images 
        if mixup_lam is not None and mixup_index is not None:
            x = mixup_lam * x + (1 - mixup_lam) * x[mixup_index]
        
        # 6. Pass to EfficientAT
        logits, _ = self.net(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def training_step(self, train_batch, batch_idx):
        x, _, labels, _, _ = train_batch
        
        # --- BALANCED AUGMENTATION ---
        # 50% chance to hear confusing augmented data, 50% chance to hear CLEAN data
        mixup_prob = 0.5  
        if torch.rand(1).item() < mixup_prob:
            alpha = 0.3
            lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(x.device)
            index = torch.randperm(x.size(0)).to(x.device)
            
            y_hat = self.forward(x, mixup_lam=lam, mixup_index=index, augment=True)
            
            loss1 = F.cross_entropy(y_hat, labels)
            loss2 = F.cross_entropy(y_hat, labels[index])
            loss = lam * loss1 + (1 - lam) * loss2
        else:
            y_hat = self.forward(x, augment=False)
            loss = F.cross_entropy(y_hat, labels)
        # -----------------------------

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, _ = val_batch
        
        # augment=False prevents SpecAugment from ruining the validation score
        y_hat = self.forward(x, augment=False)

        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct_per_sample.sum(),
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }

        for d in self.device_ids:
            results[f"devloss.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devcnt.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devn_correct.{d}"] = torch.as_tensor(0., device=self.device)

        for i, d in enumerate(devices):
            results[f"devloss.{d}"] += samples_loss[i]
            results[f"devcnt.{d}"] += 1
            results[f"devn_correct.{d}"] += n_correct_per_sample[i]

        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.as_tensor(0., device=self.device)

        for i, lbl_index in enumerate(labels):
            lbl_name = self.label_ids[lbl_index]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        self.validation_step_outputs.append({k: v.cpu() for k, v in results.items()})

    def on_validation_epoch_end(self):
        self._summarize_results(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, _ = test_batch
        y_hat = self.forward(x, augment=False) 
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct_per_sample.sum(),
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }
        
        for d in self.device_ids:
            results[f"devloss.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devcnt.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devn_correct.{d}"] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(devices):
            results[f"devloss.{d}"] += samples_loss[i]
            results[f"devcnt.{d}"] += 1
            results[f"devn_correct.{d}"] += n_correct_per_sample[i]
        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.as_tensor(0., device=self.device)
        for i, lbl_index in enumerate(labels):
            lbl_name = self.label_ids[lbl_index]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        self.test_step_outputs.append({k: v.cpu() for k, v in results.items()})

    def on_test_epoch_end(self):
        self._summarize_results(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def _summarize_results(self, step_outputs, prefix):
        outputs = {k: [] for k in step_outputs[0]}
        for step_output in step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs["loss"].mean()
        acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()
        logs = {"acc": acc, "loss": avg_loss}

        for d in self.device_ids:
            dev_cnt = outputs[f"devcnt.{d}"].sum()
            if dev_cnt > 0:
                logs[f"acc.{d}"] = outputs[f"devn_correct.{d}"].sum() / dev_cnt
            
            grp = self.device_groups[d]
            logs[f"acc.{grp}"] = logs.get(f"acc.{grp}", 0.) + outputs[f"devn_correct.{d}"].sum()
            logs[f"count.{grp}"] = logs.get(f"count.{grp}", 0.) + dev_cnt

        for grp in set(self.device_groups.values()):
            if logs[f"count.{grp}"] > 0:
                logs[f"acc.{grp}"] /= logs[f"count.{grp}"]

        label_accs = []
        for lbl in self.label_ids:
            denom = outputs[f"lblcnt.{lbl}"].sum()
            if denom > 0:
                l_acc = outputs[f"lbln_correct.{lbl}"].sum() / denom
                logs[f"acc.{lbl}"] = l_acc
                label_accs.append(l_acc)

        if label_accs:
            logs["macro_avg_acc"] = torch.mean(torch.stack(label_accs))
        
        self.log_dict({f"{prefix}/{k}": v for k, v in logs.items()})

def train(config):
    wandb_logger = WandbLogger(project=config.project_name, config=config, name=config.experiment_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/macro_avg_acc", mode="max", save_top_k=1, save_last=True,
        filename='best-effat-{epoch:02d}-{val/macro_avg_acc:.2f}'
    )

    roll_samples = 32000 * config.roll_sec
    train_dl = DataLoader(
        get_training_set(split=config.subset, roll=int(roll_samples)), 
        num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True
    )
    val_dl = DataLoader(get_test_set(), num_workers=config.num_workers, batch_size=config.batch_size)

    pl_module = EfficientATTeacherModule(config)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs, logger=wandb_logger, accelerator="gpu", devices=config.devices,
        precision=config.precision, check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback], strategy="ddp_find_unused_parameters_true" if config.devices > 1 else "auto"
    )

    trainer.fit(pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    trainer.test(ckpt_path="best", dataloaders=val_dl)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EfficientAT Fine-Tuning')
    parser.add_argument("--project_name", type=str, default="DCASE25_Teacher_FineTuning")
    parser.add_argument("--experiment_name", type=str, default="EfficientAT_mn10_Balanced")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=2)
    parser.add_argument("--subset", type=int, default=25) 
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=200) 
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--weight_decay", type=float, default=1e-3) 
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--roll_sec", type=float, default=0.1)
    
    args = parser.parse_args()
    train(args)