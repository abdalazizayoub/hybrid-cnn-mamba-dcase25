import argparse
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import transformers
import wandb
import os
import sys

# Local imports
from dataset.dcase25 import get_training_set, get_test_set
from helpers.init import worker_init_fn
from helpers.utils import mixstyle
from helpers import complexity
from models.net import get_model

class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config 

        # -------- Preprocessing Pipeline --------
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.Resample(
                orig_freq=config.orig_sample_rate,
                new_freq=config.sample_rate
            ),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.window_length,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                f_min=config.f_min,
                f_max=config.f_max
            )
        )
        self.mel_augment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True),
            torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)
        )

        # -------- Audio Mamba Model --------
        self.model = get_model(
            n_classes=config.n_classes,
            n_mels=config.n_mels,
            target_length=config.target_length,
            embed_dim=config.embed_dim,
            depth=config.depth,
            patch_size=config.patch_size,
            # Pass any other needed params here
        )  

        # -------- Metadata Definitions --------
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

    def mel_forward(self, x):
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        # Numerical stability epsilon
        x = (x + 1e-5).log()
        return x

    def forward(self, x):
        x = self.mel_forward(x)
        # Ensure 4D shape: (Batch, Channels, Freq, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)

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
        x = self.mel_forward(x)

        if self.config.mixstyle_p > 0:
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)
        
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, labels)

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # Validation uses the forward() method to ensure consistent resizing/unsqueeze
        x, files, labels, devices, _ = val_batch
        y_hat = self.forward(x)

        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct_per_sample.sum(),
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }

        # Per-device/label accumulation logic
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
        # REMOVED: Manual .half() calls. Trainer handles precision.
        x, files, labels, devices, _ = test_batch
        y_hat = self.forward(x) 

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
        # Common helper to aggregate device/label metrics
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
            logs[f"acc.{d}"] = outputs[f"devn_correct.{d}"].sum() / dev_cnt
            
            grp = self.device_groups[d]
            logs[f"acc.{grp}"] = logs.get(f"acc.{grp}", 0.) + outputs[f"devn_correct.{d}"].sum()
            logs[f"count.{grp}"] = logs.get(f"count.{grp}", 0.) + dev_cnt

        for grp in set(self.device_groups.values()):
            logs[f"acc.{grp}"] /= logs[f"count.{grp}"]

        label_accs = []
        for lbl in self.label_ids:
            l_acc = outputs[f"lbln_correct.{lbl}"].sum() / outputs[f"lblcnt.{lbl}"].sum()
            logs[f"acc.{lbl}"] = l_acc
            label_accs.append(l_acc)

        logs["macro_avg_acc"] = torch.mean(torch.stack(label_accs))
        self.log_dict({f"{prefix}/{k}": v for k, v in logs.items()})

def train(config):
    wandb_logger = WandbLogger(
        project=config.project_name,
        config=config,
        name=config.experiment_name
    )

    # Setup Checkpointing to track Best model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/macro_avg_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename='best-aum-{epoch:02d}-{val/macro_avg_acc:.2f}'
    )

    roll_samples = config.orig_sample_rate * config.roll_sec
    train_dl = DataLoader(get_training_set(config.subset, roll=roll_samples), 
                          num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True)
    test_dl = DataLoader(get_test_set(), num_workers=config.num_workers, batch_size=config.batch_size)

    pl_module = PLModule(config)

    # Complexity estimation
    sample = next(iter(test_dl))[0][0].unsqueeze(0)
    shape = pl_module.mel_forward(sample).size()
    input_size = (1, 1, shape[-2], shape[-1])
    macs, params_bytes = complexity.get_torch_macs_memory(pl_module.model, input_size=input_size)
    wandb_logger.experiment.config["MACs"] = macs
    wandb_logger.experiment.config["Parameters_Bytes"] = params_bytes

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=config.precision, # Set to "16-mixed" if you need speed
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(pl_module, train_dl, test_dl)
    
    # Test using the Best weights found during training
    trainer.test(ckpt_path="best", dataloaders=test_dl)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 25 Audio Mamba Training')
    parser.add_argument("--project_name", type=str, default="DCASE25_Task1")
    parser.add_argument("--experiment_name", type=str, default="AUM_Small_Scratch")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
    parser.add_argument("--orig_sample_rate", type=int, default=44100)
    parser.add_argument("--subset", type=int, default=25)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--patch_size", type=str, default="16,16")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mixstyle_p", type=float, default=0.6)
    parser.add_argument("--mixstyle_alpha", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--roll_sec", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--window_length", type=int, default=3072)
    parser.add_argument("--hop_length", type=int, default=500)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--n_mels", type=int, default=256)
    parser.add_argument("--target_length", type=int, default=1024)
    parser.add_argument("--freqm", type=int, default=48)
    parser.add_argument("--timem", type=int, default=64)
    parser.add_argument("--f_min", type=int, default=0)
    parser.add_argument("--f_max", type=int, default=None)
    
    args = parser.parse_args()
    train(args)
    
## Removed Weight Corruption: By removing self.model.half(), we prevent the model's weights from becoming "stale" or numerically unstable during the test phase.

#Best vs. Last: Testing on the best checkpoint ensures you aren't evaluating a model that started overfitting in the final 20 epochs.

#Preprocessing Consistency: Both validation_step and test_step now use the exact same logic (through self.forward()), ensuring that resizing and log-mel normalization are identical across all evaluation phases.