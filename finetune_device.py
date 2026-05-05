import copy
import argparse
import os
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import pytorch_lightning as pl
import transformers
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# --- xLSTM STABILITY FIX ---
os.environ["SLSTM_BACKEND"] = "vanilla"

from dataset.dcase25 import get_training_set, get_test_set
from helpers.init import worker_init_fn
from helpers import complexity

from models.multi_device_model import MultiDeviceModelContainer


class PLModule(pl.LightningModule):
    def __init__(self, config, base_model_state_dict=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.freq_mask = T.FrequencyMasking(freq_mask_param=24) 
        self.time_mask = T.TimeMasking(time_mask_param=10)

        self.train_device_ids = ['a', 'b', 'c', 's1', 's2', 's3']
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

        # ==========================================
        #  DYNAMIC ARCHITECTURE ROUTING
        # ==========================================
        model_type = getattr(config, 'model_type', 'xlstm').lower()
        if model_type == 'gru':
            from models.hybrid_gru import get_model as get_student_model
        elif model_type == 'xlstm':
            from models.hybrid_xlstm import get_model as get_student_model
        elif model_type == 'mamba':
            from models.hybrid_net import get_model as get_student_model
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        model_kwargs = {
            'n_classes': config.n_classes,
            'n_mels': config.n_mels,         
            'target_length': 33,   
            'embed_dim': config.embed_dim,   
            'depth': config.depth,           
            'patch_size': getattr(config, 'patch_size', 4),
            'd_state': getattr(config, 'd_state', 32),
            # NEW: Pass the xLSTM block recipe
            'slstm_at': getattr(config, 'slstm_at', [1]) 
        }
        
        base_model = get_student_model(**model_kwargs)

        if base_model_state_dict is not None:
            # Changed to strict=False to allow for smooth finetuning loading
            base_model.load_state_dict(base_model_state_dict, strict=False)

        self.multi_device_model = MultiDeviceModelContainer(
            base_model,
            self.train_device_ids
        )

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.validation_device = None

    def forward(self, x, devices):
        return self.multi_device_model(x, devices)

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
        x, _, labels, devices, _ = train_batch
        x = self.freq_mask(self.time_mask(x))
        y_hat = self.multi_device_model(x, devices)
        loss = F.cross_entropy(y_hat, labels)
        self.log(f"train/loss.{devices[0]}", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, _ = val_batch
        y_hat = self.forward(x, devices)
        samples_loss = F.cross_entropy(y_hat, labels)
        _, preds = torch.max(y_hat, dim=1)
        results = {
            "n_correct": (preds == labels).sum(),
            "n_pred": torch.tensor(len(labels), device=self.device),
            "devloss": samples_loss.sum(),
            "devcnt": torch.tensor(len(devices), device=self.device)
        }
        self.validation_step_outputs.append({k: v.cpu() for k, v in results.items()})
        self.validation_device = devices[0]
        return samples_loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        outputs = {k: torch.stack([x[k] for x in self.validation_step_outputs]) for k in self.validation_step_outputs[0]}
        dev_loss = outputs["devloss"].sum() / outputs["devcnt"].sum()
        dev_acc = outputs["n_correct"].sum().float() / outputs["n_pred"].sum()
        self.log(f"val/acc.{self.validation_device}", dev_acc, prog_bar=True)
        self.validation_step_outputs.clear()

    # (test_step and on_test_epoch_end remain largely identical to your source)
    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, _ = test_batch
        self.multi_device_model.half()
        x = x.half()
        y_hat = self.multi_device_model(x, devices)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        _, preds = torch.max(y_hat, dim=1)
        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {"loss": samples_loss.mean(), "n_correct": (preds == labels).sum(), "n_pred": torch.tensor(len(labels))}
        for dev_id in self.device_ids:
            results[f"devcnt.{dev_id}"] = torch.tensor(0.)
            results[f"devn_correct.{dev_id}"] = torch.tensor(0.)
        for i, dev_name in enumerate(dev_names):
            results[f"devn_correct.{dev_name}"] += (preds[i] == labels[i])
            results[f"devcnt.{dev_name}"] += 1
        self.test_step_outputs.append({k: v.cpu() for k, v in results.items()})

    def on_test_epoch_end(self):
        outputs = {k: torch.stack([x[k] for x in self.test_step_outputs]) for k in self.test_step_outputs[0]}
        acc = outputs["n_correct"].sum().float() / outputs["n_pred"].sum()
        logs = {"acc": acc}
        for dev_id in self.device_ids:
            cnt = outputs[f"devcnt.{dev_id}"].sum()
            if cnt > 0:
                logs[f"acc.{dev_id}"] = outputs[f"devn_correct.{dev_id}"].sum() / cnt
        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        self.test_step_outputs.clear()


def train(config):
    base_model_state_dict = None
    if config.ckpt_path is not None:
        print(f"\n Extracting Weights from Checkpoint: {config.ckpt_path}")
        ckpt = torch.load(config.ckpt_path, map_location="cpu")
        base_model_state_dict = {
            k.replace("student.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("student.")
        }

    pl_module = PLModule(config, base_model_state_dict=base_model_state_dict)
    wandb_logger = WandbLogger(project=config.project_name, config=config, name=config.experiment_name)

    # Calculate xLSTM block distribution for logging
    if config.model_type == "xlstm":
        num_slstm = len([i for i in config.slstm_at if i < config.depth])
        num_mlstm = config.depth - num_slstm
        wandb_logger.experiment.config.update({"xlstm_mlstm": num_mlstm, "xlstm_slstm": num_slstm})

    for device_id in pl_module.train_device_ids:
        print(f"\n🎧 Adaptation: Training Specialist for Device {device_id.upper()}")
        train_ds = get_training_set(config.subset, device=device_id, roll=int(44100 * config.roll_sec))
        train_dl = DataLoader(train_ds, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True)
        test_dl = DataLoader(get_test_set(device=device_id), num_workers=config.num_workers, batch_size=config.batch_size)

        trainer = pl.Trainer(
            max_epochs=config.n_epochs, logger=wandb_logger, accelerator="gpu", devices=1,
            precision=config.precision, callbacks=[ModelCheckpoint(save_last=True)]
        )
        trainer.fit(pl_module, train_dl, test_dl)

    print("\n🏁 Launching Global Test Suite...")
    test_dl = DataLoader(get_test_set(device=None), num_workers=config.num_workers, batch_size=config.batch_size)
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=config.precision)
    trainer.test(pl_module, dataloaders=test_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="xlstm", choices=["mamba", "gru", "xlstm"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--project_name", type=str, default="DCASE25_Finetuning")
    parser.add_argument("--experiment_name", type=str, default="xLSTM_Specialist_Adaptation")
    
    # Architecture 
    parser.add_argument("--n_mels", type=int, default=256) 
    parser.add_argument("--embed_dim", type=int, default=32) 
    parser.add_argument("--depth", type=int, default=2) 
    parser.add_argument("--slstm_at", type=int, nargs='+', default=[1])
    
    # Finetuning Hyperparams 
    parser.add_argument("--n_epochs", type=int, default=15) 
    parser.add_argument("--lr", type=float, default=0.00005) 
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--roll_sec", type=float, default=0.1)
    parser.add_argument("--subset", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)

    train(parser.parse_args())