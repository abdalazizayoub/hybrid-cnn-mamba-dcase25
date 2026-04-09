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

from dataset.dcase25 import get_training_set, get_test_set
from helpers.init import worker_init_fn
from helpers import complexity

from models.hybrid_net import get_model as get_student_model
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

        model_kwargs = {
            'n_classes': config.n_classes,
            'n_mels': config.n_mels,         
            'target_length': 33,   
            'embed_dim': config.embed_dim,   
            'depth': config.depth,           
            'patch_size': config.patch_size,
            'd_state': config.d_state,
        }
        base_model = get_student_model(**model_kwargs)

        if base_model_state_dict is not None:
            base_model.load_state_dict(base_model_state_dict, strict=True)

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
        
        x = self.freq_mask(x)
        x = self.time_mask(x)

        y_hat = self.multi_device_model(x, devices)
        loss = F.cross_entropy(y_hat, labels)

        self.log(f"lr.{devices[0]}", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log(f"train/loss.{devices[0]}", loss.detach().cpu())
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, _ = val_batch
        
        y_hat = self.forward(x, devices)
        samples_loss = F.cross_entropy(y_hat, labels)

        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        results = {
            "n_correct": n_correct,
            "n_pred": torch.tensor(len(labels), device=self.device),
            "devloss": samples_loss.sum(),
            "devn_correct": n_correct,
            "devcnt": torch.tensor(len(devices), device=self.device)
        }

        self.validation_step_outputs.append({k: v.cpu() for k, v in results.items()})
        self.validation_device = devices[0]

    def on_validation_epoch_end(self):
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        logs = {}
        dev_loss = outputs["devloss"].sum()
        dev_cnt = outputs["devcnt"].sum()
        dev_correct = outputs["devn_correct"].sum()
        device_name = self.validation_device
        
        logs[f"loss.{device_name}"] = dev_loss / dev_cnt
        logs[f"acc.{device_name}"] = dev_correct / dev_cnt
        logs[f"cnt.{device_name}"] = dev_cnt.float()

        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        self.validation_step_outputs.clear()
        self.validation_device = None

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, _ = test_batch

        self.multi_device_model.half()
        x = x.half()

        y_hat = self.multi_device_model(x, devices)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct,
            "n_pred": torch.tensor(len(labels), device=self.device)
        }

        for dev_id in self.device_ids:
            results[f"devloss.{dev_id}"] = torch.tensor(0., device=self.device)
            results[f"devcnt.{dev_id}"] = torch.tensor(0., device=self.device)
            results[f"devn_correct.{dev_id}"] = torch.tensor(0., device=self.device)

        for i, dev_name in enumerate(dev_names):
            results[f"devloss.{dev_name}"] += samples_loss[i]
            results[f"devn_correct.{dev_name}"] += n_correct_per_sample[i]
            results[f"devcnt.{dev_name}"] += 1

        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.tensor(0., device=self.device)

        for i, lbl_idx in enumerate(labels):
            lbl_name = self.label_ids[lbl_idx]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        self.test_step_outputs.append({k: v.cpu() for k, v in results.items()})

    def on_test_epoch_end(self):
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs["loss"].mean()
        acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()
        logs = {"acc": acc, "loss": avg_loss}

        for dev_id in self.device_ids:
            dev_loss = outputs[f"devloss.{dev_id}"].sum()
            dev_cnt = outputs[f"devcnt.{dev_id}"].sum()
            dev_correct = outputs[f"devn_correct.{dev_id}"].sum()
            
            if dev_cnt > 0:
                logs[f"loss.{dev_id}"] = dev_loss / dev_cnt
                logs[f"acc.{dev_id}"] = dev_correct / dev_cnt
            logs[f"cnt.{dev_id}"] = dev_cnt

            grp = self.device_groups[dev_id]
            logs[f"acc.{grp}"] = logs.get(f"acc.{grp}", 0.) + dev_correct
            logs[f"count.{grp}"] = logs.get(f"count.{grp}", 0.) + dev_cnt
            logs[f"lloss.{grp}"] = logs.get(f"lloss.{grp}", 0.) + dev_loss

        for grp in set(self.device_groups.values()):
            if logs.get(f"count.{grp}", 0) > 0:
                logs[f"acc.{grp}"] /= logs[f"count.{grp}"]
                logs[f"lloss.{grp}"] /= logs[f"count.{grp}"]

        for lbl in self.label_ids:
            lbl_loss = outputs[f"lblloss.{lbl}"].sum()
            lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
            lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()
            if lbl_cnt > 0:
                logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt
                logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt
            logs[f"cnt.{lbl}"] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs.get(f"acc.{l}", torch.tensor(0.)) for l in self.label_ids]))

        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        self.test_step_outputs.clear()


def train(config):
    base_model_state_dict = None
    if config.ckpt_path is not None:
        ckpt = torch.load(config.ckpt_path, map_location="cpu")
        base_model_state_dict = {
            k.replace("student.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("student.")
        }

    pl_module = PLModule(config, base_model_state_dict=base_model_state_dict)

    roll_samples = int(44100 * config.roll_sec)
    
    wandb_logger = WandbLogger(
        project=config.project_name,
        config=config,
        name=config.experiment_name
    )

    for device_id in pl_module.train_device_ids:
        train_ds = get_training_set(config.subset, device=device_id, roll=roll_samples)
        train_dl = DataLoader(
            dataset=train_ds, worker_init_fn=worker_init_fn,
            num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True, drop_last=True
        )
        
        test_ds = get_test_set(device=device_id)
        test_dl = DataLoader(
            dataset=test_ds, worker_init_fn=worker_init_fn,
            num_workers=config.num_workers, batch_size=config.batch_size
        )

        input_shape = (1, 1, config.n_mels, 33) 
        model = pl_module.multi_device_model.get_model_for_device(device_id)
        
        macs, current_bytes = complexity.get_torch_macs_memory(model, input_size=input_shape)
        fp16_bytes = current_bytes / 2.0
        
        wandb_logger.experiment.config.update({
            f"MACs_{device_id}_model": macs,
            f"Parameters_Bytes_{device_id}_model": fp16_bytes
        }, allow_val_change=True)

        trainer = pl.Trainer(
            max_epochs=config.n_epochs,
            logger=wandb_logger,
            accelerator="gpu",
            devices=config.devices,
            precision=config.precision,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            callbacks=[ModelCheckpoint(save_last=True)]
        )

        trainer.fit(pl_module, train_dl, test_dl)

    test_dl = DataLoader(
        dataset=get_test_set(device=None),
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs, logger=wandb_logger,
        accelerator="gpu", devices=config.devices, precision=config.precision
    )
    trainer.test(pl_module, dataloaders=test_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/abdalaziz-ayoub/Thesis_Hybrid_CNN_Mamba/checkpoints/Hybrid_256mels_32state/best-student-epoch=84-val_acc=0.48.ckpt",
    )
    parser.add_argument("--project_name", type=str, default="DCASE25_Task1")
    parser.add_argument("--experiment_name", type=str, default="Hybrid_Device_Finetune")
    
    parser.add_argument("--n_mels", type=int, default=256) 
    parser.add_argument("--embed_dim", type=int, default=28) 
    parser.add_argument("--depth", type=int, default=2) 
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--d_state", type=int, default=32)
    parser.add_argument("--n_classes", type=int, default=10)

    parser.add_argument("--n_epochs", type=int, default=15) 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.00005) 
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    
    parser.add_argument("--roll_sec", type=float, default=0.1)
    parser.add_argument("--subset", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)

    args = parser.parse_args()
    train(args)