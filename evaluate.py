import os
import sys
import argparse
import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dataset.dcase25 import get_test_set
from helpers.init import worker_init_fn


class EvaluationModule(pl.LightningModule):
    """LightningModule dedicated to test set evaluation and DCASE metric generation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
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

        model_type = getattr(config, 'model_type', 'mamba').lower()
        if model_type == 'gru':
            from models.hybrid_gru import get_model
            print("Evaluator: Loaded GRU Backbone.")
        else:
            from models.hybrid_net import get_model
            print("Evaluator: Loaded Mamba Backbone.")

        model_kwargs = {
            'n_classes': config.n_classes,
            'n_mels': config.n_mels,         
            'target_length': 33,   
            'embed_dim': config.embed_dim,   
            'depth': config.depth,           
            'patch_size': getattr(config, 'patch_size', 4),
            'd_state': getattr(config, 'd_state', 32),
        }
        
        base_model = get_model(**model_kwargs)

        if config.is_multi_device:
            print("Evaluator: Initializing Multi-Device Container.")
            from models.multi_device_model import MultiDeviceModelContainer
            train_device_ids = ['a', 'b', 'c', 's1', 's2', 's3']
            self.model = MultiDeviceModelContainer(base_model, train_device_ids)
        else:
            print("Evaluator: Initializing Standard Single Model.")
            self.model = base_model

        self.test_step_outputs = []

    def forward(self, x, devices=None):
        if self.config.is_multi_device:
            return self.model(x, devices)
        return self.model(x)

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, _ = test_batch

        y_hat = self.forward(x, devices)
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

        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.tensor(0., device=self.device)

        for i, dev_name in enumerate(dev_names):
            results[f"devloss.{dev_name}"] += samples_loss[i]
            results[f"devn_correct.{dev_name}"] += n_correct_per_sample[i]
            results[f"devcnt.{dev_name}"] += 1

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
            lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
            lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()
            if lbl_cnt > 0:
                logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs.get(f"acc.{l}", torch.tensor(0.)) for l in self.label_ids]))

        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        self.test_step_outputs.clear()
        
        # -----------------------------------------------------
        # STATISTICAL CONFIDENCE INTERVAL CALCULATIONS
        # -----------------------------------------------------
        def calc_ci(accuracy, n):
            """Calculates the 95% Confidence Interval Margin of Error"""
            if n <= 0: 
                return 0.0
            p = max(0.0, min(1.0, float(accuracy)))
            se = math.sqrt((p * (1.0 - p)) / n)
            return 1.96 * se

        total_n = logs.get('count.real', 0) + logs.get('count.seen', 0) + logs.get('count.unseen', 0)
        
        global_ci = calc_ci(logs['macro_avg_acc'], total_n)
        real_ci = calc_ci(logs.get('acc.real', 0), logs.get('count.real', 1))
        seen_ci = calc_ci(logs.get('acc.seen', 0), logs.get('count.seen', 1))
        unseen_ci = calc_ci(logs.get('acc.unseen', 0), logs.get('count.unseen', 1))

        print("\n" + "="*65)
        print("DCASE 2025 TASK 1 EVALUATION METRICS (95% Confidence Interval)")
        print("="*65)
        print(f"Global Macro Accuracy:       {logs['macro_avg_acc']*100:.2f}%  (± {global_ci*100:.2f}%)")
        print("-" * 65)
        print(f"Real Devices (A, B, C):      {logs.get('acc.real', 0)*100:.2f}%  (± {real_ci*100:.2f}%)")
        print(f"Seen Devices (S1, S2, S3):   {logs.get('acc.seen', 0)*100:.2f}%  (± {seen_ci*100:.2f}%)")
        print(f"Unseen Devices (S4, S5, S6): {logs.get('acc.unseen', 0)*100:.2f}%  (± {unseen_ci*100:.2f}%)")
        print("="*65 + "\n")


def evaluate(config):
    pl_module = EvaluationModule(config)
    
    print(f"Extracting weights from checkpoint: {config.ckpt_path}")
    ckpt = torch.load(config.ckpt_path, map_location="cpu")
    
    state_dict = ckpt["state_dict"]
    clean_state_dict = {}
    
    for k, v in state_dict.items():
        clean_key = k.replace("student.", "").replace("multi_device_model.", "")
        
        if config.is_multi_device:
            if "device_models" in k:
                clean_state_dict[clean_key] = v
            else:
                clean_state_dict["base_model." + clean_key] = v
        else:
            clean_state_dict[clean_key] = v

    pl_module.model.load_state_dict(clean_state_dict, strict=True)
    print("Weights loaded successfully.")

    test_ds = get_test_set(device=None)
    test_dl = DataLoader(
        dataset=test_ds, 
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers, 
        batch_size=config.batch_size
    )

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=config.devices, 
        precision=config.precision
    )
    
    trainer.test(pl_module, dataloaders=test_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE Model Evaluator')

    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to saved model checkpoint.")
    parser.add_argument("--model_type", type=str, default="mamba", choices=["mamba", "gru"], help="Backbone architecture type.")
    parser.add_argument("--is_multi_device", action="store_true", help="Flag to indicate if checkpoint is a multi-device container.")

    parser.add_argument("--n_mels", type=int, default=256) 
    parser.add_argument("--embed_dim", type=int, default=28) 
    parser.add_argument("--depth", type=int, default=2) 
    parser.add_argument("--n_classes", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")

    args = parser.parse_args()
    evaluate(args)