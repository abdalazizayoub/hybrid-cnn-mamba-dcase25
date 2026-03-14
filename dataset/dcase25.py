import os
import pandas as pd
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from torch.hub import download_url_to_file

from sklearn.preprocessing import LabelEncoder
from typing import Optional, List


dataset_dir = "/home/abdalaziz-ayoub/datasets/PR_DATA"
assert dataset_dir, "Specify 'TAU Urban Acoustic Scenes 2022 Mobile' dataset location in 'dataset_dir'."

# Dataset configuration
dataset_config = {
    "dataset_name": "tau25",
    "meta_csv": os.path.join(dataset_dir, "meta_clean.csv"),
    "split_path": "split_setup",
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
    "eval_dir": None,  
    "eval_fold_csv": None
}


class DCASE25Dataset(Dataset):
    """
    DCASE'25 Dataset: Generates SNTL-NTU Ultra-High Resolution Spectrograms natively.
    """
    def __init__(self, meta_csv: str, roll_samples: int = 0, n_mels: int = 128):
        df = pd.read_csv(meta_csv, sep="\t")
        df["filename"] = df["filename"].str.replace("/", os.sep)        

        self.files = df["filename"].values
        self.devices = df["source_label"].values
        self.cities = LabelEncoder().fit_transform(df["identifier"].apply(lambda loc: loc.split("-")[0]))
        
        # Hardcoding the 10 labels to ensure they map identically every single time
        self.classes = ['airport', 'bus', 'metro', 'metro_station', 'park',
                        'public_square', 'shopping_mall', 'street_pedestrian',
                        'street_traffic', 'tram']
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = torch.tensor([class_to_idx[lbl] for lbl in df["scene_label"]], dtype=torch.long)

        self.roll_samples = roll_samples

        # ==========================================
        # SNTL-NTU ULTRA-HIGH RESOLUTION SETTINGS
        # ==========================================
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,      
            n_fft=8192,             
            win_length=8192,        
            hop_length=1364,        
            n_mels=n_mels       # <-- DYNAMIC! Defaults to 128 for the Deep Mamba
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __getitem__(self, index: int):
        audio_path = os.path.join(dataset_dir, self.files[index])
        waveform, sr = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
            
        # Time-shifting directly on the raw waveform (before spectrogram conversion)
        if self.roll_samples > 0:
            shift = np.random.randint(-self.roll_samples, self.roll_samples + 1)
            waveform = waveform.roll(shift, dims=1)

        # Output shape will be [1, n_mels, 33]
        x = self.mel_transform(waveform)
        x = self.amp_to_db(x)
        
        return x, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self) -> int:
        return len(self.files)


class SubsetDataset(Dataset):
    """A dataset that selects a subset of samples based on given indices."""
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


# --- Dataset Loading Functions --- #

def download_split_file(split_name: str):
    """Downloads official DCASE dataset split files if not available."""
    os.makedirs(dataset_config["split_path"], exist_ok=True)
    split_file = os.path.join(dataset_config["split_path"], split_name)
    if not os.path.isfile(split_file):
        print(f"Downloading {split_name}...")
        download_url_to_file(dataset_config["split_url"] + split_name, split_file)
    return split_file


def get_dataset_split(meta_csv: str, split_csv: str, device: Optional[str] = None, roll: int = 0, n_mels: int = 128) -> Dataset:
    """Filters the dataset safely using the official CP-JKU split logic."""
    meta = pd.read_csv(meta_csv, sep="\t")
    split_files = pd.read_csv(split_csv, sep="\t")["filename"].values
    
    # This guarantees NO data leakage!
    subset_indices = meta[meta["filename"].isin(split_files)].index.tolist()
    
    if device:
        subset_indices = meta.loc[subset_indices, :].query("source_label == @device").index.tolist()
        
    return SubsetDataset(DCASE25Dataset(meta_csv, roll_samples=roll, n_mels=n_mels), subset_indices)


def get_training_set(split: int = 25, device: Optional[str] = None, roll: int = 0, n_mels: int = 128) -> Dataset:
    """Returns the perfectly isolated training dataset."""
    assert str(split) in ("5", "10", "25", "50", "100"), "split must be in {5, 10, 25, 50, 100}"
    subset_file = download_split_file(f"split{split}.csv")
    return get_dataset_split(dataset_config["meta_csv"], subset_file, device, roll=roll, n_mels=n_mels)


def get_test_set(device: Optional[str] = None, n_mels: int = 128) -> Dataset:
    """Returns the perfectly isolated test dataset."""
    test_split_file = download_split_file(dataset_config["test_split_csv"])
    return get_dataset_split(dataset_config["meta_csv"], test_split_file, device, roll=0, n_mels=n_mels)