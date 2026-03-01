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
assert dataset_dir, "Specify 'TAU Urban Acoustic Scenes 2022 Mobile' dataset location in 'dataset_dir'. Download from: https://zenodo.org/record/6337421"

# Dataset configuration
dataset_config = {
    "dataset_name": "tau25",
    "meta_csv": os.path.join(dataset_dir, "meta_clean.csv"),
    "split_path": "split_setup",
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
    "eval_dir": None,  # Evaluation set release on 1st of June
    "eval_fold_csv": None
}


class DCASE25Dataset(Dataset):
    """
    DCASE'25 Dataset: Loads metadata and provides access to audio samples.
    """
    def __init__(self, meta_csv: str):
        """
        Initializes the dataset.

        Args:
            meta_csv (str): Path to the dataset metadata CSV file.
        """
        df = pd.read_csv(meta_csv, sep="\t")
        df["filename"] = df["filename"].str.replace("/", os.sep)        

        self.files = df["filename"].values
        self.devices = df["source_label"].values
        self.cities = LabelEncoder().fit_transform(df["identifier"].apply(lambda loc: loc.split("-")[0]))
        self.labels = torch.tensor(LabelEncoder().fit_transform(df["scene_label"]), dtype=torch.long)

    def __getitem__(self, index: int):
        """Loads an audio sample and corresponding metadata."""
        audio_path = os.path.join(dataset_dir, self.files[index])
        waveform, _ = torchaudio.load(audio_path)
        waveform = waveform.squeeze()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  
            
           
        
        return waveform, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self) -> int:
        return len(self.files)


class SubsetDataset(Dataset):
    """
    A dataset that selects a subset of samples based on given indices.
    """

    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


class TimeShiftDataset(Dataset):
    """
    A dataset implementing time shifting of waveforms.
    """

    def __init__(self, dataset: Dataset, shift_range: int, axis: int = 1):
        self.dataset = dataset
        self.shift_range = shift_range
        self.axis = axis

    def __getitem__(self, index: int):
        waveform, file, label, device, city = self.dataset[index]
        shift = np.random.randint(-self.shift_range, self.shift_range + 1)
        return waveform.roll(shift, self.axis), file, label, device, city

    def __len__(self) -> int:
        return len(self.dataset)


# --- Dataset Loading Functions --- #

def download_split_file(split_name: str):
    """Downloads dataset split files if not available."""
    os.makedirs(dataset_config["split_path"], exist_ok=True)
    split_file = os.path.join(dataset_config["split_path"], split_name)
    if not os.path.isfile(split_file):
        print(f"Downloading {split_name}...")
        download_url_to_file(dataset_config["split_url"] + split_name, split_file)
    return split_file


def get_dataset_split(meta_csv: str, split_csv: str, device: Optional[str] = None) -> Dataset:
    """
    Filters the dataset based on the given split file and optionally by device.
    """
    meta = pd.read_csv(meta_csv, sep="\t")
    split_files = pd.read_csv(split_csv, sep="\t")["filename"].values
    subset_indices = meta[meta["filename"].isin(split_files)].index.tolist()
    if device:
        subset_indices = meta.loc[subset_indices, :].query("source_label == @device").index.tolist()
    return SubsetDataset(DCASE25Dataset(meta_csv), subset_indices)


def get_training_set(split: int = 100, device: Optional[str] = None, roll: int = 0) -> Dataset:
    """
    Returns the training dataset for a specified data split percentage.

    Args:
        split (int): Percentage of the dataset to use [5, 10, 25, 50, 100].
        device (Optional[str]): Specific device to filter on.
        roll (int): Time shift range.
    """
    assert str(split) in ("5", "10", "25", "50", "100"), "split must be in {5, 10, 25, 50, 100}"
    subset_file = download_split_file(f"split{split}.csv")
    dataset = get_dataset_split(dataset_config["meta_csv"], subset_file, device)
    return TimeShiftDataset(dataset, shift_range=roll) if roll else dataset


def get_test_set(device: Optional[str] = None) -> Dataset:
    """Returns the test dataset."""
    test_split_file = download_split_file(dataset_config["test_split_csv"])
    return get_dataset_split(dataset_config["meta_csv"], test_split_file, device)
