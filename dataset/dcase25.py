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

dataset_config = {
    "dataset_name": "tau25",
    "meta_csv": os.path.join(dataset_dir, "meta_clean.csv"),
    "split_path": "split_setup",
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
    "eval_dir": None,  
    "eval_fold_csv": None
}


# ==========================================
# 🚀 THE DEVICE AUGMENTATION TRICK
# ==========================================
class RandomMicEQ(torch.nn.Module):
    """
    Simulates the MicIRP trick by randomly altering the EQ of the audio.
    This prevents the model from overfitting to the training microphones (A, B, C).
    """
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, waveform):
        # 50% chance to apply a fake microphone EQ during training
        if torch.rand(1).item() < 0.5:
            return waveform
        
        # 1. Random Peaking Filter (Simulates a resonant frequency in a cheap mic casing)
        center_freq = torch.randint(300, 6000, (1,)).item()
        gain = torch.randint(-15, 15, (1,)).item() # Cut or boost by up to 15 dB
        Q = torch.rand(1).item() * 1.5 + 0.5 
        
        waveform = torchaudio.functional.equalizer_biquad(
            waveform, self.sample_rate, center_freq, gain, Q
        )
        
        # 2. Random Lowpass Filter (Simulates a muffled microphone or pocket recording)
        if torch.rand(1).item() < 0.3:
            cutoff = torch.randint(4000, 12000, (1,)).item()
            waveform = torchaudio.functional.lowpass_biquad(waveform, self.sample_rate, cutoff)
            
        # 3. Random Highpass Filter (Simulates a tiny smartphone speaker/mic lacking bass)
        if torch.rand(1).item() < 0.3:
            cutoff = torch.randint(100, 800, (1,)).item()
            waveform = torchaudio.functional.highpass_biquad(waveform, self.sample_rate, cutoff)
            
        return waveform


class DCASE25Dataset(Dataset):
    def __init__(self, meta_csv: str, roll_samples: int = 0, is_training: bool = False):
        df = pd.read_csv(meta_csv, sep="\t")
        df["filename"] = df["filename"].str.replace("/", os.sep)        

        self.files = df["filename"].values
        self.devices = df["source_label"].values
        self.cities = LabelEncoder().fit_transform(df["identifier"].apply(lambda loc: loc.split("-")[0]))
        
        self.classes = ['airport', 'bus', 'metro', 'metro_station', 'park',
                        'public_square', 'shopping_mall', 'street_pedestrian',
                        'street_traffic', 'tram']
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = torch.tensor([class_to_idx[lbl] for lbl in df["scene_label"]], dtype=torch.long)

        self.roll_samples = roll_samples
        self.is_training = is_training

        # Initialize our new Fake Mic EQ
        self.mic_eq = RandomMicEQ(sample_rate=44100)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,      
            n_fft=8192,             
            win_length=8192,        
            hop_length=1364,        
            n_mels=256              
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __getitem__(self, index: int):
        audio_path = os.path.join(dataset_dir, self.files[index])
        waveform, sr = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
            
        if self.roll_samples > 0:
            shift = np.random.randint(-self.roll_samples, self.roll_samples + 1)
            waveform = waveform.roll(shift, dims=1)

        # APPLY THE MIC EQ (Only during training!)
        if self.is_training:
            waveform = self.mic_eq(waveform)

        x = self.mel_transform(waveform)
        x = self.amp_to_db(x)
        
        return x, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self) -> int:
        return len(self.files)


class SubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


def download_split_file(split_name: str):
    os.makedirs(dataset_config["split_path"], exist_ok=True)
    split_file = os.path.join(dataset_config["split_path"], split_name)
    if not os.path.isfile(split_file):
        print(f"Downloading {split_name}...")
        download_url_to_file(dataset_config["split_url"] + split_name, split_file)
    return split_file


def get_dataset_split(meta_csv: str, split_csv: str, device: Optional[str] = None, roll: int = 0, is_training: bool = False) -> Dataset:
    meta = pd.read_csv(meta_csv, sep="\t")
    split_files = pd.read_csv(split_csv, sep="\t")["filename"].values
    
    subset_indices = meta[meta["filename"].isin(split_files)].index.tolist()
    
    if device:
        subset_indices = meta.loc[subset_indices, :].query("source_label == @device").index.tolist()
        
    return SubsetDataset(DCASE25Dataset(meta_csv, roll_samples=roll, is_training=is_training), subset_indices)


def get_training_set(split: int = 25, device: Optional[str] = None, roll: int = 0) -> Dataset:
    assert str(split) in ("5", "10", "25", "50", "100"), "split must be in {5, 10, 25, 50, 100}"
    subset_file = download_split_file(f"split{split}.csv")
    # Pass is_training=True so the Mic EQ activates!
    return get_dataset_split(dataset_config["meta_csv"], subset_file, device, roll=roll, is_training=True)


def get_test_set(device: Optional[str] = None) -> Dataset:
    test_split_file = download_split_file(dataset_config["test_split_csv"])
    # Pass is_training=False so validation stays pure
    return get_dataset_split(dataset_config["meta_csv"], test_split_file, device, roll=0, is_training=False)