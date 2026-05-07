"""
Visualize the effect of n_fft=1024 vs n_fft=8192 on mel spectrograms.
Randomly selects one sample from the DCASE25 dataset (meta_clean.csv),
plots both spectrograms side-by-side with the scene label, and saves to assets/.
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchaudio

warnings.filterwarnings("ignore", message="At least one mel filterbank has all zero values")

DATASET_DIR = "/home/abdalaziz-ayoub/datasets/PR_DATA"
META_CSV    = os.path.join(DATASET_DIR, "meta_clean.csv")
ASSETS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
SAMPLE_RATE = 44100
HOP_LENGTH  = 1364
N_MELS      = 128  # 256 causes empty filterbank bins for n_fft=1024 (only 513 linear bins)
N_FFTS      = [1024, 8192]

os.makedirs(ASSETS_DIR, exist_ok=True)

df = pd.read_csv(META_CSV, sep="\t")
row = df.sample(1, random_state=random.randint(0, 99999)).iloc[0]

rel_path    = row["filename"].replace("/", os.sep)
scene_label = row["scene_label"]
device      = row["source_label"]
location    = row["identifier"]
audio_path  = os.path.join(DATASET_DIR, rel_path)

print(f"Random sample selected : {audio_path}")
print(f"Scene label            : {scene_label}")
print(f"Device                 : {device}")
print(f"Location               : {location}")

waveform, sr = torchaudio.load(audio_path)

if sr != SAMPLE_RATE:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
    waveform  = resampler(waveform)

if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

duration_s = waveform.shape[-1] / SAMPLE_RATE

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(
    f"n_fft Comparison  |  File: {os.path.basename(audio_path)}  |  "
    f"Scene: {scene_label}  |  Device: {device}  |  "
    f"SR={SAMPLE_RATE} Hz  |  Duration={duration_s:.2f}s",
    fontsize=11, fontweight="bold",
)

amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

for ax, n_fft in zip(axes, N_FFTS):
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_db  = amp_to_db(mel_tf(waveform)).squeeze(0).numpy()
    n_frames   = mel_db.shape[1]
    time_res_ms = HOP_LENGTH / SAMPLE_RATE * 1000
    freq_res_hz = SAMPLE_RATE / n_fft

    img = ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_title(
        f"n_fft = {n_fft}\n"
        f"Freq resolution: {freq_res_hz:.1f} Hz/bin  |  "
        f"Time frames: {n_frames}  |  "
        f"Time resolution: {time_res_ms:.1f} ms/frame",
        fontsize=10,
    )
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Mel bin")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Amplitude (dB)")

plt.tight_layout()

out_path = os.path.join(ASSETS_DIR, "nfft_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nPlot saved to: {out_path}")
print()
print("Summary")
print("-------")
print(f"  Sample file : {audio_path}")
print(f"  Scene label : {scene_label}")
print(f"  Device      : {device}")
print(f"  Location    : {location}")
print(f"  Sample rate : {SAMPLE_RATE} Hz")
print(f"  Duration    : {duration_s:.2f} s")
print(f"  n_mels      : {N_MELS}")
print(f"  hop_length  : {HOP_LENGTH}")
for n_fft in N_FFTS:
    n_frames = int(np.ceil(waveform.shape[-1] / HOP_LENGTH)) + 1
    print(f"  n_fft={n_fft:>4d}  ->  freq res = {SAMPLE_RATE/n_fft:.1f} Hz/bin, "
          f"~{n_frames} time frames")
