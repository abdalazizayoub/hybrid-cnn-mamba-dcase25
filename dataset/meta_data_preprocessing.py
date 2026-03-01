import os
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

DATASET_DIR = "/home/abdalaziz-ayoub/datasets/PR_DATA"
META_CSV = os.path.join(DATASET_DIR, "meta.csv")
OUTPUT_CSV = os.path.join(DATASET_DIR, "meta_clean.csv")


def is_valid_audio(path: str) -> bool:
    try:
        waveform, sr = torchaudio.load(path)

        # Empty tensor
        if waveform.numel() == 0:
            return False

        # NaN or Inf
        if not torch.isfinite(waveform).all():
            return False

        # Expected shape: (C, T)l,
        if waveform.dim() != 2:
            return False

        return True

    except Exception:
        return False


def main():
    df = pd.read_csv(META_CSV, sep="\t")

    valid_rows = []
    corrupted_files = []

    print("Validating audio files...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = os.path.join(DATASET_DIR, row["filename"])

        if is_valid_audio(audio_path):
            valid_rows.append(row)
        else:
            corrupted_files.append(row["filename"])

    clean_df = pd.DataFrame(valid_rows)

    print("\nValidation summary")
    print("-" * 40)
    print(f"Total files     : {len(df)}")
    print(f"Valid files     : {len(clean_df)}")
    print(f"Corrupted files : {len(corrupted_files)}")

    if corrupted_files:
        print("\nExamples of corrupted files:")
        for f in corrupted_files[:10]:
            print(" ", f)

    clean_df.to_csv(OUTPUT_CSV, sep="\t", index=False)
    print(f"\nClean metadata saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    df = pd.read_csv("/home/abdalaziz-ayoub/datasets/PR_DATA/meta_clean.csv", sep="\t")
    print(f"Total valid files: {len(df)}")