import os
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

# Update this path if your dataset is located somewhere else!
DATASET_DIR = "/home/abdalaziz-ayoub/datasets/PR_DATA"
META_CSV = os.path.join(DATASET_DIR, "meta.csv")
OUTPUT_CSV = os.path.join(DATASET_DIR, "meta_clean.csv")

def is_valid_audio(path: str) -> bool:
    try:
        # 1. Does the file actually exist on the hard drive?
        if not os.path.exists(path):
            return False

        # 2. Can PyTorch Audio read it without crashing?
        waveform, sr = torchaudio.load(path)

        # 3. Is the file completely empty (0 bytes)?
        if waveform.numel() == 0:
            return False

        # 4. Does the audio contain corrupted math (NaNs or Infs)?
        if not torch.isfinite(waveform).all():
            return False

        # 5. Is it a valid 2D tensor? Expected shape: (Channels, Time)
        if waveform.dim() != 2:
            return False

        return True

    except Exception:
        return False

def main():
    print(f"Reading metadata from: {META_CSV}")
    df = pd.read_csv(META_CSV, sep="\t")

    valid_rows = []
    corrupted_files = []

    print("\nValidating audio files...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = os.path.join(DATASET_DIR, row["filename"])

        if is_valid_audio(audio_path):
            valid_rows.append(row)
        else:
            corrupted_files.append(row["filename"])

    clean_df = pd.DataFrame(valid_rows)

    print("\n" + "=" * 40)
    print("Validation Summary")
    print("=" * 40)
    print(f"Total files in meta.csv : {len(df)}")
    print(f"Valid files             : {len(clean_df)}")
    print(f"Corrupted/Missing files : {len(corrupted_files)}")

    if corrupted_files:
        print("\nExamples of corrupted files:")
        for f in corrupted_files[:10]:
            print(f"  - {f}")

    # Save the clean metadata
    clean_df.to_csv(OUTPUT_CSV, sep="\t", index=False)
    print(f"\nClean metadata saved successfully to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
    if os.path.exists(OUTPUT_CSV):
        df_check = pd.read_csv(OUTPUT_CSV, sep="\t")
        print(f"\nVerification: Loaded {len(df_check)} valid files from meta_clean.csv")