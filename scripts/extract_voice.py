"""
scripts/extract_voice.py

Stage 3 — Voice feature mapping/extraction.
Emits data/processed/features/voice_features.parquet with 140 handcrafted acoustic features.
Only talking tasks for subjects with audio have non-null values.
Silent tasks (Breathing, Relax, Video1, Video2, Baseline) and non-audio subjects are strictly null.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

INDEX_PATH = Path("data/processed/index/window_index.parquet")
FEAT_DIR = Path("data/processed/features")
BASELINE_AUDIO = Path("scratch/stressID_baseline/Feature Extraction/Features/HCfeatures.csv")

TALKING_TASKS = {"Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"}


def main():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    window_df = pd.read_parquet(INDEX_PATH)
    print(f"Mapping/extracting Voice features for {len(window_df)} windows...")

    if not BASELINE_AUDIO.exists():
        raise FileNotFoundError(f"Missing {BASELINE_AUDIO}")

    audio_base = pd.read_csv(BASELINE_AUDIO, header=None)
    id_col = 0
    feat_cols = [f"voice_feat_{i}" for i in range(1, audio_base.shape[1])]

    audio_dict = {}
    for _, row in audio_base.iterrows():
        rec_name = str(row[id_col])
        clean_key = rec_name.replace(".wav", "").replace(".mp4", "")
        audio_dict[clean_key] = {feat_cols[i - 1]: row[i] for i in range(1, audio_base.shape[1])}

    records = []
    for _, row in tqdm(window_df.iterrows(), total=len(window_df), desc="Voice Windows"):
        sub = row["subject_id"]
        task = row["task_id"]
        win_id = row["window_id"]
        cov = row["modality_coverage"]

        rec = {
            "window_id": win_id,
            "subject_id": sub,
            "task_id": task,
        }

        # Confound-aware: only talking tasks with audio coverage are non-null
        if task not in TALKING_TASKS or "audio" not in cov:
            for c in feat_cols:
                rec[c] = np.nan
            records.append(rec)
            continue

        key = f"{sub}_{task}"
        if key in audio_dict:
            rec.update(audio_dict[key])
        else:
            for c in feat_cols:
                rec[c] = np.nan

        records.append(rec)

    voice_df = pd.DataFrame(records)
    out_file = FEAT_DIR / "voice_features.parquet"
    voice_df.to_parquet(out_file, index=False)
    print(f"\nSaved {len(voice_df)} Voice feature rows to {out_file}")
    print(f"Total Voice feature columns: {len(feat_cols)} (Official baseline 140 cols)")


if __name__ == "__main__":
    main()
