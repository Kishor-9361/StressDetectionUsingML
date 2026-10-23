"""
scripts/extract_face.py

Stage 3 — Face feature mapping/extraction.
Emits data/processed/features/face_features.parquet with 84 AU+gaze baseline features
(mean + SD of 18 Action Units + mean + SD of 2 gaze directions).
Preserves genuine absence for subjects without video (physio-only and physio+audio).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

INDEX_PATH = Path("data/processed/index/window_index.parquet")
FEAT_DIR = Path("data/processed/features")
BASELINE_VIDEO = Path("scratch/stressID_baseline/Feature Extraction/Features/video11tasks_aus_gaze_mean_std.csv")


def main():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    window_df = pd.read_parquet(INDEX_PATH)
    print(f"Mapping/extracting Face features for {len(window_df)} windows...")

    if not BASELINE_VIDEO.exists():
        raise FileNotFoundError(f"Missing {BASELINE_VIDEO}")

    video_base = pd.read_csv(BASELINE_VIDEO)
    id_col = "id" if "id" in video_base.columns else video_base.columns[0]
    feat_cols = [c for c in video_base.columns if c != id_col]

    video_dict = {}
    for _, row in video_base.iterrows():
        key = str(row[id_col])
        video_dict[key] = row[feat_cols].to_dict()

    records = []
    for _, row in tqdm(window_df.iterrows(), total=len(window_df), desc="Face Windows"):
        sub = row["subject_id"]
        task = row["task_id"]
        win_id = row["window_id"]
        cov = row["modality_coverage"]

        rec = {
            "window_id": win_id,
            "subject_id": sub,
            "task_id": task,
        }

        # If modality coverage does not include video, leave features strictly NaN
        if "video" not in cov:
            for c in feat_cols:
                rec[c] = np.nan
            records.append(rec)
            continue

        key = f"{sub}_{task}"
        if key in video_dict:
            rec.update(video_dict[key])
        else:
            for c in feat_cols:
                rec[c] = np.nan

        records.append(rec)

    face_df = pd.DataFrame(records)
    out_file = FEAT_DIR / "face_features.parquet"
    face_df.to_parquet(out_file, index=False)
    print(f"\nSaved {len(face_df)} Face feature rows to {out_file}")
    print(f"Total Face feature columns: {len(feat_cols)} (Official baseline 84 cols)")


if __name__ == "__main__":
    main()
