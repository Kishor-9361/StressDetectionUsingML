"""
scripts/extract_resp.py

Stage 3 — Extract Respiration features per window into resp_features.parquet.
Extracts 62 features (38 breathing/time, 11 stat, 9 frequency, 4 nonlinear).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from stressres.features.resp import extract_resp_window_features

INDEX_PATH = Path("data/processed/index/window_index.parquet")
CLEAN_DIR = Path("data/processed/clean")
FEAT_DIR = Path("data/processed/features")


def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing {INDEX_PATH}. Run scripts/02_build_window_index.py first.")

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    window_df = pd.read_parquet(INDEX_PATH)
    print(f"Extracting Respiration features for {len(window_df)} windows...")

    fs = 500.0
    records = []

    cached_key = None
    cached_signal = None

    for _, row in tqdm(window_df.iterrows(), total=len(window_df), desc="RESP Windows"):
        sub = row["subject_id"]
        task = row["task_id"]
        win_id = row["window_id"]
        start_s = row["window_start_s"]
        end_s = row["window_end_s"]

        cache_key = f"{sub}_{task}"
        if cache_key != cached_key:
            pq_file = CLEAN_DIR / sub / f"{task}_physio.parquet"
            if pq_file.exists():
                cached_signal = pd.read_parquet(pq_file, columns=["clean_resp"])["clean_resp"].values
            else:
                cached_signal = None
            cached_key = cache_key

        if cached_signal is None or len(cached_signal) == 0:
            feat = {"window_id": win_id, "subject_id": sub, "task_id": task}
            records.append(feat)
            continue

        start_idx = int(start_s * fs)
        end_idx = min(int(end_s * fs), len(cached_signal))
        win_sig = cached_signal[start_idx:end_idx]

        feat = extract_resp_window_features(win_sig, fs=fs)
        feat["window_id"] = win_id
        feat["subject_id"] = sub
        feat["task_id"] = task
        records.append(feat)

    resp_df = pd.DataFrame(records)
    id_cols = ["window_id", "subject_id", "task_id"]
    other_cols = [c for c in resp_df.columns if c not in id_cols]
    resp_df = resp_df[id_cols + other_cols]

    out_file = FEAT_DIR / "resp_features.parquet"
    resp_df.to_parquet(out_file, index=False)
    print(f"\nSaved {len(resp_df)} RESP feature rows to {out_file}")
    print(f"Total RESP feature columns: {len(other_cols)} (Target 40-62)")


if __name__ == "__main__":
    main()
