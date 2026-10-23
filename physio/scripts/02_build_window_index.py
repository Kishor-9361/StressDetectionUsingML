"""
scripts/02_build_window_index.py

Stage 2 — Window Index Construction
Builds window_index.parquet with:
subject_id, task_id, window_id, window_start_s, window_end_s, duration_s,
label_binary, affect3_class, ratings (stress, relax, valence, arousal),
modality_coverage, is_baseline, is_subwindowed.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from stressres.windows.grid import build_windows_for_recording

CLEAN_DIR = Path("data/processed/clean")
INDEX_DIR = Path("data/processed/index")
RAW_ROOT = Path("Datasets/stressid")


def load_labels_lookup():
    labels_df = pd.read_csv(RAW_ROOT / "labels.csv")
    supp_df = pd.read_csv(RAW_ROOT / "labels_supplementary.csv")

    merged_lbl = pd.merge(labels_df, supp_df, on="subject/task", how="outer")

    lookup = {}
    for _, row in merged_lbl.iterrows():
        key = str(row["subject/task"])
        lookup[key] = {
            "binary_stress": row.get("binary-stress", np.nan),
            "affect3_class": row.get("affect3-class", np.nan),
            "binary_relax": row.get("binary-relax", np.nan),
            "binary_valence": row.get("binary-valence", np.nan),
            "binary_arousal": row.get("binary-arousal", np.nan),
        }
    return lookup


def load_ratings_lookup():
    self_assess_df = pd.read_csv(RAW_ROOT / "self_assessments.csv", sep=";")
    lookup = {}  # key: (subject, task) -> dict of ratings

    for _, row in self_assess_df.iterrows():
        task_rating = str(row["Tasks"])
        if "_" not in task_rating:
            continue
        parts = task_rating.rsplit("_", 1)
        task_name = parts[0]
        dimension = parts[1]  # stress, relax, valence, arousal

        for sub in self_assess_df.columns[1:]:
            val = row[sub]
            key = f"{sub}_{task_name}"
            if key not in lookup:
                lookup[key] = {}
            try:
                lookup[key][dimension] = float(val) if pd.notnull(val) else np.nan
            except Exception:
                lookup[key][dimension] = np.nan
    return lookup


def main():
    with open("config/subjects.yaml", "r") as f:
        config = yaml.safe_load(f)["stressid"]

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    labels_lookup = load_labels_lookup()
    ratings_lookup = load_ratings_lookup()

    all_windows = []

    tri_modal = set(config.get("tri_modal_subjects", []))
    physio_only = set(config.get("physio_only", []))
    physio_audio = set(config.get("physio_audio_only", []))
    physio_video = set(config.get("physio_video_only", []))

    clean_subs = sorted([d.name for d in CLEAN_DIR.iterdir() if d.is_dir()])
    print(f"Building window index from {len(clean_subs)} cleaned subjects...")

    fs = 500.0

    for sub in clean_subs:
        if sub in tri_modal:
            coverage = "physio+video+audio"
        elif sub in physio_audio:
            coverage = "physio+audio"
        elif sub in physio_video:
            coverage = "physio+video"
        else:
            coverage = "physio"

        sub_clean_dir = CLEAN_DIR / sub
        parquets = sorted(sub_clean_dir.glob("*_physio.parquet"))

        for pq in parquets:
            task = pq.name.replace("_physio.parquet", "")
            # Read row count to get duration
            df_sample = pd.read_parquet(pq, columns=["clean_ecg"])
            n_samples = len(df_sample)
            duration_s = n_samples / fs

            wins = build_windows_for_recording(sub, task, duration_s, modality_coverage=coverage)

            key = f"{sub}_{task}"
            lbl_info = labels_lookup.get(key, {})
            rtg_info = ratings_lookup.get(key, {})

            for w in wins:
                w["label_binary"] = lbl_info.get("binary_stress", np.nan)
                w["label_source"] = "self_report"
                w["affect3_class"] = lbl_info.get("affect3_class", np.nan)
                w["binary_relax"] = lbl_info.get("binary_relax", np.nan)
                w["rating_stress"] = rtg_info.get("stress", np.nan)
                w["rating_relax"] = rtg_info.get("relax", np.nan)
                w["rating_valence"] = rtg_info.get("valence", np.nan)
                w["rating_arousal"] = rtg_info.get("arousal", np.nan)
                all_windows.append(w)

    window_df = pd.DataFrame(all_windows)
    out_pq = INDEX_DIR / "window_index.parquet"
    window_df.to_parquet(out_pq, index=False)

    print("\n" + "=" * 60)
    print("STAGE 2: WINDOW INDEX SUMMARY")
    print("=" * 60)
    print(f"Total Windows: {len(window_df)}")
    print(f"Total Subjects: {window_df['subject_id'].nunique()}")
    print("\nWindows per Task:")
    print(window_df["task_id"].value_counts())
    print("\nModality Coverage:")
    print(window_df["modality_coverage"].value_counts())
    print("\nBinary Label Distribution (non-null):")
    print(window_df["label_binary"].value_counts(dropna=False))
    print(f"\nSaved window index to: {out_pq}")


if __name__ == "__main__":
    main()
