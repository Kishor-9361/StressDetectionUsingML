"""
scripts/combine_features.py

Executes Part A of MODEL_TRAINING_PIPELINE.md:
1. Combines the 3 physio tables (ECG, EDA, RESP) into physio_combined.parquet
   keyed identically across all 65 subjects and 1,476 windows.
2. Builds scope-specific joined views under data/processed/views/:
   - view_physio_all.parquet (all 65 subjects, 1,476 rows, 132 physio features)
   - view_physio_face.parquet (52 video/face subjects, 1,190 rows, 216 features)
   - view_physio_voice.parquet (53 audio/voice subjects, 1,213 rows, 272 features)
   - view_tri_modal.parquet (50 tri-modal subjects, 1,144 rows, 356 features,
                             with is_talking_task flag for 350 full-tri-sensor windows)
3. Generates data/processed/manifests/combine_manifest.json with exact subject lists,
   row counts, feature counts, and source table provenance.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

VIEWS_DIR = Path("data/processed/views")
VIEWS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT = Path("data/processed/manifests/combine_manifest.json")

def main():
    print("Loading feature tables, window index, and coverage report...")
    win_df = pd.read_parquet("data/processed/index/window_index.parquet")
    ecg_df = pd.read_parquet("data/processed/calibrated/ecg_calibrated.parquet")
    eda_df = pd.read_parquet("data/processed/calibrated/eda_calibrated.parquet")
    resp_df = pd.read_parquet("data/processed/calibrated/resp_calibrated.parquet")
    face_df = pd.read_parquet("data/processed/calibrated/face_calibrated.parquet")
    voice_df = pd.read_parquet("data/processed/calibrated/voice_calibrated.parquet")

    with open("data/processed/manifests/coverage_report.json", "r", encoding="utf-8") as f:
        cov_report = json.load(f)

    id_cols = ["window_id", "subject_id", "task_id"]
    meta_cols = [
        "window_start_s", "window_end_s", "duration_s", "is_subwindowed",
        "is_baseline", "modality_coverage", "label_binary", "label_source",
        "affect3_class", "binary_relax", "rating_stress", "rating_relax",
        "rating_valence", "rating_arousal"
    ]
    meta_df = win_df[id_cols + meta_cols].copy()

    # Feature column lists
    ecg_features = [c for c in ecg_df.columns if c not in id_cols]
    eda_features = [c for c in eda_df.columns if c not in id_cols]
    resp_features = [c for c in resp_df.columns if c not in id_cols]
    face_features = [c for c in face_df.columns if c not in id_cols]
    voice_features = [c for c in voice_df.columns if c not in id_cols]

    physio_features = ecg_features + eda_features + resp_features

    # 1. Combine Physio Tables (Lossless inner join on window_id)
    print("Combining ECG, EDA, and RESP tables...")
    physio_combined = meta_df.merge(ecg_df[id_cols + ecg_features], on=id_cols, how="inner")
    physio_combined = physio_combined.merge(eda_df[id_cols + eda_features], on=id_cols, how="inner")
    physio_combined = physio_combined.merge(resp_df[id_cols + resp_features], on=id_cols, how="inner")

    assert len(physio_combined) == 1476, f"Expected 1476 rows, got {len(physio_combined)}"
    assert physio_combined["subject_id"].nunique() == 65, f"Expected 65 subjects, got {physio_combined['subject_id'].nunique()}"

    physio_comb_path = VIEWS_DIR / "physio_combined.parquet"
    physio_combined.to_parquet(physio_comb_path, index=False)
    print(f"Saved {physio_comb_path}: shape {physio_combined.shape}")

    # 2. View 1: view_physio_all.parquet (All 65 subjects)
    view_physio_all = physio_combined.copy()
    view_physio_all_path = VIEWS_DIR / "view_physio_all.parquet"
    view_physio_all.to_parquet(view_physio_all_path, index=False)
    print(f"Saved {view_physio_all_path}: shape {view_physio_all.shape}")

    # 3. View 2: view_physio_face.parquet (Inner-joined on subjects with valid video/face)
    face_subjects = sorted(cov_report["extracted_feature_subjects"]["face"])
    print(f"Building view_physio_face for {len(face_subjects)} face subjects...")
    view_physio_face = physio_combined[physio_combined["subject_id"].isin(face_subjects)].copy()
    view_physio_face = view_physio_face.merge(face_df[id_cols + face_features], on=id_cols, how="inner")
    
    view_physio_face_path = VIEWS_DIR / "view_physio_face.parquet"
    view_physio_face.to_parquet(view_physio_face_path, index=False)
    print(f"Saved {view_physio_face_path}: shape {view_physio_face.shape}")

    # 4. View 3: view_physio_voice.parquet (Inner-joined on subjects with valid audio/voice)
    voice_subjects = sorted(cov_report["extracted_feature_subjects"]["voice"])
    print(f"Building view_physio_voice for {len(voice_subjects)} voice subjects...")
    view_physio_voice = physio_combined[physio_combined["subject_id"].isin(voice_subjects)].copy()
    view_physio_voice = view_physio_voice.merge(voice_df[id_cols + voice_features], on=id_cols, how="inner")
    
    view_physio_voice_path = VIEWS_DIR / "view_physio_voice.parquet"
    view_physio_voice.to_parquet(view_physio_voice_path, index=False)
    print(f"Saved {view_physio_voice_path}: shape {view_physio_voice.shape}")

    # 5. View 4: view_tri_modal.parquet (Subjects with all three modalities present)
    tri_subjects = sorted(list(set(face_subjects).intersection(set(voice_subjects))))
    print(f"Building view_tri_modal for {len(tri_subjects)} tri-modal subjects...")
    view_tri_modal = physio_combined[physio_combined["subject_id"].isin(tri_subjects)].copy()
    view_tri_modal = view_tri_modal.merge(face_df[id_cols + face_features], on=id_cols, how="inner")
    view_tri_modal = view_tri_modal.merge(voice_df[id_cols + voice_features], on=id_cols, how="inner")
    
    # Helper flag for talking tasks where voice is non-null
    talking_tasks = ["Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"]
    view_tri_modal["is_talking_task"] = view_tri_modal["task_id"].isin(talking_tasks)
    
    view_tri_modal_path = VIEWS_DIR / "view_tri_modal.parquet"
    view_tri_modal.to_parquet(view_tri_modal_path, index=False)
    print(f"Saved {view_tri_modal_path}: shape {view_tri_modal.shape}")

    # 6. Build combine_manifest.json
    print("Generating combine_manifest.json...")
    manifest = {
        "version": "1.0",
        "description": "Combine manifest specifying scope-specific feature views for Phase 4 model training",
        "views": {
            "view_physio_all": {
                "file": "data/processed/views/view_physio_all.parquet",
                "scope": "All subjects with physiological recordings (largest N scope)",
                "subject_count": int(view_physio_all["subject_id"].nunique()),
                "subjects": sorted(view_physio_all["subject_id"].unique().tolist()),
                "row_count": len(view_physio_all),
                "feature_count": len(physio_features),
                "modalities": ["ecg", "eda", "resp"],
                "source_tables": [
                    "data/processed/calibrated/ecg_calibrated.parquet",
                    "data/processed/calibrated/eda_calibrated.parquet",
                    "data/processed/calibrated/resp_calibrated.parquet"
                ],
                "binary_class_balance": {
                    "non_stress": int((view_physio_all["label_binary"] == 0.0).sum()),
                    "stress": int((view_physio_all["label_binary"] == 1.0).sum()),
                    "unlabeled": int(view_physio_all["label_binary"].isna().sum())
                }
            },
            "view_physio_face": {
                "file": "data/processed/views/view_physio_face.parquet",
                "scope": "Subjects with physiological and video/facial action unit data",
                "subject_count": int(view_physio_face["subject_id"].nunique()),
                "subjects": sorted(view_physio_face["subject_id"].unique().tolist()),
                "row_count": len(view_physio_face),
                "feature_count": len(physio_features) + len(face_features),
                "modalities": ["ecg", "eda", "resp", "face"],
                "source_tables": [
                    "data/processed/views/physio_combined.parquet",
                    "data/processed/calibrated/face_calibrated.parquet"
                ],
                "binary_class_balance": {
                    "non_stress": int((view_physio_face["label_binary"] == 0.0).sum()),
                    "stress": int((view_physio_face["label_binary"] == 1.0).sum()),
                    "unlabeled": int(view_physio_face["label_binary"].isna().sum())
                }
            },
            "view_physio_voice": {
                "file": "data/processed/views/view_physio_voice.parquet",
                "scope": "Subjects with physiological and acoustic voice data",
                "subject_count": int(view_physio_voice["subject_id"].nunique()),
                "subjects": sorted(view_physio_voice["subject_id"].unique().tolist()),
                "row_count": len(view_physio_voice),
                "talking_tasks_row_count": int(view_physio_voice["task_id"].isin(talking_tasks).sum()),
                "feature_count": len(physio_features) + len(voice_features),
                "modalities": ["ecg", "eda", "resp", "voice"],
                "source_tables": [
                    "data/processed/views/physio_combined.parquet",
                    "data/processed/calibrated/voice_calibrated.parquet"
                ],
                "binary_class_balance": {
                    "non_stress": int((view_physio_voice["label_binary"] == 0.0).sum()),
                    "stress": int((view_physio_voice["label_binary"] == 1.0).sum()),
                    "unlabeled": int(view_physio_voice["label_binary"].isna().sum())
                }
            },
            "view_tri_modal": {
                "file": "data/processed/views/view_tri_modal.parquet",
                "scope": "Subjects with complete tri-modal data (physio + face + voice)",
                "subject_count": int(view_tri_modal["subject_id"].nunique()),
                "subjects": sorted(view_tri_modal["subject_id"].unique().tolist()),
                "row_count": len(view_tri_modal),
                "talking_tasks_row_count": int((view_tri_modal["is_talking_task"]).sum()),
                "feature_count": len(physio_features) + len(face_features) + len(voice_features),
                "modalities": ["ecg", "eda", "resp", "face", "voice"],
                "source_tables": [
                    "data/processed/views/physio_combined.parquet",
                    "data/processed/calibrated/face_calibrated.parquet",
                    "data/processed/calibrated/voice_calibrated.parquet"
                ],
                "binary_class_balance": {
                    "all_tasks": {
                        "non_stress": int((view_tri_modal["label_binary"] == 0.0).sum()),
                        "stress": int((view_tri_modal["label_binary"] == 1.0).sum()),
                        "unlabeled": int(view_tri_modal["label_binary"].isna().sum())
                    },
                    "talking_tasks_only": {
                        "non_stress": int(((view_tri_modal["is_talking_task"]) & (view_tri_modal["label_binary"] == 0.0)).sum()),
                        "stress": int(((view_tri_modal["is_talking_task"]) & (view_tri_modal["label_binary"] == 1.0)).sum()),
                        "unlabeled": int(((view_tri_modal["is_talking_task"]) & (view_tri_modal["label_binary"].isna())).sum())
                    }
                }
            }
        }
    }

    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved combine manifest: {MANIFEST_OUT}")

if __name__ == "__main__":
    main()
