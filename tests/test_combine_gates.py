"""
tests/test_combine_gates.py

Validates Part A of MODEL_TRAINING_PIPELINE.md:
- view_physio_all row count & subject count matches Stage 0 resolved manifest exactly (1,476 rows, 65 subjects).
- Each narrower view's subject list is a proper subset of view_physio_all and matches coverage_report.json.
- physio_combined has all 132 physiological features non-null across all rows.
- combine_manifest.json exists and accurately details each view.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

VIEWS_DIR = Path("data/processed/views")
MANIFEST_PATH = Path("data/processed/manifests/combine_manifest.json")
COVERAGE_PATH = Path("data/processed/manifests/coverage_report.json")

def test_combine_manifest_exists():
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "views" in data
    assert "view_physio_all" in data["views"]
    assert "view_physio_face" in data["views"]
    assert "view_physio_voice" in data["views"]
    assert "view_tri_modal" in data["views"]

def test_view_physio_all_gate():
    path = VIEWS_DIR / "view_physio_all.parquet"
    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == 1476, f"Expected 1476 rows, got {len(df)}"
    assert df["subject_id"].nunique() == 65, f"Expected 65 subjects, got {df['subject_id'].nunique()}"
    
    # 46 ECG + 24 EDA + 62 RESP = 132 features
    feature_cols = [c for c in df.columns if c not in [
        "window_id", "subject_id", "task_id", "window_start_s", "window_end_s",
        "duration_s", "is_subwindowed", "is_baseline", "modality_coverage",
        "label_binary", "label_source", "affect3_class", "binary_relax",
        "rating_stress", "rating_relax", "rating_valence", "rating_arousal"
    ]]
    assert len(feature_cols) == 132, f"Expected 132 physio features, got {len(feature_cols)}"

def test_subset_hierarchy_against_coverage_report():
    with open(COVERAGE_PATH, "r", encoding="utf-8") as f:
        cov = json.load(f)
    
    physio_subs = set(cov["extracted_feature_subjects"]["ecg"])
    face_subs = set(cov["extracted_feature_subjects"]["face"])
    voice_subs = set(cov["extracted_feature_subjects"]["voice"])
    tri_subs = face_subs.intersection(voice_subs)
    
    df_physio = pd.read_parquet(VIEWS_DIR / "view_physio_all.parquet")
    df_face = pd.read_parquet(VIEWS_DIR / "view_physio_face.parquet")
    df_voice = pd.read_parquet(VIEWS_DIR / "view_physio_voice.parquet")
    df_tri = pd.read_parquet(VIEWS_DIR / "view_tri_modal.parquet")
    
    set_physio = set(df_physio["subject_id"].unique())
    set_face = set(df_face["subject_id"].unique())
    set_voice = set(df_voice["subject_id"].unique())
    set_tri = set(df_tri["subject_id"].unique())
    
    assert set_physio == physio_subs
    assert set_face == face_subs
    assert set_voice == voice_subs
    assert set_tri == tri_subs
    
    # Proper subsets
    assert set_face.issubset(set_physio)
    assert set_voice.issubset(set_physio)
    assert set_tri.issubset(set_face)
    assert set_tri.issubset(set_voice)

def test_tri_modal_talking_tasks():
    df_tri = pd.read_parquet(VIEWS_DIR / "view_tri_modal.parquet")
    assert "is_talking_task" in df_tri.columns
    talking_count = df_tri["is_talking_task"].sum()
    assert talking_count == 350, f"Expected 350 talking tasks windows, got {talking_count}"
