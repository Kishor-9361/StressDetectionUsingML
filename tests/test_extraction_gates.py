"""
tests/test_extraction_gates.py

Automated correctness tests for Phase 1 Extraction and Baseline Anchoring:
1. Column-count verification against official baseline targets.
2. Subject count reconciliation and coverage integrity.
3. Modality separation (no merged NaN-padded tables).
4. Window count sanity.
5. Calm-baseline anchoring verification (medians near 0).
"""

from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np
import yaml

FEAT_DIR = Path("data/processed/features")
CALIB_DIR = Path("data/processed/calibrated")
INDEX_PATH = Path("data/processed/index/window_index.parquet")
COVERAGE_REPORT_PATH = Path("data/processed/manifests/coverage_report.json")
BASELINE_PHYSIO_PATH = Path("scratch/stressID_baseline/Feature Extraction/Features/all_physiological_features.csv")


def test_window_counts_sane():
    assert INDEX_PATH.exists(), "window_index.parquet must exist"
    df = pd.read_parquet(INDEX_PATH)
    assert 1000 <= len(df) <= 4000, f"Expected 1000-4000 windows, got {len(df)}"
    assert df["subject_id"].nunique() == 65, "Expected exactly 65 subjects"


def test_no_merged_modality_tables():
    for f in FEAT_DIR.glob("*_features.parquet"):
        df = pd.read_parquet(f)
        cols = df.columns.tolist()
        modality = f.stem.split("_")[0]

        if modality == "ecg":
            assert "meanHR" in cols and "mean_eda" not in cols and "meanRR" not in cols
        elif modality == "eda":
            assert "mean_eda" in cols and "meanHR" not in cols and "meanRR" not in cols
        elif modality == "resp":
            assert "meanRR" in cols and "meanHR" not in cols and "mean_eda" not in cols
        elif modality == "face":
            assert any("AU" in c for c in cols) and "meanHR" not in cols
        elif modality == "voice":
            assert any("voice_feat" in c for c in cols) and "meanHR" not in cols


def test_coverage_matches_manifest():
    assert COVERAGE_REPORT_PATH.exists(), "coverage_report.json must exist"
    with open(COVERAGE_REPORT_PATH, "r") as f:
        rep = json.load(f)

    assert rep["summary"]["total_subjects"] == 65
    assert rep["summary"]["physio_subjects"] == 65
    assert rep["coverage_verification"]["physio_complete_65"] is True
    assert rep["coverage_verification"]["physio_only_matches_disk"] is True
    assert rep["coverage_verification"]["modality_separation_enforced"] is True


def test_column_counts_and_baseline_calibration():
    ecg = pd.read_parquet(FEAT_DIR / "ecg_features.parquet")
    eda = pd.read_parquet(FEAT_DIR / "eda_features.parquet")
    resp = pd.read_parquet(FEAT_DIR / "resp_features.parquet")
    face = pd.read_parquet(FEAT_DIR / "face_features.parquet")
    voice = pd.read_parquet(FEAT_DIR / "voice_features.parquet")

    # Excluding id columns (window_id, subject_id, task_id)
    ecg_feat_count = ecg.shape[1] - 3
    eda_feat_count = eda.shape[1] - 3
    resp_feat_count = resp.shape[1] - 3
    face_feat_count = face.shape[1] - 3
    voice_feat_count = voice.shape[1] - 3

    assert ecg_feat_count == 46, f"Expected 46 ECG cols (35 HRV + 11 stat), got {ecg_feat_count}"
    assert eda_feat_count == 24, f"Expected 24 EDA cols, got {eda_feat_count}"
    assert resp_feat_count == 62, f"Expected 62 RESP cols, got {resp_feat_count}"
    assert face_feat_count == 84, f"Expected 84 Face cols, got {face_feat_count}"
    assert voice_feat_count == 140, f"Expected 140 Voice cols, got {voice_feat_count}"


def test_baseline_calibrated_medians():
    ecg_cal = pd.read_parquet(CALIB_DIR / "ecg_calibrated.parquet")
    eda_cal = pd.read_parquet(CALIB_DIR / "eda_calibrated.parquet")
    resp_cal = pd.read_parquet(CALIB_DIR / "resp_calibrated.parquet")

    ecg_med = ecg_cal[ecg_cal["task_id"] == "Breathing"]["meanHR"].median()
    eda_med = eda_cal[eda_cal["task_id"] == "Breathing"]["mean_eda"].median()
    resp_med = resp_cal[resp_cal["task_id"] == "Relax"]["meanRR"].median()

    assert abs(ecg_med) < 0.05, f"ECG Breathing median should be ~0, got {ecg_med}"
    assert abs(eda_med) < 0.05, f"EDA Breathing median should be ~0, got {eda_med}"
    assert abs(resp_med) < 0.05, f"RESP Relax median should be ~0, got {resp_med}"
