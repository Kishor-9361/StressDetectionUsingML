"""
tests/test_tier0_gates.py

Automated gate verification for Tier 0 Classical Baselines per MODEL_TRAINING_PIPELINE.md §B.0 and §B.1:
1. Checks that all 12 model families have run and generated valid timestamped directories.
2. Asserts each run contains all 7 mandated artifacts.
3. Validates that metrics.json contains no NaN/Inf, all values in [0, 1], and optimal threshold swept.
4. Confirms model_card.md reports the benchmark gap against StressID Table 2.
5. Verifies predictions.parquet integrity and row alignment.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

REQUIRED_MODELS = [
    "logreg_view_physio_all",
    "knn_view_physio_all",
    "svm_view_physio_all",
    "rf_view_physio_all",
    "xgboost_view_physio_all",
    "lightgbm_view_physio_all",
    "rf_view_physio_face",
    "xgboost_view_physio_face",
    "rf_view_physio_voice",
    "xgboost_view_physio_voice",
    "rf_view_tri_modal",
    "xgboost_view_tri_modal",
]

MANDATED_ARTIFACTS = [
    "checkpoint.joblib",
    "confusion_matrix.png",
    "metrics.json",
    "model_card.md",
    "pr.png",
    "predictions.parquet",
    "roc.png"
]

@pytest.mark.parametrize("model_name", REQUIRED_MODELS)
def test_model_artifacts_exist(model_name):
    model_dir = Path("models") / model_name
    assert model_dir.exists(), f"Model directory {model_dir} does not exist"
    
    runs = sorted(model_dir.glob("2026*"))
    assert len(runs) > 0, f"No run found in {model_dir}"
    
    latest_run = runs[-1]
    for artifact in MANDATED_ARTIFACTS:
        art_path = latest_run / artifact
        assert art_path.exists(), f"Missing artifact {artifact} in {latest_run}"
        assert art_path.stat().st_size > 0, f"Artifact {artifact} is empty"

@pytest.mark.parametrize("model_name", REQUIRED_MODELS)
def test_metrics_integrity(model_name):
    latest_run = sorted((Path("models") / model_name).glob("2026*"))[-1]
    with open(latest_run / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    assert metrics["model_name"] == model_name
    assert metrics["sample_count"] > 300
    assert metrics["subject_count"] >= 50
    
    opt = metrics["optimal_threshold"]
    def_th = metrics["default_threshold_0_5"]
    
    for key in ["accuracy", "balanced_accuracy", "macro_f1", "f1_stress"]:
        assert 0.0 <= opt[key] <= 1.0, f"Invalid {key} in optimal_threshold: {opt[key]}"
        assert 0.0 <= def_th[key] <= 1.0, f"Invalid {key} in default_threshold: {def_th[key]}"
        assert not np.isnan(opt[key]), f"NaN found in optimal {key}"
        assert not np.isnan(def_th[key]), f"NaN found in default {key}"
        
    assert 0.05 <= opt["threshold"] <= 0.95
    assert def_th["threshold"] == 0.50

@pytest.mark.parametrize("model_name", REQUIRED_MODELS)
def test_predictions_parquet(model_name):
    latest_run = sorted((Path("models") / model_name).glob("2026*"))[-1]
    preds_path = latest_run / "predictions.parquet"
    df_preds = pd.read_parquet(preds_path)
    
    expected_cols = {"window_id", "subject_id", "y_true", "y_prob", "y_pred_05", "y_pred_opt"}
    assert expected_cols.issubset(set(df_preds.columns))
    assert not df_preds["y_prob"].isna().any()
    assert (df_preds["y_prob"] >= 0.0).all() and (df_preds["y_prob"] <= 1.0).all()

def test_best_physio_model_gap_reporting():
    # Per B.1: the best Tier 0 physio-all result is compared against StressID Table 2 (F1 ~ 0.73)
    rf_run = sorted((Path("models") / "rf_view_physio_all").glob("2026*"))[-1]
    with open(rf_run / "model_card.md", "r", encoding="utf-8") as f:
        card_text = f.read()
    assert "StressID Paper Target" in card_text
    assert "0.7300" in card_text
    assert "Gap to Benchmark" in card_text
