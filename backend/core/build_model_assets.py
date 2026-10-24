"""
Rebuild derived model assets for the expert-fusion model.

 * models/expert_models/calm_reference.json  – population calm-state feature means
   (used to re-centre a user's personal calibration baseline).
 * models/registry.json                      – active model entries + SHA-256 hashes.

Training CSVs live in data/training (git-ignored). Run from the repo root:
    python scripts/build_model_assets.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.core.artifact_manifest import ArtifactManifest
from backend.core.version_registry import VersionRegistry
from backend.runtime.runtime_engine import EXPERT_DIR, _MODEL_FILES, _safe_load

DATA = os.path.join(ROOT, "data", "training")
CSV = {"face": "face_indicators_stressid.csv", "voice": "voice_indicators_stressid.csv"}


def main():
    ref, cv_metrics = {}, {}
    for modality, fname in CSV.items():
        df = pd.read_csv(os.path.join(DATA, fname))
        X, y = df.drop(columns="label").values, df["label"].values
        ref[modality] = X[y == 0].mean(axis=0).tolist()

        est = make_pipeline(StandardScaler(), GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))
        p = cross_val_predict(est, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
        cv_metrics[modality] = {"cv_accuracy": round(float(accuracy_score(y, p > 0.5)), 4),
                                "cv_auc": round(float(roc_auc_score(y, p)), 4),
                                "n_samples": int(len(y))}

    with open(os.path.join(EXPERT_DIR, "calm_reference.json"), "w") as f:
        json.dump(ref, f)
    print("wrote calm_reference.json", {k: len(v) for k, v in ref.items()})

    # Registry: only the three production experts remain active.
    reg_path = os.path.join(ROOT, "models", "registry.json")
    old = json.load(open(reg_path)) if os.path.exists(reg_path) else {}
    registry = VersionRegistry(registry_path=reg_path)
    registry._registry["active_models"] = {}
    registry._registry["history_models"] = {}
    for key, (model_file, _) in _MODEL_FILES.items():
        m = ArtifactManifest(
            artifact_id=f"{key}_v3", artifact_type="model", version="3.0.0",
            metadata={
                "framework": "scikit-learn",
                "estimator": type(_safe_load(os.path.join(EXPERT_DIR, model_file))).__name__,
                **cv_metrics.get(key.replace("_expert", ""), {}),
            })
        m.compute_hash(os.path.join(EXPERT_DIR, model_file))
        registry.register_model(key, m)
        print("registered", key, m.hash[:12])


if __name__ == "__main__":
    main()
