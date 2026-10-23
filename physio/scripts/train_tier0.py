"""
scripts/train_tier0.py

Executes Tier 0 Classical Baselines per MODEL_TRAINING_PIPELINE.md §B.1:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM RBF)
4. Random Forest
5. XGBoost
6. LightGBM

Features:
- Reads from scope-specific views (default: view_physio_all; also supports view_physio_face, view_physio_voice, view_tri_modal)
- Runs smoke test on 2 folds before full training
- Strict Subject-Grouped Leave-One-Subject-Out (LOSO) cross-validation (GroupKFold by subject_id)
- Pipeline with fold-safe SimpleImputer + StandardScaler (no leakage across folds)
- Decision threshold sweeping (0.05–0.95, step 0.05)
- Automatically emits metrics.json, confusion_matrix.png, roc.png, pr.png, and model_card.md
- Saves trained model checkpoint
"""

# CRITICAL ON WINDOWS: lightgbm must be imported before pandas/pyarrow/scipy
# to prevent OpenMP DLL initialization conflict (access violation 0x0).
import lightgbm as lgb
from lightgbm import LGBMClassifier

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stressres.train.cv import get_subject_folds
from stressres.train.threshold import sweep_thresholds
from stressres.train.smoketest import run_smoke_test
from stressres.train.artifacts import save_model_artifacts

ID_META_COLS = [
    "window_id", "subject_id", "task_id", "window_start_s", "window_end_s",
    "duration_s", "is_subwindowed", "is_baseline", "modality_coverage",
    "label_binary", "label_source", "affect3_class", "binary_relax",
    "rating_stress", "rating_relax", "rating_valence", "rating_arousal",
    "is_talking_task"
]

def get_model_pipeline(model_type: str, params: dict):
    """
    Constructs a scikit-learn Pipeline with fold-safe imputation and scaling.
    """
    imputer = SimpleImputer(strategy="median")
    
    if model_type == "logreg":
        scaler = StandardScaler()
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
        return Pipeline([("imputer", imputer), ("scaler", scaler), ("clf", clf)])
        
    elif model_type == "knn":
        scaler = StandardScaler()
        clf = KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 11),
            weights=params.get("weights", "distance")
        )
        return Pipeline([("imputer", imputer), ("scaler", scaler), ("clf", clf)])
        
    elif model_type == "svm":
        scaler = StandardScaler()
        clf = SVC(
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            class_weight="balanced",
            probability=True,
            random_state=42
        )
        return Pipeline([("imputer", imputer), ("scaler", scaler), ("clf", clf)])
        
    elif model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        return Pipeline([("imputer", imputer), ("clf", clf)])
        
    elif model_type == "xgboost":
        clf = XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 4),
            scale_pos_weight=params.get("scale_pos_weight", 1.71),
            learning_rate=0.05,
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss"
        )
        return Pipeline([("imputer", imputer), ("clf", clf)])
        
    elif model_type == "lightgbm":
        clf = LGBMClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 4),
            scale_pos_weight=params.get("scale_pos_weight", 1.71),
            learning_rate=0.05,
            n_jobs=1,
            random_state=42,
            verbose=-1
        )
        return Pipeline([("imputer", imputer), ("clf", clf)])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_and_eval_loso(
    df: pd.DataFrame,
    feature_cols: list,
    model_type: str,
    params: dict,
    model_name: str,
    view_name: str
):
    print(f"\n=======================================================")
    print(f"Training {model_name} on {view_name}")
    print(f"Features: {len(feature_cols)} | Samples: {len(df)} | Subjects: {df['subject_id'].nunique()}")
    print(f"Hyperparameters: {params}")
    print(f"=======================================================")

    X = np.ascontiguousarray(df[feature_cols].values, dtype=np.float32)
    y = np.ascontiguousarray(df["label_binary"].values, dtype=np.int32)
    groups = df["subject_id"].values
    window_ids = df["window_id"].values

    # 1. Smoke test on 2 folds
    print("Running 2-fold smoke test...")
    def smoke_fn(X_tr, y_tr, X_v):
        pipe = get_model_pipeline(model_type, params)
        pipe.fit(X_tr, y_tr)
        return pipe.predict_proba(X_v)[:, 1]

    run_smoke_test(smoke_fn, X, y, groups, model_name=model_name)

    # 2. Strict Subject-Grouped Leave-One-Subject-Out Cross-Validation
    folds = get_subject_folds(df, group_col="subject_id", n_splits=None)
    n_folds = len(folds)
    print(f"Executing Leave-One-Subject-Out cross-validation across {n_folds} folds...")

    y_prob_all = np.zeros(len(y), dtype=float)

    for i, (train_idx, val_idx, sub_id) in enumerate(folds):
        X_train = np.ascontiguousarray(X[train_idx])
        y_train = np.ascontiguousarray(y[train_idx])
        X_val = np.ascontiguousarray(X[val_idx])

        pipe = get_model_pipeline(model_type, params)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_val)[:, 1]
        y_prob_all[val_idx] = probs

        if (i + 1) % 15 == 0 or (i + 1) == n_folds:
            print(f"  Processed {i+1}/{n_folds} folds (Latest subject: {sub_id})...")

    # 3. Decision threshold sweep
    print("Sweeping decision thresholds (0.05 - 0.95)...")
    th_results = sweep_thresholds(y, y_prob_all, start=0.05, stop=0.95, step=0.05)
    opt_th = th_results["optimal_threshold"]
    def_m = th_results["default_metrics"]
    opt_m = th_results["optimal_metrics"]

    print(f"\nResults Summary for {model_name}:")
    print(f"  Default (0.50): Accuracy={def_m['accuracy']:.4f} | BalancedAcc={def_m['balanced_accuracy']:.4f} | MacroF1={def_m['macro_f1']:.4f} | StressF1={def_m['f1']:.4f}")
    print(f"  Optimal ({opt_th:.2f}): Accuracy={opt_m['accuracy']:.4f} | BalancedAcc={opt_m['balanced_accuracy']:.4f} | MacroF1={opt_m['macro_f1']:.4f} | StressF1={opt_m['f1']:.4f}")

    # 4. Train final model on full dataset for checkpointing
    print("Fitting model on full dataset for artifact checkpointing...")
    final_pipeline = get_model_pipeline(model_type, params)
    final_pipeline.fit(X, y)

    # 5. Save all standardized artifacts
    artifacts_dir = save_model_artifacts(
        model_name=model_name,
        view_name=view_name,
        y_true=y,
        y_prob=y_prob_all,
        subject_ids=groups,
        window_ids=window_ids,
        threshold_data=th_results,
        hyperparameters=params,
        feature_names=feature_cols,
        trained_model=final_pipeline,
        evaluation_protocol="Leave-One-Subject-Out (LOSO) GroupKFold",
        notes=f"Tier 0 baseline trained on {view_name} with {len(feature_cols)} features across {n_folds} subjects."
    )

    return {
        "model_name": model_name,
        "view_name": view_name,
        "macro_f1_opt": opt_m["macro_f1"],
        "macro_f1_05": def_m["macro_f1"],
        "acc_opt": opt_m["accuracy"],
        "artifacts_dir": str(artifacts_dir)
    }

def main():
    parser = argparse.ArgumentParser(description="Train Tier 0 Classical Baselines")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "logreg", "knn", "svm", "rf", "xgboost", "lightgbm"])
    parser.add_argument("--view", type=str, default="view_physio_all",
                        choices=["view_physio_all", "view_physio_face", "view_physio_voice", "view_tri_modal"])
    args = parser.parse_args()

    view_path = Path(f"data/processed/views/{args.view}.parquet")
    if not view_path.exists():
        raise FileNotFoundError(f"View not found: {view_path}")

    print(f"Loading dataset from: {view_path}")
    raw_df = pd.read_parquet(view_path)

    # Filter out unlabeled rows (label_binary == NaN)
    df = raw_df[raw_df["label_binary"].notna()].copy()
    
    # If training on view_tri_modal or view_physio_voice, filter to talking tasks where voice is valid
    if args.view in ["view_physio_voice", "view_tri_modal"]:
        talking_tasks = ["Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"]
        df = df[df["task_id"].isin(talking_tasks)].copy()
        print(f"Filtered {args.view} to {len(df)} talking tasks where audio is valid.")

    feature_cols = [c for c in df.columns if c not in ID_META_COLS]
    print(f"Features loaded: {len(feature_cols)} features, {len(df)} valid labeled samples.")

    # Grid specifications per MODEL_TRAINING_PIPELINE.md §B.1
    grids = {
        "logreg": {"C": 1.0},
        "knn": {"n_neighbors": 11, "weights": "distance"},
        "svm": {"C": 1.0, "gamma": "scale"},
        "rf": {"n_estimators": 100, "max_depth": 12},
        "xgboost": {"max_depth": 4, "n_estimators": 200, "scale_pos_weight": 1.71},
        "lightgbm": {"max_depth": 4, "n_estimators": 200, "scale_pos_weight": 1.71}
    }

    models_to_run = list(grids.keys()) if args.model == "all" else [args.model]
    results = []

    for m_type in models_to_run:
        params = grids[m_type]
        m_name = f"{m_type}_{args.view}"
        res = train_and_eval_loso(
            df=df,
            feature_cols=feature_cols,
            model_type=m_type,
            params=params,
            model_name=m_name,
            view_name=args.view
        )
        results.append(res)

    print("\n=======================================================")
    print("ALL REQUESTED TIER 0 RUNS COMPLETED:")
    for r in results:
        print(f"  {r['model_name']} ({r['view_name']}): Macro F1 (opt) = {r['macro_f1_opt']:.4f} (at 0.5: {r['macro_f1_05']:.4f}) | Acc = {r['acc_opt']:.4f}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
