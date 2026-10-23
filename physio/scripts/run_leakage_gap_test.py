"""
scripts/run_leakage_gap_test.py

Executes the Tier 2 Leakage-Gap Test per MODEL_TRAINING_PIPELINE.md §B.3:
1. Evaluates the best Tier 0 model (Random Forest) on view_physio_all.
2. Compares:
   - Subject-independent Leave-One-Subject-Out (LOSO) GroupKFold (Acc_LOSO)
   - Random row-wise StratifiedKFold (Acc_random)
3. Computes Δ_leak = Acc_random - Acc_LOSO.
4. Decision Rule:
   - If Δ_leak < 0.05: Stop, no GRL needed. Subject identity leakage is negligible.
   - If Δ_leak >= 0.05: GRL adversarial identity suppression is indicated.
5. Saves results to reports/leakage_gap_report.json and renders a comparison plot.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stressres.train.cv import get_subject_folds
from stressres.train.threshold import sweep_thresholds
from stressres.train.smoketest import run_smoke_test

ID_META_COLS = [
    "window_id", "subject_id", "task_id", "window_start_s", "window_end_s",
    "duration_s", "is_subwindowed", "is_baseline", "modality_coverage",
    "label_binary", "label_source", "affect3_class", "binary_relax",
    "rating_stress", "rating_relax", "rating_valence", "rating_arousal",
    "is_talking_task"
]

def get_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42))
    ])

def run_leakage_gap_test():
    view_path = Path("data/processed/views/view_physio_all.parquet")
    df = pd.read_parquet(view_path)
    df = df[df["label_binary"].notna()].copy()
    
    feature_cols = [c for c in df.columns if c not in ID_META_COLS]
    X = np.ascontiguousarray(df[feature_cols].values, dtype=np.float32)
    y = np.ascontiguousarray(df["label_binary"].values, dtype=np.int32)
    groups = df["subject_id"].values
    
    print(f"Running Leakage-Gap Test on {len(df)} samples across {df['subject_id'].nunique()} subjects...")
    print(f"Feature count: {len(feature_cols)}")
    
    # 1. Subject-Grouped Leave-One-Subject-Out (LOSO)
    print("\n--- Protocol 1: Subject-Grouped Leave-One-Subject-Out (LOSO) ---")
    loso_folds = get_subject_folds(df, group_col="subject_id", n_splits=None)
    y_prob_loso = np.zeros(len(y), dtype=float)
    
    for i, (train_idx, val_idx, sub_id) in enumerate(loso_folds):
        pipe = get_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        y_prob_loso[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]
        if (i + 1) % 15 == 0 or (i + 1) == len(loso_folds):
            print(f"  LOSO Fold {i+1}/{len(loso_folds)} complete...")
            
    th_loso = sweep_thresholds(y, y_prob_loso, start=0.05, stop=0.95, step=0.05)
    acc_loso_05 = th_loso["default_metrics"]["accuracy"]
    acc_loso_opt = th_loso["optimal_metrics"]["accuracy"]
    f1_loso_opt = th_loso["optimal_metrics"]["macro_f1"]
    
    print(f"LOSO Results: Acc(0.5)={acc_loso_05:.4f} | Acc(opt)={acc_loso_opt:.4f} | MacroF1(opt)={f1_loso_opt:.4f}")
    
    # 2. Random Row-Wise Stratified K-Fold (Un-grouped, Leaky Protocol)
    print("\n--- Protocol 2: Random Row-Wise Stratified K-Fold (Un-grouped) ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_prob_random = np.zeros(len(y), dtype=float)
    
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        pipe = get_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        y_prob_random[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]
        print(f"  Random Fold {i+1}/10 complete...")
        
    th_random = sweep_thresholds(y, y_prob_random, start=0.05, stop=0.95, step=0.05)
    acc_rand_05 = th_random["default_metrics"]["accuracy"]
    acc_rand_opt = th_random["optimal_metrics"]["accuracy"]
    f1_rand_opt = th_random["optimal_metrics"]["macro_f1"]
    
    print(f"Random Row-wise Results: Acc(0.5)={acc_rand_05:.4f} | Acc(opt)={acc_rand_opt:.4f} | MacroF1(opt)={f1_rand_opt:.4f}")
    
    # 3. Calculate Leakage Gap
    delta_leak_05 = acc_rand_05 - acc_loso_05
    delta_leak_opt = acc_rand_opt - acc_loso_opt
    delta_f1 = f1_rand_opt - f1_loso_opt
    
    print("\n=======================================================")
    print("LEAKAGE GAP TEST RESULTS (Random Forest on view_physio_all):")
    print(f"  LOSO Accuracy (default 0.5):        {acc_loso_05:.4f}")
    print(f"  Random K-Fold Accuracy (default):   {acc_rand_05:.4f}")
    print(f"  Delta Leakage (default 0.5):        {delta_leak_05:+.4f} ({delta_leak_05*100:+.2f}%)")
    print(f"  LOSO Accuracy (optimal):            {acc_loso_opt:.4f}")
    print(f"  Random K-Fold Accuracy (optimal):   {acc_rand_opt:.4f}")
    print(f"  Delta Leakage (optimal thresh):     {delta_leak_opt:+.4f} ({delta_leak_opt*100:+.2f}%)")
    print(f"  Delta Macro F1 (optimal):           {delta_f1:+.4f} ({delta_f1*100:+.2f}%)")
    
    grl_recommended = bool(delta_leak_05 >= 0.05 or delta_leak_opt >= 0.05)
    print(f"  Recommendation: {'GRL INDICATED (Delta >= 5%)' if grl_recommended else 'NO GRL NEEDED (Delta < 5% or negative)'}")
    print("=======================================================\n")
    
    # 4. Save report
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "leakage_gap_report.json"
    
    results_payload = {
        "model": "RandomForestClassifier",
        "view": "view_physio_all",
        "sample_count": len(df),
        "subject_count": int(df["subject_id"].nunique()),
        "loso": {
            "accuracy_default": acc_loso_05,
            "accuracy_optimal": acc_loso_opt,
            "optimal_threshold": th_loso["optimal_threshold"],
            "macro_f1_optimal": f1_loso_opt
        },
        "random_kfold": {
            "accuracy_default": acc_rand_05,
            "accuracy_optimal": acc_rand_opt,
            "optimal_threshold": th_random["optimal_threshold"],
            "macro_f1_optimal": f1_rand_opt
        },
        "delta_leak_default": delta_leak_05,
        "delta_leak_optimal": delta_leak_opt,
        "delta_macro_f1": delta_f1,
        "grl_recommended": grl_recommended,
        "decision_threshold_cutoff": 0.05
    }
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
        
    print(f"Report written to: {report_file}")
    
    # 5. Plot comparison
    plot_dir = Path("docs/verification")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plot_dir / "leakage_gap_comparison.png"
    
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_names = ["Accuracy (0.5)", "Accuracy (Opt)", "Macro F1 (Opt)"]
    loso_vals = [acc_loso_05, acc_loso_opt, f1_loso_opt]
    rand_vals = [acc_rand_05, acc_rand_opt, f1_rand_opt]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax.bar(x - width/2, loso_vals, width, label="Subject-Grouped (LOSO)", color="#2b5c8f")
    ax.bar(x + width/2, rand_vals, width, label="Random Row-Wise (Leaky)", color="#e27c38")
    
    ax.set_ylabel("Score")
    ax.set_title(f"Identity Leakage-Gap Test (RF on view_physio_all)\nDelta_leak = {delta_leak_05*100:+.2f}% (GRL {'Recommended' if grl_recommended else 'Not Recommended'})")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim([0.4, 0.85])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for i in range(len(metrics_names)):
        ax.text(x[i] - width/2, loso_vals[i] + 0.01, f"{loso_vals[i]:.3f}", ha="center", fontsize=9)
        ax.text(x[i] + width/2, rand_vals[i] + 0.01, f"{rand_vals[i]:.3f}", ha="center", fontsize=9)
        
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Comparison plot written to: {plot_file}")

if __name__ == "__main__":
    run_leakage_gap_test()
