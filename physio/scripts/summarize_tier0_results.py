"""
scripts/summarize_tier0_results.py

Collects metrics across all models in models/ and prints verified Markdown summary tables.
"""

import json
from pathlib import Path

def main():
    models_dir = Path("models")
    rows = []
    for m_dir in sorted(models_dir.iterdir()):
        if not m_dir.is_dir():
            continue
        runs = sorted(m_dir.glob("*/metrics.json"))
        if not runs:
            continue
        latest_run = runs[-1]
        with open(latest_run, "r", encoding="utf-8") as f:
            m = json.load(f)
        def_th = m["default_threshold_0_5"]
        opt_th = m["optimal_threshold"]
        rows.append({
            "model": m["model_name"],
            "view": m["view_name"],
            "acc_05": def_th["accuracy"],
            "macro_f1_05": def_th["macro_f1"],
            "opt_th": opt_th["threshold"],
            "acc_opt": opt_th["accuracy"],
            "macro_f1_opt": opt_th["macro_f1"],
            "stress_f1_opt": opt_th["f1_stress"],
            "bal_acc_opt": opt_th["balanced_accuracy"],
            "roc_auc": m.get("roc_auc", 0.0),
            "run_dir": str(latest_run.parent.name)
        })

    print("| Model | View | Default Macro F1 | Opt Thresh | Opt Acc | Opt Macro F1 | Opt Stress F1 | Opt Bal Acc | ROC AUC |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | `{r['view']}` | {r['macro_f1_05']:.4f} | {r['opt_th']:.2f} | {r['acc_opt']:.4f} | **{r['macro_f1_opt']:.4f}** | {r['stress_f1_opt']:.4f} | {r['bal_acc_opt']:.4f} | {r['roc_auc']:.4f} |")

if __name__ == "__main__":
    main()
