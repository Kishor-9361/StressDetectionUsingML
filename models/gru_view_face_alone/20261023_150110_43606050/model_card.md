# Model Card: gru_view_face_alone

## 1. Overview & Provenance
- **Model Architecture:** gru_view_face_alone
- **Training View:** `view_face_alone`
- **Evaluation Protocol:** Subject-Grouped LOSO (55 folds)
- **Run Timestamp:** 20261023_150110
- **Run Hash:** `43606050`
- **Subjects Evaluated:** 55 subjects
- **Total Valid Windows:** 600
- **Input Feature Count:** 84 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.5933 | 0.5917 | 0.7200 (RF Table 2) | -0.1283 |
| **Balanced Accuracy** | 0.5901 | 0.5916 | - | - |
| **Macro F1** | 0.5900 | **0.5912** | **0.7300** (RF Table 2) | **-0.1388** |
| **Stress Class F1** | 0.6269 | 0.6055 | - | - |
| **ROC AUC** | 0.6267 | 0.6267 | - | - |
| **PR AUC** | 0.6256 | 0.6256 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.55 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0012 (0.5900 -> 0.5912)
- **Confusion Matrix (Default 0.5):** TN=151, FP=132, FN=112, TP=205
- **Confusion Matrix (Optimal 0.55):** TN=167, FP=116, FN=129, TP=188

## 4. Hyperparameters
```json
{
  "model_type": "gru",
  "epochs": 15,
  "lr": 0.001,
  "lambda_adv": null
}
```

## 5. Artifact Checklist
- [x] `metrics.json`
- [x] `predictions.parquet`
- [x] `confusion_matrix.png`
- [x] `roc.png`
- [x] `pr.png`
- [x] `model_card.md`
- [x] `checkpoint.joblib`

## 6. Contextual Analysis
Tier 1/2 Deep model on view_face_alone.
