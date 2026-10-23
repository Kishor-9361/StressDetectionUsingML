# Model Card: xgboost_view_physio_voice

## 1. Overview & Provenance
- **Model Architecture:** xgboost_view_physio_voice
- **Training View:** `view_physio_voice`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_133824
- **Run Hash:** `1295413a`
- **Subjects Evaluated:** 53 subjects
- **Total Valid Windows:** 371
- **Input Feature Count:** 272 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6927 | 0.6631 | 0.7200 (RF Table 2) | -0.0569 |
| **Balanced Accuracy** | 0.5131 | 0.5577 | - | - |
| **Macro F1** | 0.4747 | **0.5593** | **0.7300** (RF Table 2) | **-0.1707** |
| **Stress Class F1** | 0.8131 | 0.7731 | - | - |
| **ROC AUC** | 0.5607 | 0.5607 | - | - |
| **PR AUC** | 0.7431 | 0.7431 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.75 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0846 (0.4747 -> 0.5593)
- **Confusion Matrix (Default 0.5):** TN=9, FP=99, FN=15, TP=248
- **Confusion Matrix (Optimal 0.75):** TN=33, FP=75, FN=50, TP=213

## 4. Hyperparameters
```json
{
  "max_depth": 4,
  "n_estimators": 200,
  "scale_pos_weight": 1.71
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
Tier 0 baseline trained on view_physio_voice with 272 features across 53 subjects.
