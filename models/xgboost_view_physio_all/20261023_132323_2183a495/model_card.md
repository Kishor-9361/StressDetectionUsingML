# Model Card: xgboost_view_physio_all

## 1. Overview & Provenance
- **Model Architecture:** xgboost_view_physio_all
- **Training View:** `view_physio_all`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_132323
- **Run Hash:** `2183a495`
- **Subjects Evaluated:** 64 subjects
- **Total Valid Windows:** 1394
- **Input Feature Count:** 132 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6585 | 0.6585 | 0.7200 (RF Table 2) | -0.0615 |
| **Balanced Accuracy** | 0.6207 | 0.6207 | - | - |
| **Macro F1** | 0.6230 | **0.6230** | **0.7300** (RF Table 2) | **-0.1070** |
| **Stress Class F1** | 0.5072 | 0.5072 | - | - |
| **ROC AUC** | 0.6149 | 0.6149 | - | - |
| **PR AUC** | 0.4791 | 0.4791 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.50 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0000 (0.6230 -> 0.6230)
- **Confusion Matrix (Default 0.5):** TN=673, FP=206, FN=270, TP=245
- **Confusion Matrix (Optimal 0.50):** TN=673, FP=206, FN=270, TP=245

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
Tier 0 baseline trained on view_physio_all with 132 features across 64 subjects.
