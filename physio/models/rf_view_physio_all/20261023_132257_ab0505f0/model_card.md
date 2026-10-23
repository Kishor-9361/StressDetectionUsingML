# Model Card: rf_view_physio_all

## 1. Overview & Provenance
- **Model Architecture:** rf_view_physio_all
- **Training View:** `view_physio_all`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_132257
- **Run Hash:** `ab0505f0`
- **Subjects Evaluated:** 64 subjects
- **Total Valid Windows:** 1394
- **Input Feature Count:** 132 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6793 | 0.6736 | 0.7200 (RF Table 2) | -0.0464 |
| **Balanced Accuracy** | 0.6159 | 0.6270 | - | - |
| **Macro F1** | 0.6168 | **0.6303** | **0.7300** (RF Table 2) | **-0.0997** |
| **Stress Class F1** | 0.4621 | 0.5038 | - | - |
| **ROC AUC** | 0.6517 | 0.6517 | - | - |
| **PR AUC** | 0.5083 | 0.5083 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.45 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0135 (0.6168 -> 0.6303)
- **Confusion Matrix (Default 0.5):** TN=755, FP=124, FN=323, TP=192
- **Confusion Matrix (Optimal 0.45):** TN=708, FP=171, FN=284, TP=231

## 4. Hyperparameters
```json
{
  "n_estimators": 100,
  "max_depth": 12
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
