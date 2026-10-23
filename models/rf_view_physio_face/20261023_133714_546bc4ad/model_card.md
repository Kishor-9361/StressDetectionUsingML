# Model Card: rf_view_physio_face

## 1. Overview & Provenance
- **Model Architecture:** rf_view_physio_face
- **Training View:** `view_physio_face`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_133714
- **Run Hash:** `546bc4ad`
- **Subjects Evaluated:** 52 subjects
- **Total Valid Windows:** 1138
- **Input Feature Count:** 216 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6898 | 0.6889 | 0.7200 (RF Table 2) | -0.0311 |
| **Balanced Accuracy** | 0.6539 | 0.6701 | - | - |
| **Macro F1** | 0.6576 | **0.6698** | **0.7300** (RF Table 2) | **-0.0602** |
| **Stress Class F1** | 0.5526 | 0.5903 | - | - |
| **ROC AUC** | 0.7022 | 0.7022 | - | - |
| **PR AUC** | 0.5701 | 0.5701 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.45 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0122 (0.6576 -> 0.6698)
- **Confusion Matrix (Default 0.5):** TN=567, FP=141, FN=212, TP=218
- **Confusion Matrix (Optimal 0.45):** TN=529, FP=179, FN=175, TP=255

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
Tier 0 baseline trained on view_physio_face with 216 features across 52 subjects.
