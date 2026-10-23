# Model Card: xgboost_view_physio_face

## 1. Overview & Provenance
- **Model Architecture:** xgboost_view_physio_face
- **Training View:** `view_physio_face`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_133744
- **Run Hash:** `8633804f`
- **Subjects Evaluated:** 52 subjects
- **Total Valid Windows:** 1138
- **Input Feature Count:** 216 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6696 | 0.6924 | 0.7200 (RF Table 2) | -0.0276 |
| **Balanced Accuracy** | 0.6459 | 0.6524 | - | - |
| **Macro F1** | 0.6466 | **0.6565** | **0.7300** (RF Table 2) | **-0.0735** |
| **Stress Class F1** | 0.5566 | 0.5455 | - | - |
| **ROC AUC** | 0.6850 | 0.6850 | - | - |
| **PR AUC** | 0.5536 | 0.5536 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.60 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0099 (0.6466 -> 0.6565)
- **Confusion Matrix (Default 0.5):** TN=526, FP=182, FN=194, TP=236
- **Confusion Matrix (Optimal 0.60):** TN=578, FP=130, FN=220, TP=210

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
Tier 0 baseline trained on view_physio_face with 216 features across 52 subjects.
