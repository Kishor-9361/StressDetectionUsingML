# Model Card: xgboost_view_tri_modal

## 1. Overview & Provenance
- **Model Architecture:** xgboost_view_tri_modal
- **Training View:** `view_tri_modal`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_133908
- **Run Hash:** `98ecb92e`
- **Subjects Evaluated:** 50 subjects
- **Total Valid Windows:** 350
- **Input Feature Count:** 356 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6714 | 0.6057 | 0.7200 (RF Table 2) | -0.1143 |
| **Balanced Accuracy** | 0.4925 | 0.5434 | - | - |
| **Macro F1** | 0.4536 | **0.5403** | **0.7300** (RF Table 2) | **-0.1897** |
| **Stress Class F1** | 0.7986 | 0.7137 | - | - |
| **ROC AUC** | 0.5699 | 0.5699 | - | - |
| **PR AUC** | 0.7844 | 0.7844 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.85 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0868 (0.4536 -> 0.5403)
- **Confusion Matrix (Default 0.5):** TN=7, FP=94, FN=21, TP=228
- **Confusion Matrix (Optimal 0.85):** TN=40, FP=61, FN=77, TP=172

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
Tier 0 baseline trained on view_tri_modal with 356 features across 50 subjects.
