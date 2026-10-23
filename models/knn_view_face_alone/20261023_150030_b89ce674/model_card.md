# Model Card: knn_view_face_alone

## 1. Overview & Provenance
- **Model Architecture:** knn_view_face_alone
- **Training View:** `view_face_alone`
- **Evaluation Protocol:** Subject-Grouped LOSO (55 folds)
- **Run Timestamp:** 20261023_150030
- **Run Hash:** `b89ce674`
- **Subjects Evaluated:** 55 subjects
- **Total Valid Windows:** 600
- **Input Feature Count:** 84 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6117 | 0.6133 | 0.7200 (RF Table 2) | -0.1067 |
| **Balanced Accuracy** | 0.6137 | 0.6155 | - | - |
| **Macro F1** | 0.6117 | **0.6133** | **0.7300** (RF Table 2) | **-0.1167** |
| **Stress Class F1** | 0.6110 | 0.6120 | - | - |
| **ROC AUC** | 0.6298 | 0.6298 | - | - |
| **PR AUC** | 0.6390 | 0.6390 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.55 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0017 (0.6117 -> 0.6133)
- **Confusion Matrix (Default 0.5):** TN=184, FP=99, FN=134, TP=183
- **Confusion Matrix (Optimal 0.55):** TN=185, FP=98, FN=134, TP=183

## 4. Hyperparameters
```json
{
  "model_type": "knn",
  "params": "{'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'distance'}"
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
Tier 0 classical model on view_face_alone.
