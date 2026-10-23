# Model Card: knn_view_face_voice

## 1. Overview & Provenance
- **Model Architecture:** knn_view_face_voice
- **Training View:** `view_face_voice`
- **Evaluation Protocol:** Subject-Grouped LOSO (53 folds)
- **Run Timestamp:** 20261023_150257
- **Run Hash:** `18f005c6`
- **Subjects Evaluated:** 53 subjects
- **Total Valid Windows:** 371
- **Input Feature Count:** 224 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6199 | 0.6199 | 0.7200 (RF Table 2) | -0.1001 |
| **Balanced Accuracy** | 0.5187 | 0.5187 | - | - |
| **Macro F1** | 0.5159 | **0.5159** | **0.7300** (RF Table 2) | **-0.2141** |
| **Stress Class F1** | 0.7403 | 0.7403 | - | - |
| **ROC AUC** | 0.5400 | 0.5400 | - | - |
| **PR AUC** | 0.7218 | 0.7218 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.45 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0000 (0.5159 -> 0.5159)
- **Confusion Matrix (Default 0.5):** TN=29, FP=86, FN=55, TP=201
- **Confusion Matrix (Optimal 0.45):** TN=29, FP=86, FN=55, TP=201

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
Tier 0 classical model on view_face_voice.
