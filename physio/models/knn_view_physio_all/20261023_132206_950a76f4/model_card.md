# Model Card: knn_view_physio_all

## 1. Overview & Provenance
- **Model Architecture:** knn_view_physio_all
- **Training View:** `view_physio_all`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_132206
- **Run Hash:** `950a76f4`
- **Subjects Evaluated:** 64 subjects
- **Total Valid Windows:** 1394
- **Input Feature Count:** 132 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6420 | 0.5961 | 0.7200 (RF Table 2) | -0.1239 |
| **Balanced Accuracy** | 0.5553 | 0.6014 | - | - |
| **Macro F1** | 0.5366 | **0.5884** | **0.7300** (RF Table 2) | **-0.1416** |
| **Stress Class F1** | 0.3155 | 0.5320 | - | - |
| **ROC AUC** | 0.6302 | 0.6302 | - | - |
| **PR AUC** | 0.4796 | 0.4796 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.25 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0518 (0.5366 -> 0.5884)
- **Confusion Matrix (Default 0.5):** TN=780, FP=99, FN=400, TP=115
- **Confusion Matrix (Optimal 0.25):** TN=511, FP=368, FN=195, TP=320

## 4. Hyperparameters
```json
{
  "n_neighbors": 11,
  "weights": "distance"
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
