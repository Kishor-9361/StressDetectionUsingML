# Model Card: lightgbm_view_physio_all

## 1. Overview & Provenance
- **Model Architecture:** lightgbm_view_physio_all
- **Training View:** `view_physio_all`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_133648
- **Run Hash:** `870149b3`
- **Subjects Evaluated:** 64 subjects
- **Total Valid Windows:** 1394
- **Input Feature Count:** 132 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6499 | 0.6664 | 0.7200 (RF Table 2) | -0.0536 |
| **Balanced Accuracy** | 0.6102 | 0.6157 | - | - |
| **Macro F1** | 0.6124 | **0.6183** | **0.7300** (RF Table 2) | **-0.1117** |
| **Stress Class F1** | 0.4917 | 0.4828 | - | - |
| **ROC AUC** | 0.6097 | 0.6097 | - | - |
| **PR AUC** | 0.4779 | 0.4779 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.55 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0059 (0.6124 -> 0.6183)
- **Confusion Matrix (Default 0.5):** TN=670, FP=209, FN=279, TP=236
- **Confusion Matrix (Optimal 0.55):** TN=712, FP=167, FN=298, TP=217

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
