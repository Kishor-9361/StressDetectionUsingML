# Model Card: rf_view_physio_voice

## 1. Overview & Provenance
- **Model Architecture:** rf_view_physio_voice
- **Training View:** `view_physio_voice`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_133801
- **Run Hash:** `f3148148`
- **Subjects Evaluated:** 53 subjects
- **Total Valid Windows:** 371
- **Input Feature Count:** 272 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6981 | 0.6253 | 0.7200 (RF Table 2) | -0.0947 |
| **Balanced Accuracy** | 0.5006 | 0.5366 | - | - |
| **Macro F1** | 0.4357 | **0.5371** | **0.7300** (RF Table 2) | **-0.1929** |
| **Stress Class F1** | 0.8205 | 0.7392 | - | - |
| **ROC AUC** | 0.5441 | 0.5441 | - | - |
| **PR AUC** | 0.7263 | 0.7263 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.65 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.1014 (0.4357 -> 0.5371)
- **Confusion Matrix (Default 0.5):** TN=3, FP=105, FN=7, TP=256
- **Confusion Matrix (Optimal 0.65):** TN=35, FP=73, FN=66, TP=197

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
Tier 0 baseline trained on view_physio_voice with 272 features across 53 subjects.
