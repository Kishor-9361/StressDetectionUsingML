# Model Card: cnn_gru_view_physio_all

## 1. Overview & Provenance
- **Model Architecture:** cnn_gru_view_physio_all
- **Training View:** `view_physio_all`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_141017
- **Run Hash:** `5e4e82fa`
- **Subjects Evaluated:** 64 subjects
- **Total Valid Windows:** 1394
- **Input Feature Count:** 132 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6227 | 0.6126 | 0.7200 (RF Table 2) | -0.1074 |
| **Balanced Accuracy** | 0.5922 | 0.5963 | - | - |
| **Macro F1** | 0.5927 | **0.5933** | **0.7300** (RF Table 2) | **-0.1367** |
| **Stress Class F1** | 0.4823 | 0.5046 | - | - |
| **ROC AUC** | 0.6045 | 0.6045 | - | - |
| **PR AUC** | 0.4374 | 0.4374 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.35 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0006 (0.5927 -> 0.5933)
- **Confusion Matrix (Default 0.5):** TN=623, FP=256, FN=270, TP=245
- **Confusion Matrix (Optimal 0.35):** TN=579, FP=300, FN=240, TP=275

## 4. Hyperparameters
```json
{
  "model_type": "cnn_gru",
  "hidden_dim": 64,
  "use_grl": false,
  "lambda_adv": 0.0,
  "epochs": 25,
  "optimizer": "AdamW",
  "lr": 0.002
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
Tier 1/2 Deep baseline (cnn_gru, GRL=False) on view_physio_all across 64 subjects.
