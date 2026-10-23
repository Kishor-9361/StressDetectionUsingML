# Model Card: gru_view_physio_all

## 1. Overview & Provenance
- **Model Architecture:** gru_view_physio_all
- **Training View:** `view_physio_all`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_140910
- **Run Hash:** `f35d593d`
- **Subjects Evaluated:** 64 subjects
- **Total Valid Windows:** 1394
- **Input Feature Count:** 132 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6019 | 0.5947 | 0.7200 (RF Table 2) | -0.1253 |
| **Balanced Accuracy** | 0.5697 | 0.5817 | - | - |
| **Macro F1** | 0.5701 | **0.5775** | **0.7300** (RF Table 2) | **-0.1525** |
| **Stress Class F1** | 0.4532 | 0.4924 | - | - |
| **ROC AUC** | 0.5969 | 0.5969 | - | - |
| **PR AUC** | 0.4407 | 0.4407 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.10 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0074 (0.5701 -> 0.5775)
- **Confusion Matrix (Default 0.5):** TN=609, FP=270, FN=285, TP=230
- **Confusion Matrix (Optimal 0.10):** TN=555, FP=324, FN=241, TP=274

## 4. Hyperparameters
```json
{
  "model_type": "gru",
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
Tier 1/2 Deep baseline (gru, GRL=False) on view_physio_all across 64 subjects.
