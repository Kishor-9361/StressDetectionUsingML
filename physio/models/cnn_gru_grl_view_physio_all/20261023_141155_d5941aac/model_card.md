# Model Card: cnn_gru_grl_view_physio_all

## 1. Overview & Provenance
- **Model Architecture:** cnn_gru_grl_view_physio_all
- **Training View:** `view_physio_all`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_141155
- **Run Hash:** `d5941aac`
- **Subjects Evaluated:** 64 subjects
- **Total Valid Windows:** 1394
- **Input Feature Count:** 132 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6255 | 0.6255 | 0.7200 (RF Table 2) | -0.0945 |
| **Balanced Accuracy** | 0.5957 | 0.5957 | - | - |
| **Macro F1** | 0.5962 | **0.5962** | **0.7300** (RF Table 2) | **-0.1338** |
| **Stress Class F1** | 0.4872 | 0.4872 | - | - |
| **ROC AUC** | 0.6185 | 0.6185 | - | - |
| **PR AUC** | 0.4522 | 0.4522 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.50 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0000 (0.5962 -> 0.5962)
- **Confusion Matrix (Default 0.5):** TN=624, FP=255, FN=267, TP=248
- **Confusion Matrix (Optimal 0.50):** TN=624, FP=255, FN=267, TP=248

## 4. Hyperparameters
```json
{
  "model_type": "cnn_gru",
  "hidden_dim": 64,
  "use_grl": true,
  "lambda_adv": 0.02,
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
Tier 1/2 Deep baseline (cnn_gru, GRL=True) on view_physio_all across 64 subjects.
