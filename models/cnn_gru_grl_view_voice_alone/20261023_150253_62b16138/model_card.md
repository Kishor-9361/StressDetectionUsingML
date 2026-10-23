# Model Card: cnn_gru_grl_view_voice_alone

## 1. Overview & Provenance
- **Model Architecture:** cnn_gru_grl_view_voice_alone
- **Training View:** `view_voice_alone`
- **Evaluation Protocol:** Subject-Grouped LOSO (54 folds)
- **Run Timestamp:** 20261023_150253
- **Run Hash:** `62b16138`
- **Subjects Evaluated:** 54 subjects
- **Total Valid Windows:** 378
- **Input Feature Count:** 140 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6243 | 0.6032 | 0.7200 (RF Table 2) | -0.1168 |
| **Balanced Accuracy** | 0.5074 | 0.5313 | - | - |
| **Macro F1** | 0.5009 | **0.5313** | **0.7300** (RF Table 2) | **-0.1987** |
| **Stress Class F1** | 0.7491 | 0.7148 | - | - |
| **ROC AUC** | 0.5418 | 0.5418 | - | - |
| **PR AUC** | 0.7365 | 0.7365 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.70 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0305 (0.5009 -> 0.5313)
- **Confusion Matrix (Default 0.5):** TN=24, FP=91, FN=51, TP=212
- **Confusion Matrix (Optimal 0.70):** TN=40, FP=75, FN=75, TP=188

## 4. Hyperparameters
```json
{
  "model_type": "cnn_gru_grl",
  "epochs": 15,
  "lr": 0.001,
  "lambda_adv": 0.02
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
Tier 1/2 Deep model on view_voice_alone.
