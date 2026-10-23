# Model Card: cnn_gru_grl_view_face_voice

## 1. Overview & Provenance
- **Model Architecture:** cnn_gru_grl_view_face_voice
- **Training View:** `view_face_voice`
- **Evaluation Protocol:** Subject-Grouped LOSO (53 folds)
- **Run Timestamp:** 20261023_150411
- **Run Hash:** `5a4285da`
- **Subjects Evaluated:** 53 subjects
- **Total Valid Windows:** 371
- **Input Feature Count:** 224 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6442 | 0.6442 | 0.7200 (RF Table 2) | -0.0758 |
| **Balanced Accuracy** | 0.5506 | 0.5506 | - | - |
| **Macro F1** | 0.5510 | **0.5510** | **0.7300** (RF Table 2) | **-0.1790** |
| **Stress Class F1** | 0.7556 | 0.7556 | - | - |
| **ROC AUC** | 0.5629 | 0.5629 | - | - |
| **PR AUC** | 0.7401 | 0.7401 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.50 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0000 (0.5510 -> 0.5510)
- **Confusion Matrix (Default 0.5):** TN=35, FP=80, FN=52, TP=204
- **Confusion Matrix (Optimal 0.50):** TN=35, FP=80, FN=52, TP=204

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
Tier 1/2 Deep model on view_face_voice.
