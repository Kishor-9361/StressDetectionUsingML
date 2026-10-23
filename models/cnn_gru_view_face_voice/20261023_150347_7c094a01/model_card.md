# Model Card: cnn_gru_view_face_voice

## 1. Overview & Provenance
- **Model Architecture:** cnn_gru_view_face_voice
- **Training View:** `view_face_voice`
- **Evaluation Protocol:** Subject-Grouped LOSO (53 folds)
- **Run Timestamp:** 20261023_150347
- **Run Hash:** `7c094a01`
- **Subjects Evaluated:** 53 subjects
- **Total Valid Windows:** 371
- **Input Feature Count:** 224 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6226 | 0.6280 | 0.7200 (RF Table 2) | -0.0920 |
| **Balanced Accuracy** | 0.5446 | 0.5676 | - | - |
| **Macro F1** | 0.5453 | **0.5673** | **0.7300** (RF Table 2) | **-0.1627** |
| **Stress Class F1** | 0.7328 | 0.7294 | - | - |
| **ROC AUC** | 0.5421 | 0.5421 | - | - |
| **PR AUC** | 0.7302 | 0.7302 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.65 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0220 (0.5453 -> 0.5673)
- **Confusion Matrix (Default 0.5):** TN=39, FP=76, FN=64, TP=192
- **Confusion Matrix (Optimal 0.65):** TN=47, FP=68, FN=70, TP=186

## 4. Hyperparameters
```json
{
  "model_type": "cnn_gru",
  "epochs": 15,
  "lr": 0.001,
  "lambda_adv": null
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
