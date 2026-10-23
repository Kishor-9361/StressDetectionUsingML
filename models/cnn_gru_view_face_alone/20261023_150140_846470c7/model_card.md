# Model Card: cnn_gru_view_face_alone

## 1. Overview & Provenance
- **Model Architecture:** cnn_gru_view_face_alone
- **Training View:** `view_face_alone`
- **Evaluation Protocol:** Subject-Grouped LOSO (55 folds)
- **Run Timestamp:** 20261023_150140
- **Run Hash:** `846470c7`
- **Subjects Evaluated:** 55 subjects
- **Total Valid Windows:** 600
- **Input Feature Count:** 84 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.5900 | 0.6050 | 0.7200 (RF Table 2) | -0.1150 |
| **Balanced Accuracy** | 0.5887 | 0.6012 | - | - |
| **Macro F1** | 0.5887 | **0.6009** | **0.7300** (RF Table 2) | **-0.1291** |
| **Stress Class F1** | 0.6120 | 0.6415 | - | - |
| **ROC AUC** | 0.6104 | 0.6104 | - | - |
| **PR AUC** | 0.6172 | 0.6172 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.40 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0122 (0.5887 -> 0.6009)
- **Confusion Matrix (Default 0.5):** TN=160, FP=123, FN=123, TP=194
- **Confusion Matrix (Optimal 0.40):** TN=151, FP=132, FN=105, TP=212

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
Tier 1/2 Deep model on view_face_alone.
