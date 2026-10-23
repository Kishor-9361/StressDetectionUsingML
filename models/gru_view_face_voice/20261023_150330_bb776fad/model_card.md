# Model Card: gru_view_face_voice

## 1. Overview & Provenance
- **Model Architecture:** gru_view_face_voice
- **Training View:** `view_face_voice`
- **Evaluation Protocol:** Subject-Grouped LOSO (53 folds)
- **Run Timestamp:** 20261023_150330
- **Run Hash:** `bb776fad`
- **Subjects Evaluated:** 53 subjects
- **Total Valid Windows:** 371
- **Input Feature Count:** 224 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6226 | 0.6065 | 0.7200 (RF Table 2) | -0.1135 |
| **Balanced Accuracy** | 0.5206 | 0.5616 | - | - |
| **Macro F1** | 0.5178 | **0.5579** | **0.7300** (RF Table 2) | **-0.1721** |
| **Stress Class F1** | 0.7426 | 0.7045 | - | - |
| **ROC AUC** | 0.5408 | 0.5408 | - | - |
| **PR AUC** | 0.7288 | 0.7288 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.75 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0401 (0.5178 -> 0.5579)
- **Confusion Matrix (Default 0.5):** TN=29, FP=86, FN=54, TP=202
- **Confusion Matrix (Optimal 0.75):** TN=51, FP=64, FN=82, TP=174

## 4. Hyperparameters
```json
{
  "model_type": "gru",
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
