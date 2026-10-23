# Model Card: cnn_gru_grl_view_face_alone

## 1. Overview & Provenance
- **Model Architecture:** cnn_gru_grl_view_face_alone
- **Training View:** `view_face_alone`
- **Evaluation Protocol:** Subject-Grouped LOSO (55 folds)
- **Run Timestamp:** 20261023_150217
- **Run Hash:** `2e2b63d1`
- **Subjects Evaluated:** 55 subjects
- **Total Valid Windows:** 600
- **Input Feature Count:** 84 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.5450 | 0.5550 | 0.7200 (RF Table 2) | -0.1650 |
| **Balanced Accuracy** | 0.5423 | 0.5501 | - | - |
| **Macro F1** | 0.5422 | **0.5487** | **0.7300** (RF Table 2) | **-0.1813** |
| **Stress Class F1** | 0.5781 | 0.6021 | - | - |
| **ROC AUC** | 0.5553 | 0.5553 | - | - |
| **PR AUC** | 0.5787 | 0.5787 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.40 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0065 (0.5422 -> 0.5487)
- **Confusion Matrix (Default 0.5):** TN=140, FP=143, FN=130, TP=187
- **Confusion Matrix (Optimal 0.40):** TN=131, FP=152, FN=115, TP=202

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
Tier 1/2 Deep model on view_face_alone.
