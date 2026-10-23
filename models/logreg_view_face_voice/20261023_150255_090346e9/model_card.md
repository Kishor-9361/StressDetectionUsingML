# Model Card: logreg_view_face_voice

## 1. Overview & Provenance
- **Model Architecture:** logreg_view_face_voice
- **Training View:** `view_face_voice`
- **Evaluation Protocol:** Subject-Grouped LOSO (53 folds)
- **Run Timestamp:** 20261023_150255
- **Run Hash:** `090346e9`
- **Subjects Evaluated:** 53 subjects
- **Total Valid Windows:** 371
- **Input Feature Count:** 224 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.5795 | 0.5660 | 0.7200 (RF Table 2) | -0.1540 |
| **Balanced Accuracy** | 0.5181 | 0.5514 | - | - |
| **Macro F1** | 0.5175 | **0.5376** | **0.7300** (RF Table 2) | **-0.1924** |
| **Stress Class F1** | 0.6905 | 0.6523 | - | - |
| **ROC AUC** | 0.5782 | 0.5782 | - | - |
| **PR AUC** | 0.7748 | 0.7748 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.75 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0201 (0.5175 -> 0.5376)
- **Confusion Matrix (Default 0.5):** TN=41, FP=74, FN=82, TP=174
- **Confusion Matrix (Optimal 0.75):** TN=59, FP=56, FN=105, TP=151

## 4. Hyperparameters
```json
{
  "model_type": "logreg",
  "params": "{'C': 1.0, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 1000, 'multi_class': 'deprecated', 'n_jobs': None, 'penalty': 'l2', 'random_state': 42, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}"
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
Tier 0 classical model on view_face_voice.
