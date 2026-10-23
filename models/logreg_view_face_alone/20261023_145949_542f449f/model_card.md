# Model Card: logreg_view_face_alone

## 1. Overview & Provenance
- **Model Architecture:** logreg_view_face_alone
- **Training View:** `view_face_alone`
- **Evaluation Protocol:** Subject-Grouped LOSO (55 folds)
- **Run Timestamp:** 20261023_145949
- **Run Hash:** `542f449f`
- **Subjects Evaluated:** 55 subjects
- **Total Valid Windows:** 600
- **Input Feature Count:** 84 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6183 | 0.6183 | 0.7200 (RF Table 2) | -0.1017 |
| **Balanced Accuracy** | 0.6180 | 0.6204 | - | - |
| **Macro F1** | 0.6177 | **0.6183** | **0.7300** (RF Table 2) | **-0.1117** |
| **Stress Class F1** | 0.6336 | 0.6177 | - | - |
| **ROC AUC** | 0.6413 | 0.6413 | - | - |
| **PR AUC** | 0.6472 | 0.6472 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.55 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0007 (0.6177 -> 0.6183)
- **Confusion Matrix (Default 0.5):** TN=173, FP=110, FN=119, TP=198
- **Confusion Matrix (Optimal 0.55):** TN=186, FP=97, FN=132, TP=185

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
Tier 0 classical model on view_face_alone.
