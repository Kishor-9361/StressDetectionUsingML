# Model Card: svm_view_face_alone

## 1. Overview & Provenance
- **Model Architecture:** svm_view_face_alone
- **Training View:** `view_face_alone`
- **Evaluation Protocol:** Subject-Grouped LOSO (55 folds)
- **Run Timestamp:** 20261023_150032
- **Run Hash:** `115b5274`
- **Subjects Evaluated:** 55 subjects
- **Total Valid Windows:** 600
- **Input Feature Count:** 84 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.5983 | 0.6217 | 0.7200 (RF Table 2) | -0.0983 |
| **Balanced Accuracy** | 0.5943 | 0.6099 | - | - |
| **Macro F1** | 0.5939 | **0.5982** | **0.7300** (RF Table 2) | **-0.1318** |
| **Stress Class F1** | 0.6365 | 0.6953 | - | - |
| **ROC AUC** | 0.6467 | 0.6467 | - | - |
| **PR AUC** | 0.6425 | 0.6425 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.40 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0043 (0.5939 -> 0.5982)
- **Confusion Matrix (Default 0.5):** TN=148, FP=135, FN=106, TP=211
- **Confusion Matrix (Optimal 0.40):** TN=114, FP=169, FN=58, TP=259

## 4. Hyperparameters
```json
{
  "model_type": "svm",
  "params": "{'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': 42, 'shrinking': True, 'tol': 0.001, 'verbose': False}"
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
