# Model Card: svm_view_voice_alone

## 1. Overview & Provenance
- **Model Architecture:** svm_view_voice_alone
- **Training View:** `view_voice_alone`
- **Evaluation Protocol:** Subject-Grouped LOSO (54 folds)
- **Run Timestamp:** 20261023_150230
- **Run Hash:** `d04d48b1`
- **Subjects Evaluated:** 54 subjects
- **Total Valid Windows:** 378
- **Input Feature Count:** 140 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6693 | 0.5476 | 0.7200 (RF Table 2) | -0.1724 |
| **Balanced Accuracy** | 0.5055 | 0.5183 | - | - |
| **Macro F1** | 0.4667 | **0.5098** | **0.7300** (RF Table 2) | **-0.2202** |
| **Stress Class F1** | 0.7954 | 0.6460 | - | - |
| **ROC AUC** | 0.4941 | 0.4941 | - | - |
| **PR AUC** | 0.7001 | 0.7001 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.70 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0431 (0.4667 -> 0.5098)
- **Confusion Matrix (Default 0.5):** TN=10, FP=105, FN=20, TP=243
- **Confusion Matrix (Optimal 0.70):** TN=51, FP=64, FN=107, TP=156

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
Tier 0 classical model on view_voice_alone.
