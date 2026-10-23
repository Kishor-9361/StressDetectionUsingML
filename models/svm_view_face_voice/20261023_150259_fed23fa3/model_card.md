# Model Card: svm_view_face_voice

## 1. Overview & Provenance
- **Model Architecture:** svm_view_face_voice
- **Training View:** `view_face_voice`
- **Evaluation Protocol:** Subject-Grouped LOSO (53 folds)
- **Run Timestamp:** 20261023_150259
- **Run Hash:** `fed23fa3`
- **Subjects Evaluated:** 53 subjects
- **Total Valid Windows:** 371
- **Input Feature Count:** 224 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6523 | 0.5418 | 0.7200 (RF Table 2) | -0.1782 |
| **Balanced Accuracy** | 0.4894 | 0.5267 | - | - |
| **Macro F1** | 0.4413 | **0.5138** | **0.7300** (RF Table 2) | **-0.2162** |
| **Stress Class F1** | 0.7846 | 0.6304 | - | - |
| **ROC AUC** | 0.5097 | 0.5097 | - | - |
| **PR AUC** | 0.6964 | 0.6964 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.70 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0725 (0.4413 -> 0.5138)
- **Confusion Matrix (Default 0.5):** TN=7, FP=108, FN=21, TP=235
- **Confusion Matrix (Optimal 0.70):** TN=56, FP=59, FN=111, TP=145

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
Tier 0 classical model on view_face_voice.
