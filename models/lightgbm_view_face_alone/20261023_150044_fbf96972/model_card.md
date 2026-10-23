# Model Card: lightgbm_view_face_alone

## 1. Overview & Provenance
- **Model Architecture:** lightgbm_view_face_alone
- **Training View:** `view_face_alone`
- **Evaluation Protocol:** Subject-Grouped LOSO (55 folds)
- **Run Timestamp:** 20261023_150044
- **Run Hash:** `fbf96972`
- **Subjects Evaluated:** 55 subjects
- **Total Valid Windows:** 600
- **Input Feature Count:** 84 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6067 | 0.6167 | 0.7200 (RF Table 2) | -0.1033 |
| **Balanced Accuracy** | 0.6048 | 0.6113 | - | - |
| **Macro F1** | 0.6049 | **0.6101** | **0.7300** (RF Table 2) | **-0.1199** |
| **Stress Class F1** | 0.6312 | 0.6608 | - | - |
| **ROC AUC** | 0.6291 | 0.6291 | - | - |
| **PR AUC** | 0.5954 | 0.5954 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.45 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0052 (0.6049 -> 0.6101)
- **Confusion Matrix (Default 0.5):** TN=162, FP=121, FN=115, TP=202
- **Confusion Matrix (Optimal 0.45):** TN=146, FP=137, FN=93, TP=224

## 4. Hyperparameters
```json
{
  "model_type": "lightgbm",
  "params": "{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.05, 'max_depth': 4, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': -1, 'num_leaves': 31, 'objective': None, 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'scale_pos_weight': 0.8927444794952681, 'verbose': -1}"
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
