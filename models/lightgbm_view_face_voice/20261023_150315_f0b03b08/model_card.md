# Model Card: lightgbm_view_face_voice

## 1. Overview & Provenance
- **Model Architecture:** lightgbm_view_face_voice
- **Training View:** `view_face_voice`
- **Evaluation Protocol:** Subject-Grouped LOSO (53 folds)
- **Run Timestamp:** 20261023_150315
- **Run Hash:** `f0b03b08`
- **Subjects Evaluated:** 53 subjects
- **Total Valid Windows:** 371
- **Input Feature Count:** 224 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6092 | 0.5633 | 0.7200 (RF Table 2) | -0.1567 |
| **Balanced Accuracy** | 0.5156 | 0.5399 | - | - |
| **Macro F1** | 0.5142 | **0.5299** | **0.7300** (RF Table 2) | **-0.2001** |
| **Stress Class F1** | 0.7290 | 0.6553 | - | - |
| **ROC AUC** | 0.5568 | 0.5568 | - | - |
| **PR AUC** | 0.7304 | 0.7304 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.65 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0156 (0.5142 -> 0.5299)
- **Confusion Matrix (Default 0.5):** TN=31, FP=84, FN=61, TP=195
- **Confusion Matrix (Optimal 0.65):** TN=55, FP=60, FN=102, TP=154

## 4. Hyperparameters
```json
{
  "model_type": "lightgbm",
  "params": "{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.05, 'max_depth': 4, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': -1, 'num_leaves': 31, 'objective': None, 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'scale_pos_weight': 0.44921875, 'verbose': -1}"
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
