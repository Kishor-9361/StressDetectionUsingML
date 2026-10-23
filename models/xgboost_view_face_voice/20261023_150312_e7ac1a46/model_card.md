# Model Card: xgboost_view_face_voice

## 1. Overview & Provenance
- **Model Architecture:** xgboost_view_face_voice
- **Training View:** `view_face_voice`
- **Evaluation Protocol:** Subject-Grouped LOSO (53 folds)
- **Run Timestamp:** 20261023_150312
- **Run Hash:** `e7ac1a46`
- **Subjects Evaluated:** 53 subjects
- **Total Valid Windows:** 371
- **Input Feature Count:** 224 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6038 | 0.5606 | 0.7200 (RF Table 2) | -0.1594 |
| **Balanced Accuracy** | 0.4926 | 0.5427 | - | - |
| **Macro F1** | 0.4853 | **0.5305** | **0.7300** (RF Table 2) | **-0.1995** |
| **Stress Class F1** | 0.7322 | 0.6495 | - | - |
| **ROC AUC** | 0.5497 | 0.5497 | - | - |
| **PR AUC** | 0.7199 | 0.7199 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.65 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0452 (0.4853 -> 0.5305)
- **Confusion Matrix (Default 0.5):** TN=23, FP=92, FN=55, TP=201
- **Confusion Matrix (Optimal 0.65):** TN=57, FP=58, FN=105, TP=151

## 4. Hyperparameters
```json
{
  "model_type": "xgboost",
  "params": "{'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': None, 'device': None, 'early_stopping_rounds': None, 'enable_categorical': True, 'eval_metric': 'logloss', 'feature_types': None, 'feature_weights': None, 'gamma': None, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': 0.05, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 4, 'max_leaves': None, 'min_child_weight': None, 'missing': nan, 'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': 100, 'n_jobs': -1, 'num_parallel_tree': None, 'random_state': 42, 'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': 0.44921875, 'subsample': None, 'tree_method': None, 'validate_parameters': None, 'verbosity': None}"
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
