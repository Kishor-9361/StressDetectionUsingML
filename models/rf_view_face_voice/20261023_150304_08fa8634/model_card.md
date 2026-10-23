# Model Card: rf_view_face_voice

## 1. Overview & Provenance
- **Model Architecture:** rf_view_face_voice
- **Training View:** `view_face_voice`
- **Evaluation Protocol:** Subject-Grouped LOSO (53 folds)
- **Run Timestamp:** 20261023_150304
- **Run Hash:** `08fa8634`
- **Subjects Evaluated:** 53 subjects
- **Total Valid Windows:** 371
- **Input Feature Count:** 224 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6739 | 0.6173 | 0.7200 (RF Table 2) | -0.1027 |
| **Balanced Accuracy** | 0.5170 | 0.5407 | - | - |
| **Macro F1** | 0.4814 | **0.5413** | **0.7300** (RF Table 2) | **-0.1887** |
| **Stress Class F1** | 0.7973 | 0.7280 | - | - |
| **ROC AUC** | 0.5549 | 0.5549 | - | - |
| **PR AUC** | 0.7318 | 0.7318 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.60 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0598 (0.4814 -> 0.5413)
- **Confusion Matrix (Default 0.5):** TN=12, FP=103, FN=18, TP=238
- **Confusion Matrix (Optimal 0.60):** TN=39, FP=76, FN=66, TP=190

## 4. Hyperparameters
```json
{
  "model_type": "rf",
  "params": "{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 6, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}"
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
