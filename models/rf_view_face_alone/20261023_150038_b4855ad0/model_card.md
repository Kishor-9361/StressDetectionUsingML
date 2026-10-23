# Model Card: rf_view_face_alone

## 1. Overview & Provenance
- **Model Architecture:** rf_view_face_alone
- **Training View:** `view_face_alone`
- **Evaluation Protocol:** Subject-Grouped LOSO (55 folds)
- **Run Timestamp:** 20261023_150038
- **Run Hash:** `b4855ad0`
- **Subjects Evaluated:** 55 subjects
- **Total Valid Windows:** 600
- **Input Feature Count:** 84 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6083 | 0.6083 | 0.7200 (RF Table 2) | -0.1117 |
| **Balanced Accuracy** | 0.6030 | 0.6030 | - | - |
| **Macro F1** | 0.6018 | **0.6018** | **0.7300** (RF Table 2) | **-0.1282** |
| **Stress Class F1** | 0.6529 | 0.6529 | - | - |
| **ROC AUC** | 0.6532 | 0.6532 | - | - |
| **PR AUC** | 0.6413 | 0.6413 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.50 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0000 (0.6018 -> 0.6018)
- **Confusion Matrix (Default 0.5):** TN=144, FP=139, FN=96, TP=221
- **Confusion Matrix (Optimal 0.50):** TN=144, FP=139, FN=96, TP=221

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
Tier 0 classical model on view_face_alone.
