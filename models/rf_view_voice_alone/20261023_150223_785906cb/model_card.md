# Model Card: rf_view_voice_alone

## 1. Overview & Provenance
- **Model Architecture:** rf_view_voice_alone
- **Training View:** `view_voice_alone`
- **Evaluation Protocol:** Subject-Grouped LOSO (54 folds)
- **Run Timestamp:** 20261023_150223
- **Run Hash:** `785906cb`
- **Subjects Evaluated:** 54 subjects
- **Total Valid Windows:** 378
- **Input Feature Count:** 140 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6323 | 0.4841 | 0.7200 (RF Table 2) | -0.2359 |
| **Balanced Accuracy** | 0.4666 | 0.4996 | - | - |
| **Macro F1** | 0.4191 | **0.4712** | **0.7300** (RF Table 2) | **-0.2588** |
| **Stress Class F1** | 0.7710 | 0.5538 | - | - |
| **ROC AUC** | 0.4795 | 0.4795 | - | - |
| **PR AUC** | 0.7121 | 0.7121 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.65 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0522 (0.4191 -> 0.4712)
- **Confusion Matrix (Default 0.5):** TN=5, FP=110, FN=29, TP=234
- **Confusion Matrix (Optimal 0.65):** TN=62, FP=53, FN=142, TP=121

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
Tier 0 classical model on view_voice_alone.
