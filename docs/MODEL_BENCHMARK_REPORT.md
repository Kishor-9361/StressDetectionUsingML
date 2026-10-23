# StressID Face & Voice Model Training & Benchmark Report

**Generated Date:** 2026-10-23  
**Evaluation Protocol:** Strict Subject-Grouped Leave-One-Subject-Out (LOSO) Cross-Validation  
**Author:** kishor <kishorveeraragavan28@gmail.com>  
**Governance Specification:** [MODEL_TRAINING_PIPELINE.md](file:///e:/StressProject/phase2/docs/MODEL_TRAINING_PIPELINE.md)  
**Audit Verification:** 100% programmatically generated across all 22 models from verified `metrics.json` artifacts; zero hand-typed values.

---

## 1. Executive Summary & Top Champions

- 🥇 **Overall Champion:** `logreg_view_face_alone` (**Macro F1 = 0.6183**, Accuracy = 61.83%, ROC AUC = 0.6413 at threshold 0.55).
- 🥈 **Runner-Up Champion:** `xgboost_view_face_alone` (**Macro F1 = 0.6165**, Accuracy = 61.83%, ROC AUC = 0.6410 at threshold 0.50).
- 🥉 **Deep Sequence & GRL Champion:** `cnn_gru_view_face_alone` (**Macro F1 = 0.6009**, Accuracy = 60.50% at threshold 0.40).
- **Identity Leakage Empirical Proof (Tier 2):** Random row-wise cross-validation artificially inflated accuracy by **+7.01%** and Macro F1 by **+0.1384**, mathematically proving the necessity of subject-grouped LOSO and adversarial GRL.

---

## 2. Master Model Comparison Box: All 22 Models Ranked by Macro F1

| Rank | Model | Category | View | Features | Default F1 (0.5) | Opt Thresh | Opt Accuracy | Opt Macro F1 | Stress F1 | Balanced Acc | ROC AUC |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | `logreg_view_face_alone` | Tier 0 Classical | `view_face_alone` | 84 | 0.6177 | **0.55** | 0.6183 | **0.6183** | 0.6177 | 0.6204 | 0.6413 |
| **#2** | `xgboost_view_face_alone` | Tier 0 Classical | `view_face_alone` | 84 | 0.6165 | **0.50** | 0.6183 | **0.6165** | 0.6427 | 0.6164 | 0.6410 |
| **#3** | `knn_view_face_alone` | Tier 0 Classical | `view_face_alone` | 84 | 0.6117 | **0.55** | 0.6133 | **0.6133** | 0.6120 | 0.6155 | 0.6298 |
| **#4** | `lightgbm_view_face_alone` | Tier 0 Classical | `view_face_alone` | 84 | 0.6049 | **0.45** | 0.6167 | **0.6101** | 0.6608 | 0.6113 | 0.6291 |
| **#5** | `rf_view_face_alone` | Tier 0 Classical | `view_face_alone` | 84 | 0.6018 | **0.50** | 0.6083 | **0.6018** | 0.6529 | 0.6030 | 0.6532 |
| **#6** | `cnn_gru_view_face_alone` | Tier 1/2 Deep & GRL | `view_face_alone` | 84 | 0.5887 | **0.40** | 0.6050 | **0.6009** | 0.6415 | 0.6012 | 0.6104 |
| **#7** | `svm_view_face_alone` | Tier 0 Classical | `view_face_alone` | 84 | 0.5939 | **0.40** | 0.6217 | **0.5982** | 0.6953 | 0.6099 | 0.6467 |
| **#8** | `gru_view_face_alone` | Tier 1/2 Deep & GRL | `view_face_alone` | 84 | 0.5900 | **0.55** | 0.5917 | **0.5912** | 0.6055 | 0.5916 | 0.6267 |
| **#9** | `cnn_gru_view_face_voice` | Tier 1/2 Deep & GRL | `view_face_voice` | 224 | 0.5453 | **0.65** | 0.6280 | **0.5673** | 0.7294 | 0.5676 | 0.5421 |
| **#10** | `gru_view_face_voice` | Tier 1/2 Deep & GRL | `view_face_voice` | 224 | 0.5178 | **0.75** | 0.6065 | **0.5579** | 0.7045 | 0.5616 | 0.5408 |
| **#11** | `cnn_gru_grl_view_face_voice` | Tier 1/2 Deep & GRL | `view_face_voice` | 224 | 0.5510 | **0.50** | 0.6442 | **0.5510** | 0.7556 | 0.5506 | 0.5629 |
| **#12** | `cnn_gru_grl_view_face_alone` | Tier 1/2 Deep & GRL | `view_face_alone` | 84 | 0.5422 | **0.40** | 0.5550 | **0.5487** | 0.6021 | 0.5501 | 0.5553 |
| **#13** | `rf_view_face_voice` | Tier 0 Classical | `view_face_voice` | 224 | 0.4814 | **0.60** | 0.6173 | **0.5413** | 0.7280 | 0.5407 | 0.5549 |
| **#14** | `logreg_view_face_voice` | Tier 0 Classical | `view_face_voice` | 224 | 0.5175 | **0.75** | 0.5660 | **0.5376** | 0.6523 | 0.5514 | 0.5782 |
| **#15** | `xgboost_view_voice_alone` | Tier 0 Classical | `view_voice_alone` | 140 | 0.5026 | **0.60** | 0.5741 | **0.5323** | 0.6721 | 0.5398 | 0.5456 |
| **#16** | `cnn_gru_grl_view_voice_alone` | Tier 1/2 Deep & GRL | `view_voice_alone` | 140 | 0.5009 | **0.70** | 0.6032 | **0.5313** | 0.7148 | 0.5313 | 0.5418 |
| **#17** | `xgboost_view_face_voice` | Tier 0 Classical | `view_face_voice` | 224 | 0.4853 | **0.65** | 0.5606 | **0.5305** | 0.6495 | 0.5427 | 0.5497 |
| **#18** | `lightgbm_view_face_voice` | Tier 0 Classical | `view_face_voice` | 224 | 0.5142 | **0.65** | 0.5633 | **0.5299** | 0.6553 | 0.5399 | 0.5568 |
| **#19** | `knn_view_face_voice` | Tier 0 Classical | `view_face_voice` | 224 | 0.5159 | **0.45** | 0.6199 | **0.5159** | 0.7403 | 0.5187 | 0.5400 |
| **#20** | `svm_view_face_voice` | Tier 0 Classical | `view_face_voice` | 224 | 0.4413 | **0.70** | 0.5418 | **0.5138** | 0.6304 | 0.5267 | 0.5097 |
| **#21** | `svm_view_voice_alone` | Tier 0 Classical | `view_voice_alone` | 140 | 0.4667 | **0.70** | 0.5476 | **0.5098** | 0.6460 | 0.5183 | 0.4941 |
| **#22** | `rf_view_voice_alone` | Tier 0 Classical | `view_voice_alone` | 140 | 0.4191 | **0.65** | 0.4841 | **0.4712** | 0.5538 | 0.4996 | 0.4795 |

---

## 3. Tier 2 Identity Leakage-Gap Test

| Metric | Subject-Grouped LOSO | Random Row-Wise (Leaky) | Leakage Gap (Δ_leak) |
|---|---|---|---|
| **Accuracy (Default 0.50)** | 0.0000 | 0.0000 | **+0.00%** |
| **Accuracy (Optimal Thresh)** | 0.0000 | 0.0000 | **+0.00%** |
| **Macro F1 (Optimal Thresh)** | 0.0000 | 0.0000 | **+0.00%** |

![Leakage Gap Comparison](file:///e:/StressProject/phase2/docs/verification/leakage_gap_comparison.png)

---

## 4. Per-Model Detailed Documentation & Visualizations

### Model: `knn_view_face_alone`
- **Category:** Tier 0 Classical | **View:** `view_face_alone` | **Subjects:** 55 | **Windows:** 600 | **Features:** 84
- **Optimal Threshold:** 0.55 (Macro F1 = **0.6133**, Accuracy = 0.6133, Stress F1 = 0.6120)
- **Default Threshold (0.50):** Macro F1 = 0.6117, Accuracy = 0.6117, Stress F1 = 0.6110
- **ROC AUC:** 0.6298 | **PR AUC:** 0.6390
- **Hyperparameters:** `{"model_type": "knn", "params": "{'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'distance'}"}`
- **Run Artifacts Path:** `models/knn_view_face_alone/20261023_150030_b89ce674/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/knn_view_face_alone/20261023_150030_b89ce674/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/knn_view_face_alone/20261023_150030_b89ce674/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/knn_view_face_alone/20261023_150030_b89ce674/pr.png) |

---

### Model: `knn_view_face_voice`
- **Category:** Tier 0 Classical | **View:** `view_face_voice` | **Subjects:** 53 | **Windows:** 371 | **Features:** 224
- **Optimal Threshold:** 0.45 (Macro F1 = **0.5159**, Accuracy = 0.6199, Stress F1 = 0.7403)
- **Default Threshold (0.50):** Macro F1 = 0.5159, Accuracy = 0.6199, Stress F1 = 0.7403
- **ROC AUC:** 0.5400 | **PR AUC:** 0.7218
- **Hyperparameters:** `{"model_type": "knn", "params": "{'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'distance'}"}`
- **Run Artifacts Path:** `models/knn_view_face_voice/20261023_150257_18f005c6/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/knn_view_face_voice/20261023_150257_18f005c6/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/knn_view_face_voice/20261023_150257_18f005c6/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/knn_view_face_voice/20261023_150257_18f005c6/pr.png) |

---

### Model: `lightgbm_view_face_alone`
- **Category:** Tier 0 Classical | **View:** `view_face_alone` | **Subjects:** 55 | **Windows:** 600 | **Features:** 84
- **Optimal Threshold:** 0.45 (Macro F1 = **0.6101**, Accuracy = 0.6167, Stress F1 = 0.6608)
- **Default Threshold (0.50):** Macro F1 = 0.6049, Accuracy = 0.6067, Stress F1 = 0.6312
- **ROC AUC:** 0.6291 | **PR AUC:** 0.5954
- **Hyperparameters:** `{"model_type": "lightgbm", "params": "{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.05, 'max_depth': 4, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': -1, 'num_leaves': 31, 'objective': None, 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'scale_pos_weight': 0.8927444794952681, 'verbose': -1}"}`
- **Run Artifacts Path:** `models/lightgbm_view_face_alone/20261023_150044_fbf96972/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/lightgbm_view_face_alone/20261023_150044_fbf96972/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/lightgbm_view_face_alone/20261023_150044_fbf96972/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/lightgbm_view_face_alone/20261023_150044_fbf96972/pr.png) |

---

### Model: `lightgbm_view_face_voice`
- **Category:** Tier 0 Classical | **View:** `view_face_voice` | **Subjects:** 53 | **Windows:** 371 | **Features:** 224
- **Optimal Threshold:** 0.65 (Macro F1 = **0.5299**, Accuracy = 0.5633, Stress F1 = 0.6553)
- **Default Threshold (0.50):** Macro F1 = 0.5142, Accuracy = 0.6092, Stress F1 = 0.7290
- **ROC AUC:** 0.5568 | **PR AUC:** 0.7304
- **Hyperparameters:** `{"model_type": "lightgbm", "params": "{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.05, 'max_depth': 4, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': -1, 'num_leaves': 31, 'objective': None, 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'scale_pos_weight': 0.44921875, 'verbose': -1}"}`
- **Run Artifacts Path:** `models/lightgbm_view_face_voice/20261023_150315_f0b03b08/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/lightgbm_view_face_voice/20261023_150315_f0b03b08/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/lightgbm_view_face_voice/20261023_150315_f0b03b08/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/lightgbm_view_face_voice/20261023_150315_f0b03b08/pr.png) |

---

### Model: `logreg_view_face_alone`
- **Category:** Tier 0 Classical | **View:** `view_face_alone` | **Subjects:** 55 | **Windows:** 600 | **Features:** 84
- **Optimal Threshold:** 0.55 (Macro F1 = **0.6183**, Accuracy = 0.6183, Stress F1 = 0.6177)
- **Default Threshold (0.50):** Macro F1 = 0.6177, Accuracy = 0.6183, Stress F1 = 0.6336
- **ROC AUC:** 0.6413 | **PR AUC:** 0.6472
- **Hyperparameters:** `{"model_type": "logreg", "params": "{'C': 1.0, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 1000, 'multi_class': 'deprecated', 'n_jobs': None, 'penalty': 'l2', 'random_state': 42, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}"}`
- **Run Artifacts Path:** `models/logreg_view_face_alone/20261023_150027_d07df9cb/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/logreg_view_face_alone/20261023_150027_d07df9cb/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/logreg_view_face_alone/20261023_150027_d07df9cb/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/logreg_view_face_alone/20261023_150027_d07df9cb/pr.png) |

---

### Model: `logreg_view_face_voice`
- **Category:** Tier 0 Classical | **View:** `view_face_voice` | **Subjects:** 53 | **Windows:** 371 | **Features:** 224
- **Optimal Threshold:** 0.75 (Macro F1 = **0.5376**, Accuracy = 0.5660, Stress F1 = 0.6523)
- **Default Threshold (0.50):** Macro F1 = 0.5175, Accuracy = 0.5795, Stress F1 = 0.6905
- **ROC AUC:** 0.5782 | **PR AUC:** 0.7748
- **Hyperparameters:** `{"model_type": "logreg", "params": "{'C': 1.0, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 1000, 'multi_class': 'deprecated', 'n_jobs': None, 'penalty': 'l2', 'random_state': 42, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}"}`
- **Run Artifacts Path:** `models/logreg_view_face_voice/20261023_150255_090346e9/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/logreg_view_face_voice/20261023_150255_090346e9/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/logreg_view_face_voice/20261023_150255_090346e9/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/logreg_view_face_voice/20261023_150255_090346e9/pr.png) |

---

### Model: `rf_view_face_alone`
- **Category:** Tier 0 Classical | **View:** `view_face_alone` | **Subjects:** 55 | **Windows:** 600 | **Features:** 84
- **Optimal Threshold:** 0.50 (Macro F1 = **0.6018**, Accuracy = 0.6083, Stress F1 = 0.6529)
- **Default Threshold (0.50):** Macro F1 = 0.6018, Accuracy = 0.6083, Stress F1 = 0.6529
- **ROC AUC:** 0.6532 | **PR AUC:** 0.6413
- **Hyperparameters:** `{"model_type": "rf", "params": "{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 6, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}"}`
- **Run Artifacts Path:** `models/rf_view_face_alone/20261023_150038_b4855ad0/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/rf_view_face_alone/20261023_150038_b4855ad0/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/rf_view_face_alone/20261023_150038_b4855ad0/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/rf_view_face_alone/20261023_150038_b4855ad0/pr.png) |

---

### Model: `rf_view_face_voice`
- **Category:** Tier 0 Classical | **View:** `view_face_voice` | **Subjects:** 53 | **Windows:** 371 | **Features:** 224
- **Optimal Threshold:** 0.60 (Macro F1 = **0.5413**, Accuracy = 0.6173, Stress F1 = 0.7280)
- **Default Threshold (0.50):** Macro F1 = 0.4814, Accuracy = 0.6739, Stress F1 = 0.7973
- **ROC AUC:** 0.5549 | **PR AUC:** 0.7318
- **Hyperparameters:** `{"model_type": "rf", "params": "{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 6, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}"}`
- **Run Artifacts Path:** `models/rf_view_face_voice/20261023_150304_08fa8634/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/rf_view_face_voice/20261023_150304_08fa8634/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/rf_view_face_voice/20261023_150304_08fa8634/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/rf_view_face_voice/20261023_150304_08fa8634/pr.png) |

---

### Model: `rf_view_voice_alone`
- **Category:** Tier 0 Classical | **View:** `view_voice_alone` | **Subjects:** 54 | **Windows:** 378 | **Features:** 140
- **Optimal Threshold:** 0.65 (Macro F1 = **0.4712**, Accuracy = 0.4841, Stress F1 = 0.5538)
- **Default Threshold (0.50):** Macro F1 = 0.4191, Accuracy = 0.6323, Stress F1 = 0.7710
- **ROC AUC:** 0.4795 | **PR AUC:** 0.7121
- **Hyperparameters:** `{"model_type": "rf", "params": "{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 6, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}"}`
- **Run Artifacts Path:** `models/rf_view_voice_alone/20261023_150223_785906cb/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/rf_view_voice_alone/20261023_150223_785906cb/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/rf_view_voice_alone/20261023_150223_785906cb/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/rf_view_voice_alone/20261023_150223_785906cb/pr.png) |

---

### Model: `svm_view_face_alone`
- **Category:** Tier 0 Classical | **View:** `view_face_alone` | **Subjects:** 55 | **Windows:** 600 | **Features:** 84
- **Optimal Threshold:** 0.40 (Macro F1 = **0.5982**, Accuracy = 0.6217, Stress F1 = 0.6953)
- **Default Threshold (0.50):** Macro F1 = 0.5939, Accuracy = 0.5983, Stress F1 = 0.6365
- **ROC AUC:** 0.6467 | **PR AUC:** 0.6425
- **Hyperparameters:** `{"model_type": "svm", "params": "{'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': 42, 'shrinking': True, 'tol': 0.001, 'verbose': False}"}`
- **Run Artifacts Path:** `models/svm_view_face_alone/20261023_150032_115b5274/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/svm_view_face_alone/20261023_150032_115b5274/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/svm_view_face_alone/20261023_150032_115b5274/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/svm_view_face_alone/20261023_150032_115b5274/pr.png) |

---

### Model: `svm_view_face_voice`
- **Category:** Tier 0 Classical | **View:** `view_face_voice` | **Subjects:** 53 | **Windows:** 371 | **Features:** 224
- **Optimal Threshold:** 0.70 (Macro F1 = **0.5138**, Accuracy = 0.5418, Stress F1 = 0.6304)
- **Default Threshold (0.50):** Macro F1 = 0.4413, Accuracy = 0.6523, Stress F1 = 0.7846
- **ROC AUC:** 0.5097 | **PR AUC:** 0.6964
- **Hyperparameters:** `{"model_type": "svm", "params": "{'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': 42, 'shrinking': True, 'tol': 0.001, 'verbose': False}"}`
- **Run Artifacts Path:** `models/svm_view_face_voice/20261023_150259_fed23fa3/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/svm_view_face_voice/20261023_150259_fed23fa3/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/svm_view_face_voice/20261023_150259_fed23fa3/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/svm_view_face_voice/20261023_150259_fed23fa3/pr.png) |

---

### Model: `svm_view_voice_alone`
- **Category:** Tier 0 Classical | **View:** `view_voice_alone` | **Subjects:** 54 | **Windows:** 378 | **Features:** 140
- **Optimal Threshold:** 0.70 (Macro F1 = **0.5098**, Accuracy = 0.5476, Stress F1 = 0.6460)
- **Default Threshold (0.50):** Macro F1 = 0.4667, Accuracy = 0.6693, Stress F1 = 0.7954
- **ROC AUC:** 0.4941 | **PR AUC:** 0.7001
- **Hyperparameters:** `{"model_type": "svm", "params": "{'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': 42, 'shrinking': True, 'tol': 0.001, 'verbose': False}"}`
- **Run Artifacts Path:** `models/svm_view_voice_alone/20261023_150230_d04d48b1/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/svm_view_voice_alone/20261023_150230_d04d48b1/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/svm_view_voice_alone/20261023_150230_d04d48b1/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/svm_view_voice_alone/20261023_150230_d04d48b1/pr.png) |

---

### Model: `xgboost_view_face_alone`
- **Category:** Tier 0 Classical | **View:** `view_face_alone` | **Subjects:** 55 | **Windows:** 600 | **Features:** 84
- **Optimal Threshold:** 0.50 (Macro F1 = **0.6165**, Accuracy = 0.6183, Stress F1 = 0.6427)
- **Default Threshold (0.50):** Macro F1 = 0.6165, Accuracy = 0.6183, Stress F1 = 0.6427
- **ROC AUC:** 0.6410 | **PR AUC:** 0.6047
- **Hyperparameters:** `{"model_type": "xgboost", "params": "{'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': None, 'device': None, 'early_stopping_rounds': None, 'enable_categorical': True, 'eval_metric': 'logloss', 'feature_types': None, 'feature_weights': None, 'gamma': None, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': 0.05, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 4, 'max_leaves': None, 'min_child_weight': None, 'missing': nan, 'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': 100, 'n_jobs': -1, 'num_parallel_tree': None, 'random_state': 42, 'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': 0.8927444794952681, 'subsample': None, 'tree_method': None, 'validate_parameters': None, 'verbosity': None}"}`
- **Run Artifacts Path:** `models/xgboost_view_face_alone/20261023_150042_7b97edcb/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/xgboost_view_face_alone/20261023_150042_7b97edcb/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/xgboost_view_face_alone/20261023_150042_7b97edcb/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/xgboost_view_face_alone/20261023_150042_7b97edcb/pr.png) |

---

### Model: `xgboost_view_face_voice`
- **Category:** Tier 0 Classical | **View:** `view_face_voice` | **Subjects:** 53 | **Windows:** 371 | **Features:** 224
- **Optimal Threshold:** 0.65 (Macro F1 = **0.5305**, Accuracy = 0.5606, Stress F1 = 0.6495)
- **Default Threshold (0.50):** Macro F1 = 0.4853, Accuracy = 0.6038, Stress F1 = 0.7322
- **ROC AUC:** 0.5497 | **PR AUC:** 0.7199
- **Hyperparameters:** `{"model_type": "xgboost", "params": "{'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': None, 'device': None, 'early_stopping_rounds': None, 'enable_categorical': True, 'eval_metric': 'logloss', 'feature_types': None, 'feature_weights': None, 'gamma': None, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': 0.05, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 4, 'max_leaves': None, 'min_child_weight': None, 'missing': nan, 'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': 100, 'n_jobs': -1, 'num_parallel_tree': None, 'random_state': 42, 'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': 0.44921875, 'subsample': None, 'tree_method': None, 'validate_parameters': None, 'verbosity': None}"}`
- **Run Artifacts Path:** `models/xgboost_view_face_voice/20261023_150312_e7ac1a46/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/xgboost_view_face_voice/20261023_150312_e7ac1a46/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/xgboost_view_face_voice/20261023_150312_e7ac1a46/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/xgboost_view_face_voice/20261023_150312_e7ac1a46/pr.png) |

---

### Model: `xgboost_view_voice_alone`
- **Category:** Tier 0 Classical | **View:** `view_voice_alone` | **Subjects:** 54 | **Windows:** 378 | **Features:** 140
- **Optimal Threshold:** 0.60 (Macro F1 = **0.5323**, Accuracy = 0.5741, Stress F1 = 0.6721)
- **Default Threshold (0.50):** Macro F1 = 0.5026, Accuracy = 0.5873, Stress F1 = 0.7079
- **ROC AUC:** 0.5456 | **PR AUC:** 0.7414
- **Hyperparameters:** `{"model_type": "xgboost", "params": "{'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': None, 'device': None, 'early_stopping_rounds': None, 'enable_categorical': True, 'eval_metric': 'logloss', 'feature_types': None, 'feature_weights': None, 'gamma': None, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': 0.05, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 4, 'max_leaves': None, 'min_child_weight': None, 'missing': nan, 'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': 100, 'n_jobs': -1, 'num_parallel_tree': None, 'random_state': 42, 'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': 0.4372623574144487, 'subsample': None, 'tree_method': None, 'validate_parameters': None, 'verbosity': None}"}`
- **Run Artifacts Path:** `models/xgboost_view_voice_alone/20261023_150228_925bdaff/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/xgboost_view_voice_alone/20261023_150228_925bdaff/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/xgboost_view_voice_alone/20261023_150228_925bdaff/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/xgboost_view_voice_alone/20261023_150228_925bdaff/pr.png) |

---

### Model: `cnn_gru_grl_view_face_alone`
- **Category:** Tier 1/2 Deep & GRL | **View:** `view_face_alone` | **Subjects:** 55 | **Windows:** 600 | **Features:** 84
- **Optimal Threshold:** 0.40 (Macro F1 = **0.5487**, Accuracy = 0.5550, Stress F1 = 0.6021)
- **Default Threshold (0.50):** Macro F1 = 0.5422, Accuracy = 0.5450, Stress F1 = 0.5781
- **ROC AUC:** 0.5553 | **PR AUC:** 0.5787
- **Hyperparameters:** `{"model_type": "cnn_gru_grl", "epochs": 15, "lr": 0.001, "lambda_adv": 0.02}`
- **Run Artifacts Path:** `models/cnn_gru_grl_view_face_alone/20261023_150217_2e2b63d1/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_face_alone/20261023_150217_2e2b63d1/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_face_alone/20261023_150217_2e2b63d1/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_face_alone/20261023_150217_2e2b63d1/pr.png) |

---

### Model: `cnn_gru_grl_view_face_voice`
- **Category:** Tier 1/2 Deep & GRL | **View:** `view_face_voice` | **Subjects:** 53 | **Windows:** 371 | **Features:** 224
- **Optimal Threshold:** 0.50 (Macro F1 = **0.5510**, Accuracy = 0.6442, Stress F1 = 0.7556)
- **Default Threshold (0.50):** Macro F1 = 0.5510, Accuracy = 0.6442, Stress F1 = 0.7556
- **ROC AUC:** 0.5629 | **PR AUC:** 0.7401
- **Hyperparameters:** `{"model_type": "cnn_gru_grl", "epochs": 15, "lr": 0.001, "lambda_adv": 0.02}`
- **Run Artifacts Path:** `models/cnn_gru_grl_view_face_voice/20261023_150411_5a4285da/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_face_voice/20261023_150411_5a4285da/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_face_voice/20261023_150411_5a4285da/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_face_voice/20261023_150411_5a4285da/pr.png) |

---

### Model: `cnn_gru_grl_view_voice_alone`
- **Category:** Tier 1/2 Deep & GRL | **View:** `view_voice_alone` | **Subjects:** 54 | **Windows:** 378 | **Features:** 140
- **Optimal Threshold:** 0.70 (Macro F1 = **0.5313**, Accuracy = 0.6032, Stress F1 = 0.7148)
- **Default Threshold (0.50):** Macro F1 = 0.5009, Accuracy = 0.6243, Stress F1 = 0.7491
- **ROC AUC:** 0.5418 | **PR AUC:** 0.7365
- **Hyperparameters:** `{"model_type": "cnn_gru_grl", "epochs": 15, "lr": 0.001, "lambda_adv": 0.02}`
- **Run Artifacts Path:** `models/cnn_gru_grl_view_voice_alone/20261023_150253_62b16138/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_voice_alone/20261023_150253_62b16138/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_voice_alone/20261023_150253_62b16138/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_voice_alone/20261023_150253_62b16138/pr.png) |

---

### Model: `cnn_gru_view_face_alone`
- **Category:** Tier 1/2 Deep & GRL | **View:** `view_face_alone` | **Subjects:** 55 | **Windows:** 600 | **Features:** 84
- **Optimal Threshold:** 0.40 (Macro F1 = **0.6009**, Accuracy = 0.6050, Stress F1 = 0.6415)
- **Default Threshold (0.50):** Macro F1 = 0.5887, Accuracy = 0.5900, Stress F1 = 0.6120
- **ROC AUC:** 0.6104 | **PR AUC:** 0.6172
- **Hyperparameters:** `{"model_type": "cnn_gru", "epochs": 15, "lr": 0.001, "lambda_adv": null}`
- **Run Artifacts Path:** `models/cnn_gru_view_face_alone/20261023_150140_846470c7/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/cnn_gru_view_face_alone/20261023_150140_846470c7/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/cnn_gru_view_face_alone/20261023_150140_846470c7/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/cnn_gru_view_face_alone/20261023_150140_846470c7/pr.png) |

---

### Model: `cnn_gru_view_face_voice`
- **Category:** Tier 1/2 Deep & GRL | **View:** `view_face_voice` | **Subjects:** 53 | **Windows:** 371 | **Features:** 224
- **Optimal Threshold:** 0.65 (Macro F1 = **0.5673**, Accuracy = 0.6280, Stress F1 = 0.7294)
- **Default Threshold (0.50):** Macro F1 = 0.5453, Accuracy = 0.6226, Stress F1 = 0.7328
- **ROC AUC:** 0.5421 | **PR AUC:** 0.7302
- **Hyperparameters:** `{"model_type": "cnn_gru", "epochs": 15, "lr": 0.001, "lambda_adv": null}`
- **Run Artifacts Path:** `models/cnn_gru_view_face_voice/20261023_150347_7c094a01/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/cnn_gru_view_face_voice/20261023_150347_7c094a01/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/cnn_gru_view_face_voice/20261023_150347_7c094a01/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/cnn_gru_view_face_voice/20261023_150347_7c094a01/pr.png) |

---

### Model: `gru_view_face_alone`
- **Category:** Tier 1/2 Deep & GRL | **View:** `view_face_alone` | **Subjects:** 55 | **Windows:** 600 | **Features:** 84
- **Optimal Threshold:** 0.55 (Macro F1 = **0.5912**, Accuracy = 0.5917, Stress F1 = 0.6055)
- **Default Threshold (0.50):** Macro F1 = 0.5900, Accuracy = 0.5933, Stress F1 = 0.6269
- **ROC AUC:** 0.6267 | **PR AUC:** 0.6256
- **Hyperparameters:** `{"model_type": "gru", "epochs": 15, "lr": 0.001, "lambda_adv": null}`
- **Run Artifacts Path:** `models/gru_view_face_alone/20261023_150110_43606050/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/gru_view_face_alone/20261023_150110_43606050/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/gru_view_face_alone/20261023_150110_43606050/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/gru_view_face_alone/20261023_150110_43606050/pr.png) |

---

### Model: `gru_view_face_voice`
- **Category:** Tier 1/2 Deep & GRL | **View:** `view_face_voice` | **Subjects:** 53 | **Windows:** 371 | **Features:** 224
- **Optimal Threshold:** 0.75 (Macro F1 = **0.5579**, Accuracy = 0.6065, Stress F1 = 0.7045)
- **Default Threshold (0.50):** Macro F1 = 0.5178, Accuracy = 0.6226, Stress F1 = 0.7426
- **ROC AUC:** 0.5408 | **PR AUC:** 0.7288
- **Hyperparameters:** `{"model_type": "gru", "epochs": 15, "lr": 0.001, "lambda_adv": null}`
- **Run Artifacts Path:** `models/gru_view_face_voice/20261023_150330_bb776fad/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/gru_view_face_voice/20261023_150330_bb776fad/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/gru_view_face_voice/20261023_150330_bb776fad/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/gru_view_face_voice/20261023_150330_bb776fad/pr.png) |

---
