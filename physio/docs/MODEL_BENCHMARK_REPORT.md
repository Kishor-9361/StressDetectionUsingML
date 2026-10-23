# StressID Model Training & Benchmark Report

**Generated Date:** 2026-10-23  
**Evaluation Protocol:** Strict Subject-Grouped Leave-One-Subject-Out (LOSO) Cross-Validation  
**Author:** kishor <kishorveeraragavan28@gmail.com>  
**Governance Specification:** [MODEL_TRAINING_PIPELINE.md](file:///e:/StressProject/phase2/docs/MODEL_TRAINING_PIPELINE.md)  
**Audit Verification:** 100% programmatically generated from verified `metrics.json` artifacts; zero hand-typed values.

---

## 1. Executive Summary & Top Champions

- 🥇 **Multimodal Champion:** `rf_view_physio_face` (**Macro F1 = 0.6698**, Accuracy = 0.6889, ROC AUC = 0.7022 at threshold 0.45).
- 🥈 **Physiological Baseline Champion:** `rf_view_physio_all` (**Macro F1 = 0.6303**, Accuracy = 0.6736, ROC AUC = 0.6517 at threshold 0.45).
- 🥉 **Deep Sequence & GRL Champion:** `cnn_gru_grl_view_physio_all` (**Macro F1 = 0.5962**, Accuracy = 62.55% at default 0.50 threshold).
- **Identity Leakage Empirical Proof (Tier 2):** Random row-wise cross-validation artificially inflated accuracy by **+8.54% to +11.55%** (up to 78.91%) and Macro F1 by **+14.54%** (up to 0.7758), mathematically justifying an adversarial GRL head.

---

## 2. Master Model Comparison Box: All 15 Models Ranked by Macro F1

| Rank | Model | Category | View | Features | Default F1 (0.5) | Opt Thresh | Opt Accuracy | Opt Macro F1 | Stress F1 | Balanced Acc | ROC AUC |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | `rf_view_physio_face` | Tier 0 Classical | `view_physio_face` | 216 | 0.6576 | **0.45** | 0.6889 | **0.6698** | 0.5903 | 0.6701 | 0.7022 |
| **#2** | `xgboost_view_physio_face` | Tier 0 Classical | `view_physio_face` | 216 | 0.6466 | **0.60** | 0.6924 | **0.6565** | 0.5455 | 0.6524 | 0.6850 |
| **#3** | `rf_view_physio_all` | Tier 0 Classical | `view_physio_all` | 132 | 0.6168 | **0.45** | 0.6736 | **0.6303** | 0.5038 | 0.6270 | 0.6517 |
| **#4** | `xgboost_view_physio_all` | Tier 0 Classical | `view_physio_all` | 132 | 0.6230 | **0.50** | 0.6585 | **0.6230** | 0.5072 | 0.6207 | 0.6149 |
| **#5** | `lightgbm_view_physio_all` | Tier 0 Classical | `view_physio_all` | 132 | 0.6124 | **0.55** | 0.6664 | **0.6183** | 0.4828 | 0.6157 | 0.6097 |
| **#6** | `cnn_gru_grl_view_physio_all` | Tier 1/2 Deep & GRL | `view_physio_all` | 132 | 0.5962 | **0.50** | 0.6255 | **0.5962** | 0.4872 | 0.5957 | 0.6185 |
| **#7** | `cnn_gru_view_physio_all` | Tier 1/2 Deep & GRL | `view_physio_all` | 132 | 0.5927 | **0.35** | 0.6126 | **0.5933** | 0.5046 | 0.5963 | 0.6045 |
| **#8** | `logreg_view_physio_all` | Tier 0 Classical | `view_physio_all` | 132 | 0.5852 | **0.65** | 0.6313 | **0.5921** | 0.4657 | 0.5906 | 0.5998 |
| **#9** | `knn_view_physio_all` | Tier 0 Classical | `view_physio_all` | 132 | 0.5366 | **0.25** | 0.5961 | **0.5884** | 0.5320 | 0.6014 | 0.6302 |
| **#10** | `svm_view_physio_all` | Tier 0 Classical | `view_physio_all` | 132 | 0.5333 | **0.35** | 0.5911 | **0.5839** | 0.5289 | 0.5974 | 0.6065 |
| **#11** | `gru_view_physio_all` | Tier 1/2 Deep & GRL | `view_physio_all` | 132 | 0.5701 | **0.10** | 0.5947 | **0.5775** | 0.4924 | 0.5817 | 0.5969 |
| **#12** | `xgboost_view_physio_voice` | Tier 0 Classical | `view_physio_voice` | 272 | 0.4747 | **0.75** | 0.6631 | **0.5593** | 0.7731 | 0.5577 | 0.5607 |
| **#13** | `xgboost_view_tri_modal` | Tier 0 Classical | `view_tri_modal` | 356 | 0.4536 | **0.85** | 0.6057 | **0.5403** | 0.7137 | 0.5434 | 0.5699 |
| **#14** | `rf_view_physio_voice` | Tier 0 Classical | `view_physio_voice` | 272 | 0.4357 | **0.65** | 0.6253 | **0.5371** | 0.7392 | 0.5366 | 0.5441 |
| **#15** | `rf_view_tri_modal` | Tier 0 Classical | `view_tri_modal` | 356 | 0.4208 | **0.70** | 0.5886 | **0.5338** | 0.6936 | 0.5402 | 0.5778 |

---

## 3. Tier 2 Identity Leakage-Gap Test

| Metric | Subject-Grouped LOSO | Random Row-Wise (Leaky) | Leakage Gap (Δ_leak) |
|---|---|---|---|
| **Accuracy (Default 0.50)** | 0.6793 | 0.7647 | **+8.54%** |
| **Accuracy (Optimal Thresh)** | 0.6736 | 0.7891 | **+11.55%** |
| **Macro F1 (Optimal Thresh)** | 0.6303 | 0.7758 | **+14.54%** |

![Leakage Gap Comparison](file:///e:/StressProject/phase2/docs/verification/leakage_gap_comparison.png)

---

## 4. Per-Model Detailed Documentation & Visualizations

### Model: `rf_view_physio_all`
- **Category:** Tier 0 Classical | **View:** `view_physio_all` | **Subjects:** 64 | **Windows:** 1394 | **Features:** 132
- **Optimal Threshold:** 0.45 (Macro F1 = **0.6303**, Accuracy = 0.6736, Stress F1 = 0.5038)
- **Default Threshold (0.50):** Macro F1 = 0.6168, Accuracy = 0.6793, Stress F1 = 0.4621
- **ROC AUC:** 0.6517 | **PR AUC:** 0.5083
- **Hyperparameters:** `{"n_estimators": 100, "max_depth": 12}`
- **Run Artifacts Path:** `models/rf_view_physio_all/20261023_132257_ab0505f0/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/rf_view_physio_all/20261023_132257_ab0505f0/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/rf_view_physio_all/20261023_132257_ab0505f0/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/rf_view_physio_all/20261023_132257_ab0505f0/pr.png) |

---

### Model: `xgboost_view_physio_all`
- **Category:** Tier 0 Classical | **View:** `view_physio_all` | **Subjects:** 64 | **Windows:** 1394 | **Features:** 132
- **Optimal Threshold:** 0.50 (Macro F1 = **0.6230**, Accuracy = 0.6585, Stress F1 = 0.5072)
- **Default Threshold (0.50):** Macro F1 = 0.6230, Accuracy = 0.6585, Stress F1 = 0.5072
- **ROC AUC:** 0.6149 | **PR AUC:** 0.4791
- **Hyperparameters:** `{"max_depth": 4, "n_estimators": 200, "scale_pos_weight": 1.71}`
- **Run Artifacts Path:** `models/xgboost_view_physio_all/20261023_132323_2183a495/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/xgboost_view_physio_all/20261023_132323_2183a495/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/xgboost_view_physio_all/20261023_132323_2183a495/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/xgboost_view_physio_all/20261023_132323_2183a495/pr.png) |

---

### Model: `lightgbm_view_physio_all`
- **Category:** Tier 0 Classical | **View:** `view_physio_all` | **Subjects:** 64 | **Windows:** 1394 | **Features:** 132
- **Optimal Threshold:** 0.55 (Macro F1 = **0.6183**, Accuracy = 0.6664, Stress F1 = 0.4828)
- **Default Threshold (0.50):** Macro F1 = 0.6124, Accuracy = 0.6499, Stress F1 = 0.4917
- **ROC AUC:** 0.6097 | **PR AUC:** 0.4779
- **Hyperparameters:** `{"max_depth": 4, "n_estimators": 200, "scale_pos_weight": 1.71}`
- **Run Artifacts Path:** `models/lightgbm_view_physio_all/20261023_133648_870149b3/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/lightgbm_view_physio_all/20261023_133648_870149b3/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/lightgbm_view_physio_all/20261023_133648_870149b3/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/lightgbm_view_physio_all/20261023_133648_870149b3/pr.png) |

---

### Model: `logreg_view_physio_all`
- **Category:** Tier 0 Classical | **View:** `view_physio_all` | **Subjects:** 64 | **Windows:** 1394 | **Features:** 132
- **Optimal Threshold:** 0.65 (Macro F1 = **0.5921**, Accuracy = 0.6313, Stress F1 = 0.4657)
- **Default Threshold (0.50):** Macro F1 = 0.5852, Accuracy = 0.6055, Stress F1 = 0.4936
- **ROC AUC:** 0.5998 | **PR AUC:** 0.4959
- **Hyperparameters:** `{"C": 1.0}`
- **Run Artifacts Path:** `models/logreg_view_physio_all/20261023_132146_d69bf891/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/logreg_view_physio_all/20261023_132146_d69bf891/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/logreg_view_physio_all/20261023_132146_d69bf891/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/logreg_view_physio_all/20261023_132146_d69bf891/pr.png) |

---

### Model: `knn_view_physio_all`
- **Category:** Tier 0 Classical | **View:** `view_physio_all` | **Subjects:** 64 | **Windows:** 1394 | **Features:** 132
- **Optimal Threshold:** 0.25 (Macro F1 = **0.5884**, Accuracy = 0.5961, Stress F1 = 0.5320)
- **Default Threshold (0.50):** Macro F1 = 0.5366, Accuracy = 0.6420, Stress F1 = 0.3155
- **ROC AUC:** 0.6302 | **PR AUC:** 0.4796
- **Hyperparameters:** `{"n_neighbors": 11, "weights": "distance"}`
- **Run Artifacts Path:** `models/knn_view_physio_all/20261023_132206_950a76f4/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/knn_view_physio_all/20261023_132206_950a76f4/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/knn_view_physio_all/20261023_132206_950a76f4/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/knn_view_physio_all/20261023_132206_950a76f4/pr.png) |

---

### Model: `svm_view_physio_all`
- **Category:** Tier 0 Classical | **View:** `view_physio_all` | **Subjects:** 64 | **Windows:** 1394 | **Features:** 132
- **Optimal Threshold:** 0.35 (Macro F1 = **0.5839**, Accuracy = 0.5911, Stress F1 = 0.5289)
- **Default Threshold (0.50):** Macro F1 = 0.5333, Accuracy = 0.6198, Stress F1 = 0.3325
- **ROC AUC:** 0.6065 | **PR AUC:** 0.4490
- **Hyperparameters:** `{"C": 1.0, "gamma": "scale"}`
- **Run Artifacts Path:** `models/svm_view_physio_all/20261023_132229_ebb71b49/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/svm_view_physio_all/20261023_132229_ebb71b49/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/svm_view_physio_all/20261023_132229_ebb71b49/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/svm_view_physio_all/20261023_132229_ebb71b49/pr.png) |

---

### Model: `rf_view_physio_face`
- **Category:** Tier 0 Classical | **View:** `view_physio_face` | **Subjects:** 52 | **Windows:** 1138 | **Features:** 216
- **Optimal Threshold:** 0.45 (Macro F1 = **0.6698**, Accuracy = 0.6889, Stress F1 = 0.5903)
- **Default Threshold (0.50):** Macro F1 = 0.6576, Accuracy = 0.6898, Stress F1 = 0.5526
- **ROC AUC:** 0.7022 | **PR AUC:** 0.5701
- **Hyperparameters:** `{"n_estimators": 100, "max_depth": 12}`
- **Run Artifacts Path:** `models/rf_view_physio_face/20261023_133714_546bc4ad/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/rf_view_physio_face/20261023_133714_546bc4ad/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/rf_view_physio_face/20261023_133714_546bc4ad/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/rf_view_physio_face/20261023_133714_546bc4ad/pr.png) |

---

### Model: `xgboost_view_physio_face`
- **Category:** Tier 0 Classical | **View:** `view_physio_face` | **Subjects:** 52 | **Windows:** 1138 | **Features:** 216
- **Optimal Threshold:** 0.60 (Macro F1 = **0.6565**, Accuracy = 0.6924, Stress F1 = 0.5455)
- **Default Threshold (0.50):** Macro F1 = 0.6466, Accuracy = 0.6696, Stress F1 = 0.5566
- **ROC AUC:** 0.6850 | **PR AUC:** 0.5536
- **Hyperparameters:** `{"max_depth": 4, "n_estimators": 200, "scale_pos_weight": 1.71}`
- **Run Artifacts Path:** `models/xgboost_view_physio_face/20261023_133744_8633804f/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/xgboost_view_physio_face/20261023_133744_8633804f/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/xgboost_view_physio_face/20261023_133744_8633804f/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/xgboost_view_physio_face/20261023_133744_8633804f/pr.png) |

---

### Model: `rf_view_physio_voice`
- **Category:** Tier 0 Classical | **View:** `view_physio_voice` | **Subjects:** 53 | **Windows:** 371 | **Features:** 272
- **Optimal Threshold:** 0.65 (Macro F1 = **0.5371**, Accuracy = 0.6253, Stress F1 = 0.7392)
- **Default Threshold (0.50):** Macro F1 = 0.4357, Accuracy = 0.6981, Stress F1 = 0.8205
- **ROC AUC:** 0.5441 | **PR AUC:** 0.7263
- **Hyperparameters:** `{"n_estimators": 100, "max_depth": 12}`
- **Run Artifacts Path:** `models/rf_view_physio_voice/20261023_133801_f3148148/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/rf_view_physio_voice/20261023_133801_f3148148/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/rf_view_physio_voice/20261023_133801_f3148148/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/rf_view_physio_voice/20261023_133801_f3148148/pr.png) |

---

### Model: `xgboost_view_physio_voice`
- **Category:** Tier 0 Classical | **View:** `view_physio_voice` | **Subjects:** 53 | **Windows:** 371 | **Features:** 272
- **Optimal Threshold:** 0.75 (Macro F1 = **0.5593**, Accuracy = 0.6631, Stress F1 = 0.7731)
- **Default Threshold (0.50):** Macro F1 = 0.4747, Accuracy = 0.6927, Stress F1 = 0.8131
- **ROC AUC:** 0.5607 | **PR AUC:** 0.7431
- **Hyperparameters:** `{"max_depth": 4, "n_estimators": 200, "scale_pos_weight": 1.71}`
- **Run Artifacts Path:** `models/xgboost_view_physio_voice/20261023_133824_1295413a/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/xgboost_view_physio_voice/20261023_133824_1295413a/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/xgboost_view_physio_voice/20261023_133824_1295413a/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/xgboost_view_physio_voice/20261023_133824_1295413a/pr.png) |

---

### Model: `rf_view_tri_modal`
- **Category:** Tier 0 Classical | **View:** `view_tri_modal` | **Subjects:** 50 | **Windows:** 350 | **Features:** 356
- **Optimal Threshold:** 0.70 (Macro F1 = **0.5338**, Accuracy = 0.5886, Stress F1 = 0.6936)
- **Default Threshold (0.50):** Macro F1 = 0.4208, Accuracy = 0.7000, Stress F1 = 0.8229
- **ROC AUC:** 0.5778 | **PR AUC:** 0.7815
- **Hyperparameters:** `{"n_estimators": 100, "max_depth": 12}`
- **Run Artifacts Path:** `models/rf_view_tri_modal/20261023_133842_ad69a8d3/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/rf_view_tri_modal/20261023_133842_ad69a8d3/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/rf_view_tri_modal/20261023_133842_ad69a8d3/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/rf_view_tri_modal/20261023_133842_ad69a8d3/pr.png) |

---

### Model: `xgboost_view_tri_modal`
- **Category:** Tier 0 Classical | **View:** `view_tri_modal` | **Subjects:** 50 | **Windows:** 350 | **Features:** 356
- **Optimal Threshold:** 0.85 (Macro F1 = **0.5403**, Accuracy = 0.6057, Stress F1 = 0.7137)
- **Default Threshold (0.50):** Macro F1 = 0.4536, Accuracy = 0.6714, Stress F1 = 0.7986
- **ROC AUC:** 0.5699 | **PR AUC:** 0.7844
- **Hyperparameters:** `{"max_depth": 4, "n_estimators": 200, "scale_pos_weight": 1.71}`
- **Run Artifacts Path:** `models/xgboost_view_tri_modal/20261023_133908_98ecb92e/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/xgboost_view_tri_modal/20261023_133908_98ecb92e/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/xgboost_view_tri_modal/20261023_133908_98ecb92e/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/xgboost_view_tri_modal/20261023_133908_98ecb92e/pr.png) |

---

### Model: `cnn_gru_grl_view_physio_all`
- **Category:** Tier 1/2 Deep & GRL | **View:** `view_physio_all` | **Subjects:** 64 | **Windows:** 1394 | **Features:** 132
- **Optimal Threshold:** 0.50 (Macro F1 = **0.5962**, Accuracy = 0.6255, Stress F1 = 0.4872)
- **Default Threshold (0.50):** Macro F1 = 0.5962, Accuracy = 0.6255, Stress F1 = 0.4872
- **ROC AUC:** 0.6185 | **PR AUC:** 0.4522
- **Hyperparameters:** `{"model_type": "cnn_gru", "hidden_dim": 64, "use_grl": true, "lambda_adv": 0.02, "epochs": 25, "optimizer": "AdamW", "lr": 0.002}`
- **Run Artifacts Path:** `models/cnn_gru_grl_view_physio_all/20261023_141155_d5941aac/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_physio_all/20261023_141155_d5941aac/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_physio_all/20261023_141155_d5941aac/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/cnn_gru_grl_view_physio_all/20261023_141155_d5941aac/pr.png) |

---

### Model: `cnn_gru_view_physio_all`
- **Category:** Tier 1/2 Deep & GRL | **View:** `view_physio_all` | **Subjects:** 64 | **Windows:** 1394 | **Features:** 132
- **Optimal Threshold:** 0.35 (Macro F1 = **0.5933**, Accuracy = 0.6126, Stress F1 = 0.5046)
- **Default Threshold (0.50):** Macro F1 = 0.5927, Accuracy = 0.6227, Stress F1 = 0.4823
- **ROC AUC:** 0.6045 | **PR AUC:** 0.4374
- **Hyperparameters:** `{"model_type": "cnn_gru", "hidden_dim": 64, "use_grl": false, "lambda_adv": 0.0, "epochs": 25, "optimizer": "AdamW", "lr": 0.002}`
- **Run Artifacts Path:** `models/cnn_gru_view_physio_all/20261023_141017_5e4e82fa/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/cnn_gru_view_physio_all/20261023_141017_5e4e82fa/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/cnn_gru_view_physio_all/20261023_141017_5e4e82fa/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/cnn_gru_view_physio_all/20261023_141017_5e4e82fa/pr.png) |

---

### Model: `gru_view_physio_all`
- **Category:** Tier 1/2 Deep & GRL | **View:** `view_physio_all` | **Subjects:** 64 | **Windows:** 1394 | **Features:** 132
- **Optimal Threshold:** 0.10 (Macro F1 = **0.5775**, Accuracy = 0.5947, Stress F1 = 0.4924)
- **Default Threshold (0.50):** Macro F1 = 0.5701, Accuracy = 0.6019, Stress F1 = 0.4532
- **ROC AUC:** 0.5969 | **PR AUC:** 0.4407
- **Hyperparameters:** `{"model_type": "gru", "hidden_dim": 64, "use_grl": false, "lambda_adv": 0.0, "epochs": 25, "optimizer": "AdamW", "lr": 0.002}`
- **Run Artifacts Path:** `models/gru_view_physio_all/20261023_140910_f35d593d/`

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|:---:|:---:|:---:|
| ![Confusion Matrix](file:///e:/StressProject/phase2/models/gru_view_physio_all/20261023_140910_f35d593d/confusion_matrix.png) | ![ROC Curve](file:///e:/StressProject/phase2/models/gru_view_physio_all/20261023_140910_f35d593d/roc.png) | ![PR Curve](file:///e:/StressProject/phase2/models/gru_view_physio_all/20261023_140910_f35d593d/pr.png) |

---
