# Model Card: svm_view_physio_all

## 1. Overview & Provenance
- **Model Architecture:** svm_view_physio_all
- **Training View:** `view_physio_all`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_132229
- **Run Hash:** `ebb71b49`
- **Subjects Evaluated:** 64 subjects
- **Total Valid Windows:** 1394
- **Input Feature Count:** 132 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6198 | 0.5911 | 0.7200 (RF Table 2) | -0.1289 |
| **Balanced Accuracy** | 0.5445 | 0.5974 | - | - |
| **Macro F1** | 0.5333 | **0.5839** | **0.7300** (RF Table 2) | **-0.1461** |
| **Stress Class F1** | 0.3325 | 0.5289 | - | - |
| **ROC AUC** | 0.6065 | 0.6065 | - | - |
| **PR AUC** | 0.4490 | 0.4490 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.35 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0505 (0.5333 -> 0.5839)
- **Confusion Matrix (Default 0.5):** TN=732, FP=147, FN=383, TP=132
- **Confusion Matrix (Optimal 0.35):** TN=504, FP=375, FN=195, TP=320

## 4. Hyperparameters
```json
{
  "C": 1.0,
  "gamma": "scale"
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
Tier 0 baseline trained on view_physio_all with 132 features across 64 subjects.
