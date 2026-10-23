# Model Card: logreg_view_physio_all

## 1. Overview & Provenance
- **Model Architecture:** logreg_view_physio_all
- **Training View:** `view_physio_all`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_132146
- **Run Hash:** `d69bf891`
- **Subjects Evaluated:** 64 subjects
- **Total Valid Windows:** 1394
- **Input Feature Count:** 132 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6055 | 0.6313 | 0.7200 (RF Table 2) | -0.0887 |
| **Balanced Accuracy** | 0.5878 | 0.5906 | - | - |
| **Macro F1** | 0.5852 | **0.5921** | **0.7300** (RF Table 2) | **-0.1379** |
| **Stress Class F1** | 0.4936 | 0.4657 | - | - |
| **ROC AUC** | 0.5998 | 0.5998 | - | - |
| **PR AUC** | 0.4959 | 0.4959 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.65 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.0069 (0.5852 -> 0.5921)
- **Confusion Matrix (Default 0.5):** TN=576, FP=303, FN=247, TP=268
- **Confusion Matrix (Optimal 0.65):** TN=656, FP=223, FN=291, TP=224

## 4. Hyperparameters
```json
{
  "C": 1.0
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
