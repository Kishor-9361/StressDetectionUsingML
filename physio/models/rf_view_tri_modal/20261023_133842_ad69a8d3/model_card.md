# Model Card: rf_view_tri_modal

## 1. Overview & Provenance
- **Model Architecture:** rf_view_tri_modal
- **Training View:** `view_tri_modal`
- **Evaluation Protocol:** Leave-One-Subject-Out (LOSO) GroupKFold
- **Run Timestamp:** 20261023_133842
- **Run Hash:** `ad69a8d3`
- **Subjects Evaluated:** 50 subjects
- **Total Valid Windows:** 350
- **Input Feature Count:** 356 features

## 2. Benchmark Comparison against Published Targets
| Benchmark Metric | Model (Default 0.5) | Model (Optimal Thresh) | StressID Paper Target | Gap to Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.7000 | 0.5886 | 0.7200 (RF Table 2) | -0.1314 |
| **Balanced Accuracy** | 0.4949 | 0.5402 | - | - |
| **Macro F1** | 0.4208 | **0.5338** | **0.7300** (RF Table 2) | **-0.1962** |
| **Stress Class F1** | 0.8229 | 0.6936 | - | - |
| **ROC AUC** | 0.5778 | 0.5778 | - | - |
| **PR AUC** | 0.7815 | 0.7815 | - | - |

## 3. Threshold Tuning Analysis
- **Default Decision Threshold:** 0.50
- **Optimal Decision Threshold:** 0.70 (swept 0.05 to 0.95, step 0.05)
- **Macro F1 Improvement:** +0.1130 (0.4208 -> 0.5338)
- **Confusion Matrix (Default 0.5):** TN=1, FP=100, FN=5, TP=244
- **Confusion Matrix (Optimal 0.70):** TN=43, FP=58, FN=86, TP=163

## 4. Hyperparameters
```json
{
  "n_estimators": 100,
  "max_depth": 12
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
Tier 0 baseline trained on view_tri_modal with 356 features across 50 subjects.
