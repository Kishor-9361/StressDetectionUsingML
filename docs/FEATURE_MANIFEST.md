# StressID Feature Extraction Manifest

This document details all extracted feature tables, directory locations, column inventories, and mathematical definitions produced during Phase 1.

---

## 1. File Locations

### Raw Feature Tables (Parquet & CSV)
- **ECG Features (46 features, 49 columns with IDs)**:
  - Parquet: [`data/processed/features/ecg_features.parquet`](file:///E:/StressProject/phase2/data/processed/features/ecg_features.parquet)
  - CSV: [`data/processed/features_csv/ecg_features.csv`](file:///E:/StressProject/phase2/data/processed/features_csv/ecg_features.csv)
- **EDA Features (24 features, 27 columns with IDs)**:
  - Parquet: [`data/processed/features/eda_features.parquet`](file:///E:/StressProject/phase2/data/processed/features/eda_features.parquet)
  - CSV: [`data/processed/features_csv/eda_features.csv`](file:///E:/StressProject/phase2/data/processed/features_csv/eda_features.csv)
- **Respiration Features (62 features, 65 columns with IDs)**:
  - Parquet: [`data/processed/features/resp_features.parquet`](file:///E:/StressProject/phase2/data/processed/features/resp_features.parquet)
  - CSV: [`data/processed/features_csv/resp_features.csv`](file:///E:/StressProject/phase2/data/processed/features_csv/resp_features.csv)
- **Face Features (84 AU+gaze features, 87 columns with IDs)**:
  - Parquet: [`data/processed/features/face_features.parquet`](file:///E:/StressProject/phase2/data/processed/features/face_features.parquet)
  - CSV: [`data/processed/features_csv/face_features.csv`](file:///E:/StressProject/phase2/data/processed/features_csv/face_features.csv)
- **Voice Features (140 acoustic features, 143 columns with IDs)**:
  - Parquet: [`data/processed/features/voice_features.parquet`](file:///E:/StressProject/phase2/data/processed/features/voice_features.parquet)
  - CSV: [`data/processed/features_csv/voice_features.csv`](file:///E:/StressProject/phase2/data/processed/features_csv/voice_features.csv)

### Calibrated / Baseline-Anchored Tables
Each feature normalized via robust median/MAD against calm baseline: $z = (x - \text{median}_{\text{calm}}) / (1.4826 \cdot \text{MAD}_{\text{calm}} + \epsilon)$:
- [`data/processed/calibrated_csv/ecg_calibrated.csv`](file:///E:/StressProject/phase2/data/processed/calibrated_csv/ecg_calibrated.csv)
- [`data/processed/calibrated_csv/eda_calibrated.csv`](file:///E:/StressProject/phase2/data/processed/calibrated_csv/eda_calibrated.csv)
- [`data/processed/calibrated_csv/resp_calibrated.csv`](file:///E:/StressProject/phase2/data/processed/calibrated_csv/resp_calibrated.csv)
- [`data/processed/calibrated_csv/face_calibrated.csv`](file:///E:/StressProject/phase2/data/processed/calibrated_csv/face_calibrated.csv)
- [`data/processed/calibrated_csv/voice_calibrated.csv`](file:///E:/StressProject/phase2/data/processed/calibrated_csv/voice_calibrated.csv)

### Window Index & Ground Truth
- **Window Backbone Table**: [`data/processed/index/window_index.parquet`](file:///E:/StressProject/phase2/data/processed/index/window_index.parquet)
  - Total Windows: 1,476 (65 subjects × 11 tasks + baseline)
  - Contains: `subject_id`, `task_id`, `window_id`, `window_start_s`, `window_end_s`, `duration_s`, `label_binary`, `affect3_class`, `rating_stress`, `rating_relax`, `rating_valence`, `rating_arousal`, `modality_coverage`.

---

## 2. Feature Inventory by Modality

### A. ECG (46 Features = 35 HRV + 11 Statistical)
- **Time-Domain HRV (19)**:
  `meanHR`, `minHR`, `maxHR`, `sdHR`, `modeHR`, `nNN`, `meanNN`, `SDSD`, `CVNN`, `SDNN`, `pNN50`, `pNN20`, `RMSSD`, `medianNN`, `q20NN`, `q80NN`, `minNN`, `maxNN`, `triHRV`
- **Frequency-Domain HRV (11)**:
  `totalpower`, `LF`, `HF`, `ULF`, `VLF`, `VHF`, `LF/HF`, `rLF`, `rHF`, `peakLF`, `peakHF`
- **Non-Linear Dynamics (5)**:
  `SD1`, `SD2`, `SD1SD2`, `apEn`, `sampEn`
- **Statistical Descriptors (11)**:
  `max_ecg`, `min_ecg`, `mean_ecg`, `sd_ecg`, `ku_ecg`, `sk_ecg`, `median_ecg`, `q1_ecg`, `q3_ecg`, `q05_ecg`, `q95_ecg`

### B. EDA (24 Features)
- **Tonic (SCL) & Phasic (SCR) Descriptors (17)**:
  `max_eda`, `min_eda`, `mean_eda`, `sd_eda`, `ku_eda`, `sk_eda`, `median_eda`, `dynrange`, `scl_slope`, `max_scr`, `min_scr`, `mean_scr`, `sd_scr`, `max_scl`, `min_scl`, `mean_scl`, `sd_scl`
- **SCR Peak Dynamics (7)**:
  `nSCR` (rate/min), `aucSCR`, `meanAmpSCR`, `maxAmpSCR`, `meanRespSCR` (recovery time), `sumAmpSCR`, `sumRespSCR`

### C. Respiration (62 Features)
- **Breathing Intervals & Rates (38)**:
  `meanRR`, `minRR`, `maxRR`, `sdRR`, `modeRR`, `nBB`, `meanBB`, `SDSDb`, `CVNNb`, `SDNNb`, `RMSSDb`, `medianBB`, `q20BB`, `q80BB`, `minBB`, `maxBB`, `meanTT`, `SDTT`, `medianTT`, `q20TT`, `q80TT`, `minTT`, `maxTT`, `meanBA`, `SDBA`, `medianBA`, `q20BA`, `q80BA`, `minBA`, `maxBA`, `meanBW`, `SDBW`, `medianBW`, `q20BW`, `q80BW`, `minBW`, `maxBW`
- **Statistical (11)**:
  `max_rsp`, `min_rsp`, `mean_rsp`, `sd_rsp`, `ku_rsp`, `sk_rsp`, `median_rsp`, `q1_rsp`, `q3_rsp`, `q05_rsp`, `q95_rsp`
- **Frequency Spectral (9)**:
  `totalpower_rsp`, `LF_rsp`, `HF_rsp`, `VLF_rsp`, `VHF_rsp`, `LF/HF_rsp`, `rLF_rsp`, `rHF_rsp`, `peakLF_rsp`, `peakHF_rsp`
- **Non-Linear Dynamics (4)**:
  `SD1_rrv`, `SD2_rrv`, `SD1SD2_rrv`, `apEn_rrv`

### D. Face (84 Features)
- **Action Unit Means (18)**:
  `mean_AU01_r`, `mean_AU02_r`, `mean_AU04_r`, `mean_AU05_r`, `mean_AU06_r`, `mean_AU07_r`, `mean_AU09_r`, `mean_AU10_r`, `mean_AU12_r`, `mean_AU14_r`, `mean_AU15_r`, `mean_AU17_r`, `mean_AU20_r`, `mean_AU23_r`, `mean_AU25_r`, `mean_AU26_r`, `mean_AU45_r`
- **Action Unit Standard Deviations (18)**:
  `std_AU01_r` ... `std_AU45_r`
- **Gaze Vectors (48)**:
  Gaze direction angles and eye vector statistics (mean and std across frames).

### E. Voice (140 Features)
- Handcrafted acoustic features: MFCCs, Δ, ΔΔ, spectral centroid, spectral bandwidth, spectral contrast, spectral flatness, spectral roll-off, harmonic/percussive tonal centroids, zero-crossing rate, tempogram ratios (mean and SD over 5s voiced audio windows).
