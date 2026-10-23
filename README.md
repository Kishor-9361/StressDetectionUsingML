# StressRes — Reproducible Multimodal Stress Detection

## Overview
StressRes is a rigorous restart of the multimodal stress detection pipeline focusing on the **StressID** dataset. It enforces strict audit compliance:
- Ground-truth metrics read directly from serialized artifacts (no hand-typed numbers).
- Modality-separated feature tables (`ecg_features.parquet`, `eda_features.parquet`, `resp_features.parquet`, `face_features.parquet`, `voice_features.parquet`).
- Exact calibration against published baselines (35 ECG, 23 EDA, 40 RESP features).
- Robust baseline-anchoring per subject.

## Directory Layout
- `config/`: Datasets, windowing, features, and subject manifests.
- `docs/`: Governing specifications, protocols, and append-only `RESEARCH_LOG.md`.
- `src/stressres/`: Core library code (`clean`, `features`, `quality`, `windows`, `baseline`).
- `scripts/`: Modular execution scripts for pipeline stages.
- `tests/`: Automated test suite for data integrity, leakage prevention, and reproducibility.
- `notebooks/`: Exploratory analysis and EDA.
- `data/`: Processed parquets (gitignored).
- `Datasets/`: Raw datasets (gitignored).

## Getting Started
Follow `docs/AGENT_INSTRUCTIONS.md` and `docs/EXTRACTION_AND_EDA_PIPELINE.md`.
