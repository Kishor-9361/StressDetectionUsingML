# Agent Directive and Execution Instructions

## Directive
Execute `docs/EXTRACTION_AND_EDA_PIPELINE.md` end to end (Part A then Part B). It is the detailed version of Phases 1-2 in `docs/RESTART_PIPELINE_SPEC.md` — read that file first for the git/commit convention and the overall restart rationale, and `docs/RESEARCH_LOG.md` for prior decisions and open items before starting. Follow every gate exactly as stated; don't skip a gate or report a number that isn't read from an artifact. Commit after each gate per the convention in `RESTART_PIPELINE_SPEC.md` §2. Append new findings/decisions to `docs/RESEARCH_LOG.md` as dated entries — never edit past entries. Stop and report back if the subject-count reconciliation (§A.1) doesn't resolve cleanly, or if any gate fails.

## Explicit Reading Order
1. `docs/RESTART_PIPELINE_SPEC.md` (Git/commit conventions, restart rationale, overall architecture)
2. `docs/RESEARCH_LOG.md` (Skim for prior findings, baseline numbers, decisions, and open items)
3. `docs/EXTRACTION_AND_EDA_PIPELINE.md` (The primary operational document to execute end-to-end)
4. `docs/FEATURE_EXTRACTION_PROTOCOL.md` and `docs/PIPELINE_IMPLEMENTATION_SPEC.md` (Referenced inline when executing specific stages/modules)
5. `docs/DASHBOARD_PIPELINE_SPEC.md` (For dashboard development following EDA)

## Hard Constraints
- Ground truth from artifacts only: Never hand-type numbers or metrics into reports or markdown files.
- Modality separation: Never merge modalities into a NaN-padded single table. Keep `ecg_features.parquet`, `eda_features.parquet`, `resp_features.parquet`, `face_features.parquet`, and `voice_features.parquet` separate.
- Baseline calibration: Calm-task reference baseline (Breathing for ECG/EDA/Face/Voice; Relax for Respiration).
- Strict gate compliance: Commit after each gate passes.
