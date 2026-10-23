# StressID Extraction + EDA — Detailed Execution Spec for the Coding Agent

**This expands Phases 1-2 of `RESTART_PIPELINE_SPEC.md` into concrete, cell/script-level steps.** Read that document first for the git/commit workflow and the "why" of each decision; this document is the "exactly how" for extraction and EDA only. Model training (Phase 4) is deliberately NOT detailed further here — per the user's own sequencing, it waits until EDA is done and informs window-size/model-tier choices.

Governing sources already in hand — do not re-fetch: `RESEARCH_LOG.md`, `FEATURE_EXTRACTION_PROTOCOL.md`, `PIPELINE_IMPLEMENTATION_SPEC.md`, `DASHBOARD_PIPELINE_SPEC.md`.

---

## Part A — Extraction Pipeline (detailed)

### A.0 Pre-flight (do this before writing any extraction code)

1. Confirm raw StressID data is actually on disk (per the old audit's Blocker 2, raw data was missing in the previous repo — do not assume it's there). If raw physio/video/audio files aren't present, stop and ask for them rather than extracting from someone's old pre-computed `.npz`/`.csv` (that would just inherit the old, uncalibrated pipeline's output).
2. Scaffold the repo per `PIPELINE_IMPLEMENTATION_SPEC.md`'s layout (`stress-research/{config,src/stressres,scripts,tests,notebooks,data/{raw,processed}}`), StressID-only — omit WESAD/EmpathicSchool directories for now.
3. Pin dependencies exactly per that spec's `pyproject.toml` snippet; critically `neurokit2` must be a version that supports `method='biosppy'` for `ecg_process`/`eda_process`/`rsp_process`.
4. `git init`, initial commit (`phase0`) with the empty scaffold + pinned deps + this restart's governing docs copied into `docs/`.

### A.1 Stage 0 — Inventory and subject-count reconciliation (BLOCKING — do not skip)

1. Build `config/subjects.yaml` by scanning the raw data directory programmatically (don't hand-type subject lists).
2. Cross-reference three sources of subject counts that have disagreed before: (a) your own disk scan, (b) the official `Technical-info_final.pdf` manifest (65 physio / 54 or 55 video or audio — ambiguous, resolve it), (c) the official `all_physiological_features.csv` row count ÷ tasks-per-subject.
3. Write a small script, `scripts/reconcile_subjects.py`, that prints the exact set difference between any two of these lists (`set(a) - set(b)`, `set(b) - set(a)`) — not just the counts. A count mismatch with no identified subjects is not resolved.
4. Log the final, resolved subject manifest as a new dated `RESEARCH_LOG.md` entry, naming every subject ID excluded and why (missing a required task, corrupt file, etc. — not "doesn't match some other count").
5. **Gate:** `config/subjects.yaml` subject count matches a believable, individually-justified list. Commit (`phase1-extract` begins here).

### A.2 Stage 1 — Signal cleaning

For each of ECG, EDA, RESP (StressID physio is a single multi-channel stream at 500 Hz per the official recording spec):

1. Load raw signal, segment by task using the official task-timestamp metadata (not guessed offsets).
2. Butterworth bandpass/lowpass per `FEATURE_EXTRACTION_PROTOCOL.md` §3.1-3.3's filter bands, applied with `scipy.signal.filtfilt` (zero-phase — do not use `lfilter`, which introduces phase lag that will misalign peak detection against the label timeline).
3. Run `neurokit2.ecg_process(signal, sampling_rate=500, method='biosppy')` / `eda_process(..., method='biosppy')` / `rsp_process(..., method='biosppy')` — this exact method argument is what matches the official StressID baseline; do not substitute `'neurokit'` even though it's the library default.
4. Save the cleaned, per-task signal arrays to `data/processed/cleaned/<subject>/<task>_<modality>.parquet` — keep cleaned signals on disk (not just features) so a later re-extraction with different windowing doesn't require re-filtering raw data.

### A.3 Stage 2 — Windowing

1. For the 7 interactive stressor tasks: one window = one full task (no sub-windowing).
2. For the 4 long tasks (Breathing 180s, Video1 120s, Video2 180s, Relax 150s): sub-window at 60s with 30s stride, per the HRV ultra-short-term validity literature already logged (RMSSD stable from ~10s at rest; frequency-domain HRV needs the full 60s).
3. Emit a `window_index.parquet` with columns `subject_id, task_id, window_id, window_start_s, window_end_s, modality_coverage` (bitmask of which modalities have valid data for this window) — this is the backbone every feature table joins against; build and verify it before writing a single feature extractor.

### A.4 Stage 3 — Per-modality feature extraction (SEPARATE tables, verified column counts)

For each modality, write its own extractor script and its own output table — never a joined, NaN-padded table at this stage:

1. **ECG** (`scripts/extract_ecg.py` → `data/processed/features/ecg_features.parquet`): R-peak detection, RR-interval correction (reject >20% deviation from 5-neighbor median, interpolate cubically, record `pct_ectopic`), then the full `neurokit2.ecg_process` HRV output (time/frequency/non-linear) = target 35 columns. Assert column count == 35 in a test; if your extractor produces fewer, add the missing HRV families (see the corrected §3.1 of `FEATURE_EXTRACTION_PROTOCOL.md` for which families are likely missing) rather than reporting a wrong count as if it were right.
2. **EDA** (`scripts/extract_eda.py` → `eda_features.parquet`): lowpass + downsample to 4-10Hz, `eda_phasic(method='cvxEDA')` (record the method — it changes SCR counts), SCR peak detection with a stated amplitude threshold. Target 23 columns; assert and extend per §3.2.
3. **RESP** (`scripts/extract_resp.py` → `resp_features.parquet`): bandpass 0.1-0.35Hz, inhale/exhale segmentation, full RRV output. Target 40 columns; assert and extend per §3.3. **Anchor against `Relax`, not `Breathing`**, per the logged confound (StressID's "calm baseline" breathing task deliberately manipulates respiration).
4. **Face** (`scripts/extract_face.py` → `face_features.parquet`): downsample video to 5fps, run OpenFace 2.0, extract the 18-AU official set + 2 gaze vectors, mean+SD each = 84 columns.
5. **Voice** (`scripts/extract_voice.py` → `voice_features.parquet`): only talking-task subjects will have non-null rows here — do not impute zeros for silent tasks, leave them genuinely absent and record that in `modality_coverage`.
6. **Coverage report** (`scripts/build_coverage_report.py` → `coverage_report.json`): per modality, which subjects/tasks from the Stage-0 manifest are present vs. absent, cross-checked against the official missing-modality manifest (physio-only: `hh2e, wfsl, 37ir, hvpa, ql3b, r0a3, uyrl, qx2o, dmbd`; physio+audio-only: `u3v9, 7m3c`; physio+video-only: `f6q3`).
7. **Gate:** all 5 feature tables exist, physio column counts verified against `all_physiological_features.csv` (re-clone `robustml-eurecom/stressID` only if the earlier clone is gone), coverage report matches the official manifest with zero unexplained drift. Commit.

### A.5 Stage 4 — Baseline anchoring

1. Compute per-subject calm-baseline statistics (robust median/MAD, not mean/std) from the Breathing task for ECG/EDA/face/voice, and from the Relax task for respiration.
2. Apply `z = (x - median_calm) / (1.4826 * MAD_calm + ε)` per feature, per subject — never normalize against whole-recording statistics (the exact bug found in the old project's `normalize_features.py`).
3. Save as `data/processed/calibrated/<modality>_calibrated.parquet`, keeping the uncalibrated version too (you'll want both for the EDA's "with/without baseline-anchoring" comparison).
4. **Gate:** spot-check 3 subjects' calibrated vs. raw distributions by eye (a plot, saved to `docs/verification/`) — calibrated calm-task values should center near zero.
5. **Gate:** 3-subject R-peak visual verification per `FEATURE_EXTRACTION_PROTOCOL.md` §3.1 (plot peaks over raw ECG for 10s baseline + 10s stressor). Commit — this closes Phase 1.

---

## Part B — EDA Notebook (detailed, cell-by-cell)

**File:** `notebooks/01_stressid_eda.ipynb`. Every section below is a notebook section ending in a **written** conclusion cell (markdown, 2-4 sentences, not just a plot) — the notebook is read by a human deciding what to build next, not just an artifact.

### B.0 Setup cell
Load all 5 feature tables + `window_index.parquet` + `coverage_report.json`. Assert row/column counts match Part A's gates before any analysis runs (fail loudly if extraction silently changed since the notebook was last run).

### B.1 Coverage & integrity
- Table: subjects × modality availability, cross-tabulated against the official manifest.
- Plot: a simple presence/absence heatmap (subject × modality).
- **Decision to produce:** which subjects, if any, should be excluded from which model tiers (e.g. a subject missing a required calibration task can't be baseline-anchored and should be flagged, not silently zero-filled).

### B.2 Label distribution and threshold validity
- Histogram of self-report ratings (0-10) per task, with the ≥5 binarization cutoff marked.
- Check: is the distribution actually bimodal around the cutoff, or is "stressed" a lossy cut through a continuous distribution? Compute % of ratings within ±1 of the cutoff (the ambiguous zone).
- Class balance table, per task and overall.
- **Decision to produce:** whether the binary target is defensible as-is, or whether an ordinal/continuous target should be kept as a secondary experiment later (do not change the target now — just record the finding).

### B.3 Per-subject effect size (the label-validity engine)
- For each modality, each subject, each stress task: Cohen's d between that task's windows and the subject's own calm-baseline windows.
- Heatmap: subjects (rows) × tasks (columns), colored by |d|.
- Flag subjects/tasks with |d| < 0.2 (negligible effect) explicitly by ID — these are the ones where the label may not reflect a real physiological/behavioral shift for that subject.
- **Decision to produce:** a concrete exclusion or down-weighting list, OR a decision to keep everyone and let the model sort it out — state which, and why.

### B.4 Within-task dynamics (windowing sanity check)
- For the 4 long sub-windowed tasks only: plot each feature's trajectory across sub-windows (median line + subject-level traces) to check whether the signal actually drifts within a task.
- For the 7 one-window interactive tasks: this is a sanity check that the "no sub-windowing" decision isn't hiding real within-task variation — if feasible, pull a few raw signal traces and visually inspect, don't just trust the design doc.
- **Decision to produce:** confirm or revise the windowing choice already logged; if revised, say exactly what changes and why.

### B.5 Feature distributions and separability ceiling
- Per-feature calm-vs-stress distribution plots (violin or overlaid histograms), physio/face/voice separately, calibrated vs. uncalibrated side by side.
- Quick ceiling check: single best feature's AUC per modality (not a full model — just `roc_auc_score` on one feature at a time) as a sanity floor before any real model is trained.
- **Decision to produce:** does calibration (Stage A.5) actually help separability, or hurt it for any feature family? This directly informs whether Phase 4's models should train on calibrated, raw, or both feature sets.

### B.6 Signal quality triage
- Distribution of `q_ecg`/`q_eda`/`q_resp` quality metrics (computed during Part A extraction) across all windows.
- Table: what fraction of windows would be dropped at quality thresholds 0.5 / 0.7 / 0.9.
- **Decision to produce:** the quality threshold Phase 4 training should use, with the resulting usable-window count stated explicitly (not hidden until training time).

### B.7 Window-size sensitivity (informs Tier 0/1 choice in Phase 4)
- This is the one section that deliberately previews a training decision: re-run the single-best-feature AUC check (B.5) and the quality-pass-rate check (B.6) imagining 2s/5s/10s sub-windows for the 4 long tasks (you already have the raw cleaned signal from Stage A.2, so this doesn't require a second extraction pass — just re-window a copy for this analysis).
- **Decision to produce:** which window size Phase 4 Tier 0 should actually use — this is the single most load-bearing EDA output, since it prevents training classical baselines 3× speculatively the way the old project did.

### B.8 Modality complementarity
- Pairwise correlation of modality-level "stress score" (e.g. the single-best-feature or a simple per-modality logistic regression's predicted probability) across modalities, on the subset of subjects with all modalities present.
- **Decision to produce:** whether fusion (Phase 4 Tier 3) is even likely to help — if all modalities agree almost perfectly, fusion adds little; if they disagree substantially and each only partially tracks the label, that's the actual case for fusion, worth stating explicitly rather than assuming.

### B.9 Final decision summary (top of notebook, written last)
A short list, e.g.:
```
1. Exclude subjects: [...] (reason: ...)
2. Binary target kept as-is; ordinal deferred (reason: ...)
3. Window size for Phase 4 Tier 0: Ns (reason: ...)
4. Quality threshold: 0.X (reason: ..., Y% of windows retained)
5. Calibration: keep / drop for feature family Z (reason: ...)
6. Fusion priority: low/medium/high (reason: ...)
```
Copy this list verbatim into a new dated `RESEARCH_LOG.md` entry — this is the artifact Phase 3 (dashboard) and Phase 4 (training) both consume.

**Gate for the whole notebook:** runs top-to-bottom with no errors on the Phase-1 extraction output, every section (B.1-B.8) has a written conclusion, and B.9's decision list exists and is logged. Commit as `phase2-eda`.
