# Project Restart — Execution Spec for the Coding Agent

**Audience:** the coding agent that will actually run this (Claude Code / Cursor / whatever executes in the repo).
**Scope:** StressID dataset ONLY. WESAD and EmpathicSchool are out of scope until this restart is proven.
**Mandate:** this is a RE-EXECUTION of work already designed and logged, not new research. Do not re-derive protocol, re-read papers, or re-plan architecture — follow the four governing documents below. If something here conflicts with them, the more recent, more specific instruction wins (this doc > RESEARCH_LOG.md's latest dated entry > the three spec docs).
**Hard constraint carried over from the user:** use tokens/compute efficiently. Don't regenerate artifacts that already pass their gate. Don't re-run a stage if its output and tests already exist and are checked in.

---

## 0. Why this restart exists (read once, then stop re-reading)

An independent repo audit of the previous codebase (`Kishor-9361/StressDetectionUsingML`, branch `loso_new`) found:

- **Doc vs. artifact divergence on every deep model.** `README.md` claimed CNNBaseline+GRL hit 89.99% accuracy / 0.9399 AUC; the model's own `aggregate_metrics.json` says 71.6% / 0.75 AUC. Same pattern for all 4 deep models and the Random Forest (claimed 77.46%/100 trees/69 features vs. actual 74.30%/50 trees/360 features). **Every number in every doc was hand-typed and drifted from the artifact that produced it.**
- **A model was being served in production with random, untrained weights** (`ssvb_casa_ais_production.pt` missing on disk; runtime silently fell back to an untrained module) — and nothing detected this.
- **Feature-set schizophrenia**: three incompatible feature counts in simultaneous use (69-dim scale-invariant vector, 360-dim windowed-stats vector, 368-dim claimed in a model card) with no single source of truth.
- Raw datasets were never committed; only pre-extracted tensors survive. Retraining from raw signal was not actually reproducible.
- 15-fold `GroupKFold` by subject was correctly leakage-free (one real thing that was right).

**This restart exists to make the three failure modes above structurally impossible, not just to retrain with better numbers.** Concretely, that means:

1. **A metric is only ever stated by being read out of a JSON/CSV artifact at report-generation time.** No person or agent types a percentage into a markdown file by hand, ever. The consolidated HTML report (Phase 5) is a template filled programmatically from `metrics.json` files — if a number isn't in an artifact, it doesn't exist.
2. **One feature-set definition, versioned, with a hash.** The extraction phase emits a `feature_manifest.json` (name, modality, count, source commit) that every later phase imports rather than redefining.
3. **No destination overwrite.** Every trained model is written to a path containing its config hash and timestamp; nothing already on disk is ever silently replaced.
4. **The old project's 91-subject combined numbers (71.6% CNN+GRL, 74.3% RF, etc.) are NOT this project's targets.** They were measured on a different dataset mix (StressID+WESAD+EmpathicSchool, 69-dim face/voice/physio) that is out of scope now. Treat them only as architectural inspiration (which model families existed, roughly what parameter budgets they used), never as a performance bar for the StressID-only restart. The real bar is the StressID paper's own Table 2 unimodal baselines (physio F1 ≈ 0.73 on all 715 tasks) — see RESEARCH_LOG.md.

---

## 1. Governing documents (do not duplicate — read and follow)

| Document | Covers | Status |
|---|---|---|
| `RESEARCH_LOG.md` | Dataset dossier (WESAD/StressID protocols, label semantics), ground-truth analysis, append-only dated log of every correction including the 2026-10-23 feature-count/evaluation-set corrections | Authoritative, append-only. Add new entries here as you go; never edit past ones. |
| `FEATURE_EXTRACTION_PROTOCOL.md` | The *what/why* of extraction: windowing rules, per-modality feature families, baseline-anchoring strategy, StressID missing-modality manifest | **Just corrected** (2026-10-23 edits) to the real 35/23/40 ECG/EDA/RESP feature counts. The remaining-feature-name gaps flagged in those edits must be closed during Phase 1 using the official extractor as ground truth. |
| `PIPELINE_IMPLEMENTATION_SPEC.md` | The *how*: repo layout, data contracts (`RawSession`, `Signal`, `Block`, `WindowSpec`), `config/subjects.yaml`, stage-by-stage gates, pinned deps | Use as-is for repo scaffolding, trimmed to StressID-only (drop WESAD-specific stage notes for this restart; keep the file itself intact for when WESAD is reintroduced later). |
| `DASHBOARD_PIPELINE_SPEC.md` | 9-page dashboard spec | **Resequenced below (§4)** — the dashboard is now explicitly built after and from the EDA notebook, not in parallel with extraction. |

Do not re-fetch the StressID paper or the StressID baseline repo. Their relevant findings are already distilled into `RESEARCH_LOG.md`.

---

## 2. Git workflow (applies to every phase below)

- One `git init` at the start of Phase 0, on a fresh, empty repo (per the user: no existing files carried over except as reference backup).
- **One commit per phase minimum**, made immediately when that phase's gate (defined per-phase below) passes — not batched at the end.
- Commit message format:
  ```
  <phase tag>: <one-line summary of what this phase produced>

  <2-4 lines: what gate it passed, what artifact/metric proves it>
  ```
  Phase tags: `phase0`, `phase1-extract`, `phase2-eda`, `phase3-dashboard`, `phase4-train:<model_name>`, `phase5-report`.
- Never force-push, never amend. If a phase's gate fails, fix forward with a new commit — the failed attempt stays in history (this project has already suffered once from undocumented silent overwrites; don't repeat it with git itself).
- Each model in Phase 4 gets its own commit (not one giant "trained all models" commit) so a single model's regression is bisectable.

---

## 3. Phase 1 — StressID extraction with calibration

**Goal:** one reproducible script (or small set of scripts) that takes raw StressID data on disk and produces per-modality, baseline-anchored feature tables, matching the official baseline's feature counts closely enough to calibrate against `all_physiological_features.csv`.

Follow `PIPELINE_IMPLEMENTATION_SPEC.md`'s stage order (Stage 0 Inventory → Stage 5 Anchoring), restricted to StressID:

1. **Stage 0 — Inventory.** Build `config/subjects.yaml` for StressID only. Resolve the outstanding subject-count discrepancy before anything else: reconcile 65 physio / 54 or 53 video / 55 or 54 audio by writing an explicit set-difference script (which subject IDs are in one list but not another) and logging the resolved, final counts as a new dated entry in `RESEARCH_LOG.md`. Do not proceed to Stage 1 with an unresolved count — this exact ambiguity is what silently dropped subjects last time.
2. **Stage 1 — Clean.** Butterworth filtering on ECG/EDA/RESP (noise + baseline-wander removal), per `FEATURE_EXTRACTION_PROTOCOL.md` §3.1–3.3, using `neurokit2` with `sampling_rate=500, method='biosppy'` to match the official baseline exactly (this is a corrected, verified detail from the full paper read — do not substitute `method='neurokit'`).
3. **Stage 2 — Window.** One window per interactive task (7 stressor tasks); sub-window only the 4 long tasks (Breathing 180s, Video1 120s, Video2 180s, Relax 150s) at 60s/30s stride per `FEATURE_EXTRACTION_PROTOCOL.md` §2.
4. **Stage 3 — Extract features, per modality, into SEPARATE tables** (`ecg_features.parquet`, `eda_features.parquet`, `resp_features.parquet`, `face_features.parquet`, `voice_features.parquet`) — never merge into one NaN-padded table keyed to the intersection of modalities. This is the exact bug that silently dropped ~12 StressID subjects in the previous project. Each table carries its own subject list; a subjects-coverage report (`coverage_report.json`) states, per modality, which of the StressID missing-modality manifest subjects (physio-only: `hh2e, wfsl, 37ir, hvpa, ql3b, r0a3, uyrl, qx2o, dmbd`; physio+audio-only: `u3v9, 7m3c`; physio+video-only: `f6q3`) were correctly retained.
5. **Column-count verification gate:** write a test that loads the official `robustml-eurecom/stressID` CSV (`all_physiological_features.csv`, already available from the earlier clone — re-clone only if it's gone) and asserts your ECG/EDA/RESP tables have 35/23/40 columns respectively, or explicitly documents and justifies any deviation in `RESEARCH_LOG.md`. This closes the "remaining ~N features" gaps flagged in the FEATURE_EXTRACTION_PROTOCOL.md edits.
6. **Stage 4 — Face/Voice.** Match the 84-dim official AU+gaze baseline (mean+SD of 18 AU intensities + mean+SD of 2 gaze vectors) per `FEATURE_EXTRACTION_PROTOCOL.md` §3.4, and voice features per §3.5 (confound-aware: only talking tasks have audio).
7. **Stage 5 — Baseline-anchor.** Robust median/MAD normalization against each subject's calm-task baseline (Breathing for ECG/EDA/face/voice; **Relax**, not Breathing, for respiration — the confound substitution already logged). Never normalize against whole-recording statistics (the exact bug found in the old `normalize_features.py`).
8. **Gate to pass before committing this phase:** (a) per-modality subject coverage matches the reconciled Stage-0 manifest exactly, with zero unexplained drift; (b) feature column counts verified against the official CSV; (c) a 3-subject visual spot-check of R-peak detection (per `FEATURE_EXTRACTION_PROTOCOL.md` §3.1's "verify before moving on") is done and a sample plot saved to `docs/verification/`.
9. Commit as `phase1-extract`.

---

## 4. Phase 2 — EDA notebook (new deliverable, build BEFORE the dashboard)

**Goal:** a single Jupyter notebook, `notebooks/01_stressid_eda.ipynb`, that is the project's full exploratory analysis of the Phase-1 output. This is the human-readable analysis that the dashboard (Phase 3) will be built to operationalize — so its structure should mirror the dashboard's planned pages but as static, narrated analysis with your own written conclusions, not interactive widgets.

Sections (in order, each ending with a written 2-4 sentence conclusion, not just a plot):

1. **Coverage & integrity** — subject/task counts per modality vs. the manifest; any missing combination flagged with a reason.
2. **Label distribution** — self-report rating histograms per task, calibration of the ≥5 "stressed" binarization threshold (is the distribution bimodal at 5, or is this a lossy cut?).
3. **Per-subject effect size** — Cohen's d between calm-baseline and each stress-task window, per modality, per subject (this is the label-validity engine from `DASHBOARD_PIPELINE_SPEC.md` Page 3 — reuse its exact definition). Subjects/tasks with near-zero effect size are candidates for exclusion or down-weighting; name them explicitly.
4. **Within-task dynamics** — for the long sub-windowed tasks, does the signal actually drift across sub-windows the way the one-window-per-task decision assumes it doesn't for short tasks? Sanity-check the windowing decision, don't just illustrate it.
5. **Feature distributions & separability ceiling** — per-feature calm-vs-stress distributions (physio, face, voice separately); a quick linear-separability check (e.g. single best feature's AUC) as a ceiling sanity check before any model is trained.
6. **Signal quality triage** — `q_ecg`/`q_eda`/`q_resp` quality-metric distributions; what fraction of windows would be excluded at which threshold.
7. **Modality complementarity** — pairwise correlation of modality-level stress scores (does face agree with physio on which windows are "stressed"?).

**Gate:** notebook runs top-to-bottom without errors on the Phase-1 output, every section has a written conclusion, and at least one concrete, actionable decision is produced for Phase 3 (e.g., "exclude subject X", "use quality threshold Y", "prioritize dashboard page Z because section 3 found the biggest open question there"). Record these decisions as a short list at the top of the notebook AND as a new dated entry in `RESEARCH_LOG.md`.

Commit as `phase2-eda`.

---

## 5. Phase 3 — Dashboard (built AFTER and FROM the EDA notebook)

Reuse `DASHBOARD_PIPELINE_SPEC.md`'s 9 pages, tech stack (Streamlit/Plotly/Matplotlib), and the "reads the same materialized parquet artifacts" principle — but with this sequencing change: **do not build all 9 pages speculatively.** Build, in order:

1. Whichever pages the EDA notebook's decision list (end of Phase 2) flagged as answering an open question — build these first and in depth.
2. The remaining pages, in the original spec's priority order (Page 0 Coverage, Page 2 Raw Signal Viewer, Page 3 Effect Size, ... per the spec's own "build first" ordering), but each page's design should cite which EDA-notebook finding motivated the view it shows, not just the generic spec description.
3. Wire the `decisions.parquet` append-only ledger from the spec so dashboard-driven decisions during later phases also feed back into `RESEARCH_LOG.md`.

**Gate:** dashboard loads Phase-1 artifacts live (no hardcoded numbers), every page traces to an EDA-notebook finding or to the spec's stated "decision this drives," and a screenshot of each built page is saved to `docs/dashboard_screenshots/` for the final HTML report.

Commit as `phase3-dashboard`.

---

## 6. Phase 4 — Model training (StressID only, LOSO, tiered)

**A second document (a "Master Audit and Phase Evaluation Report" covering the old project's 13 research phases) was reviewed on 2026-10-23.** Its headline numbers are NOT trustworthy — several match claims the first repo audit already proved were contradicted by the real artifacts (see `RESEARCH_LOG.md` 2026-10-23 entry 3). Its only use here is the list of model families actually attempted and the general shape of problems hit along the way. Do not train "everything that was ever tried" — that is the resource-wasting pattern this restart exists to avoid. Train in tiers, and stop advancing a tier if the previous tier already answers the question that tier exists to answer.

**Tier 0 — Classical baselines (run first, cheap, establishes the separability ceiling from the EDA).** Minutes, not hours, per model:
- Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Random Forest, XGBoost, LightGBM

**Tier 1 — Sequence models (only once Tier 0 numbers are sane and window-size is picked from the EDA).** Train on the single window size the EDA/dashboard decided, not all three speculatively:
- CNN-GRU (plain), GRU, LSTM, TCN, Transformer — pick 2-3 of these, not all 5, based on which the EDA's within-task-dynamics section (§4 Phase 2) suggests actually has temporal structure worth modeling

**Tier 2 — Adversarial identity suppression (only if the leakage-gap test below actually shows leakage).** Before training any GRL variant, run the exact diagnostic the old project used in its Phase 1: train the best Tier-0/Tier-1 model under (a) random row-wise split and (b) subject-LOSO split, and compute `Δ_leak = Acc_random − Acc_LOSO`. Only add a GRL identity-suppression head to a model if its own `Δ_leak` is meaningfully non-zero (the old project's threshold for "severe" was >10%; use judgment, but don't add GRL to a model that doesn't leak). If you do add GRL: sweep `λ_adv` starting well below 0.15 (the old project's own log shows λ=0.15 caused training collapse to 47% accuracy — start at λ=0.02 as they eventually settled on, and smoke-test any new λ on 1-2 folds before committing to a full run).

**Tier 3 — Fusion (only after unimodal Tier 0-2 experts are solid individually).** Start with the simplest fusion that could work (concatenation + MLP) before justifying anything fancier:
- Early concatenation + MLP → gated softmax fusion → cross-attention fusion — only build each next one if the previous one's recall/F1 is actually a bottleneck, not by default

**Tier 4 — Flagship composite (train last, only if Tier 3 isn't already sufficient).** A single cross-attention + adversarial + modality-dropout-gating architecture in the spirit of the old `SSVB-CASA-AIS`, built from scratch on the StressID-only feature set — do not reuse its old checkpoint, old feature dimensions, or old hyperparameters as anything but inspiration.

**Resource-efficiency gates that apply across every tier:**

1. **Smoke-test before full run.** Every new model/hyperparameter config gets a 1-2 fold trial first (few epochs for deep models). If accuracy is at-chance or loss diverges/NaNs, stop and fix before spending the full LOSO budget. This is the single gate that would have caught the old project's λ=0.15 GRL collapse before it burned a full 15-fold training run.
2. **Threshold tuning is a required step per model, not an afterthought.** After training, sweep the decision threshold (e.g. 0.05–0.95 step 0.05) on validation folds and report metrics at BOTH the default 0.5 threshold AND the threshold that maximizes F1 (or balanced accuracy — state which and why). Pick the deployment threshold based on the actual use case (a clinical-alert scenario favors precision; a safety-monitoring scenario favors recall) and say so explicitly in the model card — do not silently report only whichever threshold looks best.
3. **Don't advance a tier without a reason.** If Tier 0's Random Forest already reaches the StressID paper's Table 2 physio ceiling (~F1 0.73) and the EDA found no missing-modality or temporal-dynamics reason to go further, that is a legitimate stopping point — document it as such rather than training Tiers 1-4 by default.

**Non-negotiable rules for every model (these exist specifically because of audit findings §3, §5):**

1. **Protocol:** `GroupKFold` by subject, grouped on StressID's real subject count (resolved in Phase 1 Stage 0) — not 15 folds copied from the old 91-subject setup. Use leave-one-subject-out or a fold count appropriate to the real StressID subject count; state which and why.
2. **Seed:** fix `seed=42` everywhere (data split, numpy, torch) — consistent with the old project's one correct practice.
3. **Every model run writes, automatically, with no manual step:**
   - `model_card.md` — generated from a template filled by the metrics artifact, never hand-typed.
   - `metrics.json` — accuracy, precision, recall, F1, AUC, at default threshold AND at the optimal threshold found on a validation fold (report both, labeled, exactly as the audit's Results Ledger format does — this format already proved itself useful for catching discrepancies).
   - **Metric images**: confusion matrix, ROC curve, PR curve, saved as `.png` next to the metrics, named `<model>_confusion_matrix.png`, `<model>_roc.png`, `<model>_pr.png`.
   - Checkpoint/weights file, named with a content hash or timestamp — never overwrite a previous run's file at the same path (audit finding §4.2 "Model Weights Overwrite Conflict" and "Phase 3 Production Destination Overwrite" are exactly the failure mode this prevents).
4. **No README or presentation doc gets a hand-typed number, ever.** If you want a number in prose anywhere, it is pulled by a script from `metrics.json` at report-generation time (Phase 5 does this for the consolidated report; any other doc must do the same or not state the number).
5. **Report the full subject count actually used per model** (physio-only subjects can go into physio-input models even without face/voice — per the project's unimodal-capable requirement) and do not silently restrict to the all-modality intersection unless the model being trained is explicitly a fusion model, in which case state the subset size and why, exactly as the StressID paper's Table 2 vs Table 3 distinction (already logged in `RESEARCH_LOG.md`) requires.

**Gate per model:** metrics.json + model_card.md + 3 metric images exist, numbers in model_card.md are verified (by a script, not visually) to equal metrics.json, and the model's input feature count/subject count is logged and matches what Phase 1 actually produced.

Commit each model as `phase4-train:<model_name>` immediately after its gate passes — do not wait to batch all models into one commit.

---

## 7. Phase 5 — Consolidated HTML report

A single script, `scripts/build_report.py`, that:

1. Scans every model's output directory for `model_card.md` + `metrics.json` + the three metric images.
2. Renders one static, self-contained HTML file (`docs/model_zoo_report.html`) with one section per model: model name, architecture summary/hyperparameters (pulled from the model's config, not retyped), the metrics table (pulled from `metrics.json`), and the three embedded metric images (base64-inline or relative paths bundled alongside).
3. Includes a top-level comparison table across all trained models (sortable by F1/AUC) so the report answers "which model should be canonical" the way audit finding §6.4 asked the old project to.
4. Includes a short "known limitations" section auto-populated from any `RESEARCH_LOG.md` OPEN items still unresolved at report time (e.g. broken-fusion, subject-count drift) — if those are now resolved, the report should say so explicitly rather than go silent.

**Gate:** report builds with zero models reporting "N/A" for a metric that exists in their own `metrics.json` (i.e., no partial render due to a path mismatch), and every number in the report changes if you edit the underlying `metrics.json` and rerun the script — proving it's live-generated, not cached prose.

Commit as `phase5-report`.

---

## 8. Definition of done for the whole restart

- [ ] Phase 1 gate passed, subject-count discrepancy resolved and logged.
- [ ] Phase 2 EDA notebook runs clean, decisions logged in `RESEARCH_LOG.md`.
- [ ] Phase 3 dashboard built, traceable to EDA findings.
- [ ] All 9 Phase 4 models trained with full artifact sets, no hand-typed numbers anywhere.
- [ ] Phase 5 report builds and is internally consistent with every model's `metrics.json`.
- [ ] `RESEARCH_LOG.md` has a final dated entry summarizing the restart: real StressID-only numbers, any new OPEN items, and an explicit statement of which old-repo numbers (if any) are now superseded.
- [ ] Every phase has its own commit with the required attribution footer.
