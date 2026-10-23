# StressID Model Training — Combine + Train Spec for the Coding Agent

**Sequencing note:** this runs BEFORE the dashboard (Phase 3 of `RESTART_PIPELINE_SPEC.md` is deferred until after this document). See `RESEARCH_LOG.md` 2026-10-23 entry 4 for why. Governing docs: `RESEARCH_LOG.md`, `FEATURE_EXTRACTION_PROTOCOL.md`, `RESTART_PIPELINE_SPEC.md` §6-7 (tiering/threshold-tuning rules already defined there — this document makes them concrete and executable; don't re-derive them).

**Before starting:** resolve the two open items from the EDA notebook review (RESEARCH_LOG.md entry 4) — specifically confirm `j1u8`'s real modality coverage against `coverage_report.json` rather than the notebook's prose. A wrong physio-only count here means a wrong subject pool for the physio-only Tier 0 models below.

---

## Part A — Combine (do this first; it is new, not previously specced)

**Goal:** one joined view per legitimate training scope, built from the 5 separate per-modality calibrated tables (`ecg_features.parquet`, `eda_features.parquet`, `resp_features.parquet`, `face_features.parquet`, `voice_features.parquet`), WITHOUT repeating the old project's bug of collapsing everything into one all-modality-intersection table that silently drops physio-only subjects.

1. **Keep the 3 physio tables combined into one `physio_combined.parquet`** (ECG+EDA+RESP are keyed identically — same subject/task/window — so this join is lossless and always valid; no subject is ever dropped by this merge since every subject has all 3 physio signals per the Stage-0 manifest).
2. **Build scope-specific joined views, each named for exactly what it contains — never one generic "all_features" table:**
   - `view_physio_all.parquet` — physio_combined, all ~65 subjects. This is the input for any unimodal-physio model and the largest-N training scope.
   - `view_physio_face.parquet` — physio_combined + face, inner-joined on subject (the ~53+1 video-covered subjects). Use for face-inclusive bimodal models.
   - `view_physio_voice.parquet` — physio_combined + voice, inner-joined (the ~54+2 audio-covered subjects).
   - `view_tri_modal.parquet` — physio_combined + face + voice, inner-joined on the subjects with all three (the ~52-53 "talking task" subjects). This is the ONLY view fusion/Tier-3-4 models may train on, and it must be explicitly named as a reduced subset in every model card that uses it — mirroring the StressID paper's own Table 2 (all tasks) vs Table 3 (370-subset) distinction already logged.
3. **Write `combine_manifest.json`**: for each view, the exact subject list, row count, and which per-modality source tables it drew from. This is the single place a reviewer checks "was this model trained on the subject pool I think it was."
4. **Gate:** `view_physio_all.parquet` row count / subject count matches the Stage-0 resolved manifest exactly (no silent drop); each narrower view's subject list is a proper subset of `view_physio_all`'s, and is cross-checked against `coverage_report.json`, not re-derived by eye.
5. Commit as `phase4a-combine`.

---

## Part B — Training (executes the tiers already defined in RESTART_PIPELINE_SPEC.md §6)

### B.0 Shared training infrastructure (build once, reuse for every model)

- `src/stressres/train/cv.py`: `GroupKFold` by `subject_id`, fold count = the resolved subject count from Part A step 4 (true LOSO means n_splits == n_subjects for that view; if that's computationally wasteful for deep models, use a fixed reduced fold count — e.g. 10 — and say explicitly in every model card that this is "subject-grouped k-fold," not strict LOSO, so no one mistakes it for the stricter protocol).
- `src/stressres/train/threshold.py`: given validation-fold probabilities + labels, sweep thresholds 0.05–0.95 step 0.05, return `{threshold: {accuracy, precision, recall, f1, balanced_accuracy}}`. Called identically by every model's training script — this is the single implementation threshold-tuning uses project-wide, never reimplemented per model.
- `src/stressres/train/smoketest.py`: given a model-training function and 2 folds (or a 1-epoch/low-epoch run for deep models), check (a) no NaN/Inf loss, (b) accuracy is meaningfully above the majority-class baseline (~63% per the EDA's class balance). Raise/abort if either fails — this must run and pass BEFORE any full-fold or full-epoch training call for ANY model, no exceptions.
- `src/stressres/train/artifacts.py`: given a trained model + its fold predictions, writes `metrics.json` (both default-0.5 and optimal-F1-threshold metrics, labeled), 3 metric images (`confusion_matrix.png`, `roc.png`, `pr.png`), and renders `model_card.md` from a template filled from `metrics.json` — never by hand. Also writes the checkpoint to a path containing a content hash + timestamp (e.g. `models/<model_name>/<run_hash>/`), never overwriting a prior run at the same path.

### B.1 Tier 0 — Classical baselines (run first)

| Model | Input view | Key hyperparameters to grid (keep small — this tier is meant to be cheap) |
|---|---|---|
| Logistic Regression | `view_physio_all` | `C ∈ {0.01, 0.1, 1, 10}`, `class_weight='balanced'` |
| K-Nearest Neighbors | `view_physio_all` | `n_neighbors ∈ {5, 11, 21}`, `weights ∈ {uniform, distance}` |
| SVM (RBF) | `view_physio_all` | `C ∈ {1, 10}`, `gamma ∈ {scale, 0.01}`, `class_weight='balanced'` (watch for the old project's SVM recall-collapse at 10s windows — if recall is nearly 0, that's this exact failure mode, not a bug in your code) |
| Random Forest | `view_physio_all` | `n_estimators ∈ {100, 300}`, `max_depth ∈ {None, 12}`, `class_weight='balanced'` |
| XGBoost | `view_physio_all` | `max_depth ∈ {4, 6}`, `n_estimators ∈ {200}`, `scale_pos_weight` set from the actual class ratio (~1.7, from the EDA's 63/37 split — not hand-guessed) |
| LightGBM | `view_physio_all` | mirror XGBoost's grid |

- Repeat the Random Forest and XGBoost runs on `view_physio_face`, `view_physio_voice`, and `view_tri_modal` too (small extra cost, and it's the data you need for Part A's view validation and for the Tier-3 fusion comparison later) — but report these as explicitly separate model cards per view, never averaged together.
- **Gate:** every Tier 0 model has passed its smoke test, has full artifacts per B.0, and the best Tier 0 physio-all result is compared against the StressID paper's own Table 2 physio ceiling (F1 ≈ 0.73) in its model card — state the gap, don't just report your own number in isolation.
- Commit each model as `phase4-train:<model_name>-tier0`.

### B.2 Tier 1 — Sequence models (only after Tier 0 lands and window size is settled)

Window size: the EDA notebook validated the existing 60s/30s task-structured windowing rather than running a 2s/5s/10s sweep (see RESEARCH_LOG.md entry 4) — proceed with 60s windows for sequence models unless you have reason to revisit that.

Pick 2-3 from `{CNN-GRU, GRU, LSTM, TCN, Transformer}` based on which has the most support from the EDA's §4 within-task-dynamics finding (HR stays stable within a task → there isn't strong evidence of rich temporal structure to exploit, which argues for starting with the cheapest sequence model, e.g. a plain GRU, rather than defaulting to the heaviest). Document that reasoning in `RESEARCH_LOG.md` before picking.

- Same `GroupKFold`-by-subject protocol, same `view_physio_all` scope (sequence models can also run on `view_tri_modal` as a second configuration, same rule as Tier 0 about reporting separately).
- Smoke-test with 1-2 epochs on 1-2 folds before any full run.
- Commit each as `phase4-train:<model_name>-tier1`.

### B.3 Tier 2 — Adversarial identity suppression (conditional — run the leakage-gap test first)

1. Take the best Tier 0/1 model. Train it once under random row-wise k-fold (NOT grouped by subject) and once under the normal subject-grouped protocol. Compute `Δ_leak = Acc_random − Acc_LOSO`.
2. **If `Δ_leak` is small (roughly <5%), stop here — do not build a GRL variant.** Log this finding in `RESEARCH_LOG.md` and skip to Tier 3/4 planning.
3. **If `Δ_leak` is large, add a GRL subject-identity head** to that model. Start the adversarial weight at `λ_adv = 0.02` (not higher — the prior project's own log shows 0.15 caused training collapse to 47% accuracy). Smoke-test any new λ value on 1-2 folds before a full run.
4. Commit the leakage-gap test itself (`phase4-train:leakage-gap-test`) regardless of outcome — a negative result (no leakage) is still a result worth having on record.

### B.4 Tier 3 — Fusion (conditional — run modality-complementarity check first)

The EDA already found low-to-moderate cross-modality correlation (r 0.12–0.35), which is the actual argument for attempting fusion — proceed with Tier 3, but in order of complexity, stopping as soon as a simpler approach stops leaving an obvious gap:
1. Early concatenation + MLP on `view_tri_modal`.
2. If recall is still a bottleneck (check against Tier 0/1's recall numbers): gated softmax fusion.
3. If modality-specific attention still looks like it would help (check per-modality error patterns, not just aggregate F1): cross-attention (CASA) fusion.
- Every fusion model's card must state the `view_tri_modal` subject count explicitly and compare against the matched-subset unimodal baseline from B.1 (never against the full `view_physio_all` baseline — that is exactly the StressID Table 2 vs Table 3 mistake already logged and corrected).

### B.5 Tier 4 — Flagship composite (last, conditional)

Only build if Tier 3's best fusion result still leaves a gap vs. the StressID paper's own Table 3 decision-fusion number (F1 ≈ 0.72 on the matched subset). Built from scratch on this project's StressID-only feature set; the old `SSVB-CASA-AIS` is architectural inspiration only (cross-attention + GRL + modality-dropout gating), not a checkpoint or hyperparameter source.

---

## Part C — After this document

Once Tier 0 (minimum) through whichever tier was actually reached are done with full artifacts:
1. Build the consolidated HTML report (Phase 5 of `RESTART_PIPELINE_SPEC.md` — unchanged by this reorder).
2. THEN build the dashboard (Phase 3 of `RESTART_PIPELINE_SPEC.md`), which can now also surface training results (per-model, per-view metrics) alongside the EDA findings it was already going to show — this is a natural benefit of training before the dashboard, worth keeping even though it wasn't the original plan.
3. Log a final `RESEARCH_LOG.md` entry naming which tiers were reached, the leakage-gap test's outcome, and the single best model + its honest, artifact-verified metrics.
