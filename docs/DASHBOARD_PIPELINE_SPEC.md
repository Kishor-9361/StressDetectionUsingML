# Data Inspection Dashboard — Pipeline Specification

**Companion to:** `PIPELINE_IMPLEMENTATION_SPEC.md`, `FEATURE_EXTRACTION_PROTOCOL.md`, `RESEARCH_LOG.md`
**Role:** the instrument that converts raw and extracted data into **modelling decisions**, before any model is built.
**Version:** 1.0 — 2026-07-24

---

## 0. What this dashboard is for

It is not a results dashboard and not a demo. It has exactly one purpose:

> **Answer, from the data itself, the questions whose answers determine what you build next** — which model family, which signal processing, whether augmentation is worth anything, which window length, and which subjects and recordings to trust.

Every page below ends with **"Decision this drives."** If a view doesn't drive a decision, it isn't in the dashboard. That constraint is what keeps this from becoming the 134-markdown-file sprawl you're recovering from.

### The governing principle: the label is a hypothesis, not a fact

Per `RESEARCH_LOG.md` §A.4, no window in either dataset has a true stress label — every window inherits a block- or task-level annotation. So the dashboard's deepest job is to **test whether each label is credible for each subject**, before a model is asked to trust it. A subject who shows no physiological separation between their "calm" and "stress" recordings is either a non-responder or a data problem, and you need their ID *before* training, not after.

---

## 1. Architecture

### 1.1 Stack — deliberately boring

| Concern | Choice | Why |
|---|---|---|
| App | **Streamlit**, multipage | One user, one machine. React would be malpractice here. |
| Plots | Plotly for interactive, Matplotlib for static/exportable | Plotly for the window-size slider; Matplotlib for figures that go in the paper |
| Compute | pandas + numpy, `pyarrow` parquet | Same artifacts the pipeline writes |
| Caching | `st.cache_data` on every load and every heavy stat | The effect-size grid is expensive; compute once |
| Signal display | `neurokit2` plotting helpers | Same library the extractors use — no second interpretation |

### 1.2 Reads only — never writes to the data path

```
data/processed/
├── clean/       ← Page 2 (raw signal viewer)
├── index/       ← Page 1 (protocol), Page 6 (window sensitivity)
├── features/    ← Pages 3, 4, 5, 7, 8
├── baselines/   ← Page 3 (anchoring check)
└── manifests/   ← Page 0 (coverage, quality)
```

The dashboard's **only** output is `data/processed/manifests/decisions.parquet` — an append-only ledger of the decisions each page produced, with the timestamp and the config hash of the data it saw. That ledger feeds straight back into `RESEARCH_LOG.md`. Nothing else is written.

### 1.3 It grows with the pipeline

You do not wait for the whole extraction pipeline to finish before building the dashboard. Each page unlocks as its input artifact appears:

| After extraction stage… | These pages work |
|---|---|
| Stage 0 (inventory) | Page 0 — Coverage |
| Stage 1 (clean signals) | Page 2 — Raw signal viewer |
| Stage 2 (window index) | Page 1 — Protocol timeline · Page 6 — Window sensitivity (labels only) |
| Stage 3 (features) | Pages 3, 4, 5, 7, 8 |
| Stage 4–5 (baselines) | Page 3 — anchoring overlay |

Build Page 2 the day Stage 1 finishes. Looking at real signals early is worth more than any amount of planning.

---

## 2. Page-by-page specification

Nine pages. Ordered so that a newcomer to the data reads them top to bottom and arrives at a modelling plan.

---

### Page 0 — Coverage & Integrity

**Question:** What do I actually have, and does it match what the papers say I should have?

**Reads:** `manifests/inventory.parquet`, `config/subjects.yaml`

**Views:**
- Subject × modality availability grid (present / absent / degraded), WESAD and StressID side by side.
- Headline counts against expectation: physio 65, video 53–54, audio 54–55 for StressID; 15 for WESAD. The off-by-one you found is flagged in red until resolved.
- Missing-task matrix (subject × task), with the reason from the technical note on hover.
- Duration distribution per task — surfaces truncated recordings (`wfsl`, `h7j3` ended early).

**Decision this drives:** the exact subject pool for each unimodal experiment, and which subjects need baseline-task overrides. Written to the ledger as the canonical pools.

---

### Page 2 — Raw Signal Viewer ⭐ build this first

**Question:** Can I *see* stress in this data with my own eyes? If not, no model will find it.

**Reads:** `clean/{dataset}/{subject}/{modality}.npz`

**Views:**
- One subject, one session. ECG / EDA / RESP stacked on a shared time axis, condition/task blocks shaded behind the traces (baseline grey, stress red, amusement/relax green).
- Detected R-peaks and SCR peaks overlaid on the raw signal — this doubles as visual QA of the extractors.
- Zoom to any 10 s span. A "jump to stress onset" button and a "jump to baseline" button, so you can eyeball the same subject's two states back to back.
- Derived-signal strip beneath: instantaneous HR (from RR), tonic SCL, phasic SCR driver.

**What you are looking for:** does EDA visibly climb during the TSST? Does HR rise? For a clear responder, you will see it without statistics. For subject `wesad_s2` — the one your old diagnostics flagged — this view tells you in thirty seconds whether the "corruption" was ever real (your earlier work concluded it was a threshold artifact; confirm it by eye).

**Decision this drives:** whether a subject is a responder, a non-responder, or a data-quality casualty — the human-judgment input to Page 3's automated version.

---

### Page 3 — Per-Subject Effect Size ⭐ the label-validity engine

**Question:** For each subject, is the "stress" label physiologically real?

**Reads:** `features/{dataset}_{modality}_windows.parquet`, `baselines/`

**The core computation** — for each subject, each feature, Cohen's *d* between that subject's calm-condition windows and stress-condition windows:

```
d = (mean_stress − mean_calm) / pooled_sd
```

**Views:**
- **Heatmap:** subjects (rows) × features (columns), *d* as diverging colour. Sort rows by mean |*d*| so non-responders sink to the bottom and are immediately visible.
- Per-subject summary bar: how many features show |*d*| > 0.5, > 0.8.
- **Responder classification:** a subject with |*d*| < 0.2 on *every* physiological feature is a non-responder or a bad recording. List them by ID.
- Overlaid on Page 2's viewer: click a subject here, jump to their raw trace there.
- Anchoring check: same heatmap computed on `_raw` vs `_anch` features — anchoring should sharpen the contrast for real responders and shouldn't manufacture it for non-responders.

**Why this is the most important page.** It operationalises "the label is a hypothesis." Your earlier ad-hoc diagnostics (`m8g5`, `71i5`, `wesad_s2`) were exactly this computation, done one subject at a time, by hand, after training. This does all 80 subjects at once, before training, in one figure. Subject `71i5` (74% stress ratio vs 42% dataset average) will stand out here as a structural outlier, and you'll decide what to do about them from evidence.

**Decision this drives:**
1. Which subjects to flag, down-weight, or exclude — recorded with the *d* values that justify it, not a vibe.
2. Which features carry real signal across subjects (high mean |*d*|) vs which are noise — your first, honest feature-importance estimate, model-free.
3. Whether baseline anchoring helps (compare the two heatmaps) — this is a preview of ablation A1.

---

### Page 4 — Within-Task Dynamics ⭐ answers "peak vs. sustained"

**Question:** Within a labelled recording, is the person stressed the whole time, or only at moments? Where?

**Reads:** `features/` with per-window timestamps

**Views:**
- Align every stress recording to its onset (t = 0). Plot mean ± IQR of HR, SCL, SCR-rate across subjects over task-relative time.
- Ridgeline of a chosen feature over task time, one ridge per subject — shows heterogeneity in *when* people spike.
- **MIL preview:** for one recording, the per-window feature trace with the windows that most exceed the subject's baseline highlighted. This is a mock-up of what attention-MIL would learn to select — build it here as motivation before committing to MIL modelling.

**What you are looking for:** ramp-and-sustain vs spike-and-decay. If stress response is concentrated in a fraction of each recording, then (a) whole-recording labels are noisy in a *specific, measurable* way, (b) transition trimming is justified with a number, and (c) MIL is the right architecture rather than a fashionable one.

**Decision this drives:** transition-trim length (from data, not the default 30 s), and whether MIL is warranted. This figure goes in the paper regardless of the answer.

---

### Page 5 — Label & Rating Distributions

**Question:** How much information does the binary label throw away, and is the problem balanced?

**Reads:** `features/` label columns

**Views:**
- WESAD: minutes per condition per subject — the 53/30/17 imbalance made concrete.
- StressID: histogram of the raw 0–10 stress ratings, overall and per task, with the binarisation cut at 5 drawn on. **This shows exactly how much you destroy by binarising** — if most ratings cluster at 4–6, the binary label is nearly noise near the boundary.
- Per-subject rating range: who uses the full scale, who sits at one end. A subject who rates everything 7–9 is not comparable to one who rates everything 2–4, which is a subject-level calibration problem the binary label hides.
- Class balance per unimodal subject pool, since audio's pool is 70% stress (talking-tasks confound).

**Decision this drives:**
1. **Binary vs ordinal target.** If the rating histogram is concentrated near the cut, argue for the ordinal/regression target — and you have the figure to justify it.
2. **Whether class imbalance needs handling** (class weights, threshold tuning) — and confirms that the audio subset's imbalance is a confound, not a property of stress.

---

### Page 6 — Window-Size Sensitivity ⭐ the view you specifically asked for

**Question:** How does everything change as the window grows from 10 s to 120 s?

**Reads:** re-windowed features at multiple lengths (a small dedicated extraction over {10, 30, 60, 120} s for a subset of subjects)

**Views:**
- **Interactive slider: window length → live recompute** of (a) mean per-subject Cohen's *d*, (b) a fast per-subject linear-probe AUC, (c) feature admissibility count, (d) windows-per-subject.
- Overlay the HRV validity thresholds from Baek & Cho 2021 — shade where SDNN, LF, HF become unreliable, so you *see* the trade between statistical power (more windows) and physiological validity (longer windows).
- Two curves crossing: separability tends to rise with window length (cleaner features) while sample count falls. The crossing region is your defensible window choice, shown rather than asserted.

**What you are looking for:** confirmation that 60 s is right for WESAD, and honest evidence about the cost of StressID's 60 s tasks. If separability keeps climbing past 60 s, that is an argument that StressID's short tasks structurally limit physio performance — a real finding about the dataset, not your method.

**Decision this drives:** the final window/stride per dataset × modality, justified with a curve. And a second-order decision: if longer windows help substantially, temporal models that integrate context (sequence models) may be worth it; if not, per-window tabular models are the correct default.

---

### Page 7 — Feature Structure & Separability Ceiling ⭐ answers "which model can win"

**Question:** Before building anything, what accuracy is even reachable, and what model family fits the feature geometry?

**Reads:** `features/`

**Views:**
- **Correlation matrix** per modality, clustered. High redundancy (HRV features are heavily collinear) tells you tree ensembles or a linear model with regularisation will beat a wide MLP, and that PCA/feature selection is worth trying.
- **Linear vs non-linear separability probe:** under proper LOSO, fit (a) logistic regression and (b) a shallow gradient-boosted tree on each modality. The *gap* between them estimates how much non-linearity is actually present. A small gap means deep models will not help — which your own prior results already hinted at (RF beat every deep variant).
- **2-D projection** (PCA and UMAP) of windows, coloured by label and, separately, by subject. If windows cluster by *subject* far more strongly than by *label* (they will), that is the between-subject variance problem made visible, and it is the single strongest argument for baseline anchoring and subject-independent evaluation.
- **Per-feature univariate AUC** ranked — a model-free importance that you can compare against whatever your models later claim.

**Decision this drives:** the model shortlist. Concretely:
- Redundant, mostly-linear, tabular, few thousand rows → **gradient boosting and regularised linear models are the favourites.** Deep models must *earn* their place against them, not be assumed.
- Strong subject clustering in the projection → anchoring and LOSO are non-negotiable, and identity-invariance is worth pursuing.
- The linear-probe ceiling is your reality check: if the best probe reaches 0.75 AUC, a deep model reporting 0.95 is a leak, not a breakthrough.

---

### Page 8 — Signal Quality Triage

**Question:** Which recordings do I exclude, and why — decided before modelling, in writing?

**Reads:** `manifests/quality.parquet`, `features/` quality columns

**Views:**
- Quality-score distribution per modality per subject (ECG SQI, EDA in-range fraction, RESP breath-rate plausibility, face detection rate, VAD ratio).
- A triage table: recordings below threshold, with the failing metric and a link to Page 2 to inspect by eye before cutting.
- Impact preview: how many windows and subjects each exclusion rule removes, so you see the cost of strictness.

**Decision this drives:** the exclusion list, with a reason per recording, written to the ledger. This is the principled replacement for the previous project's after-the-fact "diagnostics."

---

### Page 9 — Modality Complementarity (multimodal subjects only)

**Question:** When two modalities disagree on the same window, which is right — and does fusion actually help?

**Reads:** joined `features/` across modalities on `(subject, task, time-overlap)`

**Views:**
- Agreement matrix: on the all-modality subset, per-window univariate-probe predictions per modality, and how often they agree vs the label.
- Cases where physio says stress and face says calm (and vice versa), pulled up for inspection — these are where fusion earns or loses.
- A preview of decision-level vs feature-level fusion value: does averaging the per-modality probe scores beat the best single modality on the shared subjects? The StressID authors found decision-level fusion won (0.72 vs 0.64 feature-level); this checks whether that holds on your extraction.

**Decision this drives:** fusion strategy — and whether multimodal is even worth the complexity over the best single modality on the shared pool. If the answer is "physio alone matches fusion," that is a clean, honest, defensible result for a unimodal-first system.

---

## 3. What the dashboard tells you about the four modelling questions

You asked specifically what this stage should reveal. Mapped explicitly:

| Your question | The page that answers it | The signal to read |
|---|---|---|
| **Which ML/DL model?** | Page 7 (separability, redundancy, linear-vs-tree gap, ceiling) | Small linear-vs-tree gap + redundant features + few rows → boosting/linear win; deep models must beat the probe ceiling |
| **What signal processing?** | Pages 2 + 8 (visual QA + quality triage) | Where artifacts live, which filters matter, what to exclude |
| **Augmentation needed?** | Pages 5 + 7 (balance + subject clustering) | See below — likely low value |
| **Which window?** | Page 6 (sensitivity sweep) | Where the separability/sample-count curves cross, bounded by HRV validity |
| **Temporal model worth it?** | Pages 4 + 6 (dynamics + window growth) | If longer windows and within-task dynamics carry signal, sequence models may help; else tabular per-window |

### On augmentation — the honest senior-engineer read

Your prior GAN augmentation moved Combined accuracy from 74.14% to 74.38% — inside the noise. That is the expected outcome, and the dashboard will show you *why*: Page 7's projection will show windows clustering by **subject**, not by class. Your bottleneck is **68 subjects**, and augmentation synthesises windows, not subjects. It cannot manufacture between-subject variance, which is the thing LOSO actually tests.

So the dashboard's likely verdict on augmentation is: **not the lever.** If anything is worth trying, it is subject-level augmentation (mixup *between* subjects, or physiologically-plausible perturbation that simulates a new individual), not window-level GAN synthesis. Page 7 will tell you whether even that is justified. Don't build augmentation until Page 7 says the geometry supports it.

---

## 4. Build order

| Step | Page | Depends on | Effort |
|---|---|---|---|
| 1 | Page 0 — Coverage | Stage 0 inventory | half day |
| 2 | Page 2 — Raw viewer | Stage 1 clean | 1 day — **build first, it's the highest-value view** |
| 3 | Page 1 — Protocol timeline | Stage 2 index | half day |
| 4 | Page 3 — Effect size | Stage 3 features | 1 day — the label-validity engine |
| 5 | Page 8 — Quality triage | Stage 3 + quality manifest | half day |
| 6 | Page 5 — Label distributions | Stage 3 | half day |
| 7 | Page 6 — Window sensitivity | multi-length extraction | 1–2 days |
| 8 | Page 7 — Separability | Stage 3 | 1 day |
| 9 | Page 4 — Dynamics | Stage 3 timestamps | 1 day |
| 10 | Page 9 — Complementarity | all modalities extracted | 1 day |

Pages 3, 6, and 7 are the three that change modelling decisions most. If time is short, those plus Page 2 are the dashboard.

---

## 5. Tests

The dashboard computes statistics, so it can be wrong. Guard the ones that drive decisions.

| Test | Assertion |
|---|---|
| `test_effect_size_sign` | On a synthetic subject with injected stress elevation, Cohen's *d* is positive and large |
| `test_probe_loso_no_leak` | The Page 7 linear probe uses subject-independent splits — no subject in train and test |
| `test_window_sensitivity_monotone_count` | Windows-per-subject decreases monotonically as window length grows |
| `test_quality_threshold_applied` | Recordings below the quality cut are excluded from the probe, not just greyed in the plot |
| `test_reads_only` | The dashboard opens no artifact in write mode except `decisions.parquet` |

The second one matters most: an EDA dashboard that previews accuracy with a leaky split will tell you the ceiling is 0.95 and send you chasing a number that doesn't exist. The probe must obey the same LOSO discipline as the real models.

---

## 6. The decision ledger — the dashboard's actual output

Every page writes its conclusion to `manifests/decisions.parquet`:

```
page             str    'effect_size' | 'window_sensitivity' | ...
decision         str    human-readable, e.g. 'exclude subject 37ir: q_ecg<0.3 all tasks'
evidence         str    the number that justifies it, e.g. 'mean|d|=0.08 across 20 features'
config_hash      str    which data version produced this
timestamp        str
```

At the end of this stage you have a table that reads like the methods section of the paper writing itself: every exclusion, every window choice, every target-variable decision, each with the evidence that produced it. That table is the antidote to the previous project's problem — no decision in it exists without a number behind it, and none of it came from an AI's say-so.

---

## 7. Open decisions this dashboard exists to resolve

Carry these into `RESEARCH_LOG.md` as the questions the dashboard must close:

1. **Which subjects are non-responders?** (Page 3) — exclude, down-weight, or keep and report separately.
2. **Binary or ordinal target?** (Page 5) — decided by how much the rating histogram concentrates near the cut.
3. **Final window/stride per dataset × modality?** (Page 6) — the crossing of the separability and sample-count curves.
4. **Model shortlist.** (Page 7) — and the linear-probe ceiling every later model is checked against.
5. **Transition-trim length.** (Page 4) — from the within-task dynamics, not the 30 s default.
6. **Is augmentation worth building?** (Pages 5 + 7) — probably not; let the geometry decide.
7. **Fusion strategy, or unimodal-only?** (Page 9) — whether multimodal beats the best single modality on the shared pool.

When these seven are answered with evidence, you are ready to model — and for the first time in this project, you'll be able to say exactly why each modelling choice was made.
