# Feature Extraction Protocol — WESAD & StressID

**Companion to:** `RESEARCH_LOG.md`
**Version:** 1.0 — 2026-07-24
**Governing constraint:** every modality must be independently usable. Multimodal is an *option*, never a requirement.

---

## 0. The constraint that determines the whole schema

The system must work when the user brings only a chest strap, or only a webcam, or only a microphone. That is not a modelling decision — it is a **data layout** decision, and getting it wrong at extraction time is unrecoverable later.

### The mistake to avoid

The previous pipeline merged everything into one 368-column table with `NaN` where a modality was absent. That layout has three fatal properties:

1. **It forces multimodal.** Any row is only complete if all modalities exist. Rows become useless the moment one sensor is missing.
2. **It silently discards subjects.** The old pipeline used **53 StressID subjects**. StressID has **65 subjects with physiological data.** Twelve subjects were dropped — almost certainly because they lack video, audio, or both. For a problem whose bottleneck is subject count, that is a 23% loss of the most valuable resource, thrown away by a schema choice.
3. **It couples window grids.** Face at 15 fps, audio at 32 kHz, and ECG at 500 Hz do not want the same window length, and forcing them to share one grid degrades all three.

### The layout to use instead

**One table per modality.** Each is independently loadable, independently trainable, independently evaluable.

```
processed/
├── wesad/
│   ├── physio_chest_windows.parquet     # 15 subjects
│   └── physio_wrist_windows.parquet     # 15 subjects
├── stressid/
│   ├── physio_windows.parquet           # 65 subjects  ← not 53
│   ├── face_windows.parquet             # 54 subjects
│   └── voice_windows.parquet            # 55 subjects
├── manifests/
│   ├── coverage.parquet                 # subject × task × modality availability
│   └── quality.parquet                  # per-window quality scores
└── baselines/
    └── baseline_profiles.parquet        # per subject × modality calm statistics
```

Every table shares a **join key** — `(dataset, subject_id, task, t_start, t_end)` — so multimodal alignment is a later join, never a precondition. Alignment happens by time overlap at fusion time, not by schema at extraction time.

### Consequence for evaluation

Each modality has a **different subject pool**. Report two numbers, always:

- **Deployment number:** LOSO over all subjects that have that modality (65 physio / 54 face / 55 voice). This is the honest performance of a unimodal deployment.
- **Comparison number:** LOSO over the common subset only. This is the only fair way to say "face beats voice," because otherwise you are comparing across different populations.

Never report just one. A reviewer will ask.

---

## 1. Raw data inventory

### 1.1 WESAD

```
WESAD/
├── S2/
│   ├── S2.pkl                # ← the only file you need
│   ├── S2_readme.txt
│   ├── S2_quest.csv          # self-report questionnaires (unused as labels)
│   ├── S2_respiban.txt       # raw chest export
│   └── S2_E4_Data.zip        # raw wrist export
├── S3/ ...
└── (S1 and S12 do not exist — 15 subjects, non-contiguous numbering)
```

`SX.pkl` is a pickled dict (Python 2 pickle — load with `encoding='latin1'`):

```python
{
  'subject': 'S2',
  'signal': {
    'chest': {                      # all 700 Hz, shape (N, ...)
      'ACC':  (N, 3),
      'ECG':  (N, 1),
      'EMG':  (N, 1),
      'EDA':  (N, 1),
      'Temp': (N, 1),
      'Resp': (N, 1),
    },
    'wrist': {                      # native rates, N differs per signal
      'ACC':  (N_32,  3),           # 32 Hz
      'BVP':  (N_64,  1),           # 64 Hz
      'EDA':  (N_4,   1),           #  4 Hz
      'TEMP': (N_4,   1),           #  4 Hz
    }
  },
  'label': (N,)                     # 700 Hz, aligned to chest
}
```

**Label codes:**

| Code | Meaning | Use |
|---|---|---|
| 0 | not defined / transient | **discard** |
| 1 | baseline | non-stress (0) |
| 2 | **stress (TSST)** | **stress (1)** |
| 3 | amusement | non-stress (0) |
| 4 | meditation | non-stress (0), or discard |
| 5, 6, 7 | should be ignored | **discard** |

The original paper's benchmark used only labels 1, 2, 3. **Decide explicitly whether meditation (4) is in your non-stress class and write the decision in the log** — it changes the class balance and therefore every accuracy number you compare against.

⚠ The wrist signals are **not** on the 700 Hz label axis. You must resample the label array to each wrist signal's rate (nearest-neighbour on the time axis), or resample the wrist signals up. Do not assume index alignment.

### 1.2 StressID

Access is credentialed — a permanent academic staff member must sign the licence and email `stressid.dataset@inria.fr`.

Recordings are split per task and named `subjectID_task`. The 11 task names: `Breathing`, `Counting1`, `Counting2`, `Counting3`, `Math`, `Reading`, `Relax`, `Speaking`, `Stroop`, `Video1`, `Video2`.

Physiological signals: ECG, EDA, RESP at **500 Hz**. Video: 720p, **15 fps**. Audio: 32 kHz, 16-bit.

Labels live in an annotation table: per `(subject, task)`, four 0–10 ratings — stress, relaxation, valence, arousal — plus the derived binary and 3-class labels.

### 1.3 StressID missing-modality manifest ⭐

Source: *Technical information*, https://project.inria.fr/stressid/files/2023/11/Technical-info_final.pdf

**Hard-code this. It is the ground truth for your coverage manifest.**

| Group | Subject IDs | Available |
|---|---|---|
| Physio only (camera malfunction) | `hh2e`, `wfsl`, `37ir`, `hvpa`, `ql3b`, `r0a3`, `uyrl`, `qx2o`, `dmbd` | physio |
| Physio + audio (no public video) | `u3v9`, `7m3c` | physio, audio |
| Physio + video (silent participant) | `f6q3` | physio, video |
| All others (53) | — | physio, video, audio |

**Resulting pools: physio 65 · video 54 · audio 55.**

Missing individual tasks:

| Subject | Missing | Reason |
|---|---|---|
| `wfsl`, `h7j3` | `Relax` | Participant ended early |
| `7h5u` | `Video2` | Video clip playback failure |
| `j1u8` | `Video1` | Video clip playback failure |
| `k67g` | `Breathing`, `Video1` (video only) | Sync issue — physio exists |
| `h8r2` | `Breathing` (video only) | Sync issue — physio exists |
| `hh2e` | arousal + valence *labels* | Participant did not answer |

⚠ `k67g` and `h8r2` are missing the **video of the Breathing task**, which is the designated calm baseline. Their **face** baseline profile must come from `Relax` instead. Handle this explicitly; do not let it fail silently.

⚠ **Publication constraint.** These subjects did not consent to their image appearing in papers or talks: `dmbd`, `45lx`, `cxj0`, `5f7t`, `y9z6`, `u3v9`, `7m3c`, `x1q3`, `9j3o`, `c3m7`, `a1k9`, `h8s1`. Never put their frames in a figure. Put this list in a constant named `NO_IMAGE_PUBLICATION`.

---

## 2. The windowing decision

This is the highest-leverage choice in the whole pipeline, and the two datasets need **different answers**. Do not use one window size for both.

### 2.1 What the HRV literature actually permits

Minimum recording length for ultra-short-term HRV to agree with the 5-minute standard:

| Feature | At rest | During movement / speech |
|---|---|---|
| RMSSD | ~10 s | **~240 s** |
| pNN50 | ~10 s | ~30 s |
| SDNN | ~30 s | **not reliable** |
| Mean RR / HR | ~30 s | not reliable |
| LF, LF/HF | ~30 s | ~30 s |
| HF | ~10 s | **not reliable** |
| VLF, Total power | ~60 s | ~120 s |

Sources: Baek & Cho (2021), *Is Ultra-Short-Term Heart Rate Variability Valid in Non-static Conditions?*, Frontiers in Physiology 12:596060; Munoz et al. (2015), *Validity of (Ultra-)Short Recordings for HRV Measurements*, PLOS ONE 10(9):e0138921.

Two things to take from this:

1. **RMSSD is the robust one.** It outperforms SDNN at every recording length. If you can only trust one HRV feature, trust RMSSD.
2. **Validity collapses under movement and speech.** This matters enormously here, because *subjects speak during the stress tasks in both datasets*. Frequency-domain HRV during a speaking task is not measuring autonomic tone — it is partly measuring respiration driven by speech.

### 2.2 The decision

| Dataset | Window | Stride | Rationale |
|---|---|---|---|
| **WESAD** | **60 s** | 5 s | Matches the original paper (60 s, following Kreibig 2010). Blocks are 10–20 min, so there is room. Frequency-domain HRV is admissible. |
| **StressID — interactive tasks** | **the whole 60 s task = 1 window** | — | The task *is* 60 s. Sub-windowing produces 10 s fragments where nothing but RMSSD survives, and manufactures fake sample counts from correlated data. |
| **StressID — Breathing (180 s), Video1 (120 s), Video2 (180 s), Relax (150 s)** | 60 s | 30 s | Long enough to sub-window |
| **Face (both datasets)** | 10 s | 5 s | Expression dynamics are fast; no HRV constraint applies |
| **Voice (both datasets)** | 5 s of *voiced* audio | 2.5 s | Gated on VAD; silence is not evidence |

**⚠ Do not sub-window StressID's 1-minute tasks for physiology.** The previous pipeline produced 16,974 physio windows from 53 subjects. With 10 s windows at 5 s stride on 60 s tasks you get ~11 near-identical windows per task, all sharing the same label and nearly the same ECG. That inflates your row count roughly tenfold and inflates any window-level bootstrap confidence interval by roughly √10. It does not add information.

**Honest count check:** StressID physio, one window per interactive task plus sub-windows on the long tasks, over 65 subjects ≈ **1,500–2,500 windows**. That is the real number. It is small. Accept it.

### 2.3 Feature admissibility by window length

Build this as a guard in code, not a comment.

```python
ADMISSIBLE = {
    'mean_hr':   {'min_s': 30, 'requires_static': False},
    'rmssd':     {'min_s': 10, 'requires_static': False},
    'pnn50':     {'min_s': 30, 'requires_static': False},
    'sdnn':      {'min_s': 30, 'requires_static': True},   # drop on speaking tasks
    'lf', 'hf', 'lf_hf', 'vlf': {'min_s': 60, 'requires_static': True},
}
SPEAKING_TASKS = {
    'stressid': {'Speaking','Reading','Counting1','Counting2','Counting3','Math','Stroop'},
    'wesad':    {'stress'},   # TSST involves speech
}
```

A feature that is inadmissible for a window is written as `NaN` **with a recorded reason**, not silently imputed.

---

## 3. Per-modality processing

Each subsection is independently implementable and independently testable. Build them in the order given.

---

### 3.1 ECG → cardiac features

**Preprocessing**
1. Bandpass 0.5–40 Hz, Butterworth, zero-phase (`filtfilt`) to avoid phase distortion.
2. R-peak detection. Default to `neurokit2.ecg_peaks(method='neurokit')`; Pan-Tompkins is a reasonable cross-check.
3. **RR-interval correction — do not skip this.** Mark an interval ectopic if it deviates more than 20% from the median of its 5 neighbours. Interpolate corrected intervals cubically. Record `pct_ectopic` per window.
4. Reject the window if `pct_ectopic > 0.05` or if fewer than 20 valid RR intervals were found.

**Features** (~20, all admissibility-gated)

| Family | Features |
|---|---|
| Rate | `mean_hr`, `sd_hr`, `min_hr`, `max_hr`, `mean_rr` |
| Time-domain HRV | `rmssd`, `sdnn`, `pnn20`, `pnn50`, `sdsd`, `cvnn`, `tinn`, `hti` |
| Frequency-domain | `lf_power`, `hf_power`, `lf_hf_ratio`, `lf_norm`, `hf_norm`, `total_power` |
| Non-linear | `sd1`, `sd2`, `sd1_sd2`, `sample_entropy` |

**Quality metric** `q_ecg ∈ [0,1]`: combine (a) `1 − pct_ectopic`, (b) fraction of the window with a detected peak within physiological range (30–200 bpm), (c) a template-matching signal quality index. `q = 0` if any component fails hard.

**Verify before moving on:** plot one subject's detected R-peaks over the raw ECG for 10 seconds of baseline and 10 seconds of TSST. If the peaks are wrong, everything downstream is noise. Do this with your eyes, once, for at least three subjects.

---

### 3.2 EDA → electrodermal features

⚠ **Measurement sites differ.** WESAD chest EDA is on the **abdomen** (rectus abdominis); WESAD wrist EDA is on the wrist; StressID EDA is on the **palm**. Palmar EDA is substantially more reactive than either. These are three different signals and their absolute values are not comparable across datasets — another reason not to pool.

**Preprocessing**
1. Lowpass 1–5 Hz (5 Hz per WESAD; 1 Hz is common for tonic work).
2. Downsample to 4–10 Hz. StressID's 500 Hz EDA is heavily oversampled — EDA has no meaningful content above ~2 Hz.
3. Decompose into tonic (SCL) and phasic (SCR). Use `neurokit2.eda_phasic(method='cvxEDA')`; `'highpass'` is a faster fallback. **Record which method you used** — they give different SCR counts.
4. Detect SCR peaks with a minimum amplitude threshold (0.01–0.05 µS; state your choice).

**Features** (~15)

| Family | Features |
|---|---|
| Tonic | `mean_scl`, `sd_scl`, `slope_scl`, `range_scl`, `corr_scl_time` |
| Phasic | `mean_scr`, `sd_scr`, `auc_scr` |
| SCR events | `n_scr_peaks`, `scr_rate_per_min`, `mean_scr_amplitude`, `sum_scr_amplitude`, `mean_scr_risetime`, `mean_scr_recovery` |
| Raw | `mean_eda`, `min_eda`, `max_eda`, `range_eda`, `slope_eda` |

**Quality metric** `q_eda`: fraction of samples inside 0.01–100 µS, rate-of-change plausibility (EDA cannot jump instantly), and — where accelerometer data exists (WESAD) — motion artifact fraction. Flat-line segments (electrode detachment) → `q = 0`.

---

### 3.3 Respiration → breathing features

⚠ **Read the confound warning before using these features.** In WESAD, `σ` of exhalation duration is the single most important feature in the published benchmark (Gini importance 0.35 in the binary task), and the paper's own authors note that classifiers may have partly learned *speaking vs. not speaking*, since subjects spoke during the TSST and were silent during baseline. In StressID, the calm baseline task is a **guided breathing exercise**, which deliberately manipulates respiration.

**Therefore:**
- Always report results **with and without** respiration features. The delta is a finding, not an inconvenience.
- For StressID, anchor respiration features against **`Relax`**, not `Breathing`.

**Preprocessing**
1. Bandpass 0.1–0.35 Hz (WESAD's choice — 6 to 21 breaths/min).
2. Peak/trough detection to segment inhalation and exhalation.

**Features** (~15): `resp_rate`, `mean_inhale_dur`, `sd_inhale_dur`, `mean_exhale_dur`, `sd_exhale_dur`, `ie_ratio`, `stretch_range`, `inspiration_volume`, `resp_duration`, plus RRV time-domain features (`rrv_rmssd`, `rrv_sdbb`) and `resp_sd`, `resp_range`.

**Quality metric** `q_resp`: breath count in physiological range, absence of clipping, peak-detection confidence.

---

### 3.4 Face → action units and gaze

Match the StressID baseline so your numbers are comparable to theirs.

**Preprocessing**
1. Downsample video to **5 fps** (StressID's baseline choice — you have 15 fps native).
2. Run **OpenFace 2.0** (Baltrušaitis et al. 2018) per frame.
3. Extract the AU set used in the StressID baseline: **AU 1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45** — both presence (binary) and intensity (0–5). Plus two per-eye gaze direction vectors.
4. Drop frames where OpenFace confidence is below 0.8 or no face was detected.

**Features per 10 s window**
- Mean and SD of each AU intensity → 36
- Mean presence rate of each AU → 18
- Mean and SD of gaze vectors → 12
- Head pose (pitch/yaw/roll) mean, SD, and velocity → 18
- Blink rate from AU45, `mean_ear`, `sd_ear`
- **Temporal derivatives**: mean |Δ| of AU intensity between frames → 18. Velocity matters more than absolute geometry and is naturally more identity-invariant.

**Quality metric** `q_face`: fraction of frames with a detected face at confidence ≥ 0.8, bounding-box stability, illumination variance. `q = 0` below 60% detection.

---

### 3.5 Voice → acoustic features

⚠ **The strongest confound in the whole project.** In StressID, audio exists only for the 7 talking tasks, and those tasks are **70% labelled stress**. A voice model trained naively will partly learn *"is this a talking task"* rather than *"is this person stressed."* WESAD has no audio at all, so there is no cross-dataset check available.

**Mitigations, all of which you should implement:**
- Report the majority-class baseline for the audio subset explicitly and prominently.
- Evaluate voice on the **talking tasks only**, where a within-task-type comparison is possible.
- Report per-task-type breakdown, not just a pooled number.

**Preprocessing**
1. Downsample 32 kHz → 16 kHz.
2. **Voice activity detection** — amplitude-based per the StressID baseline, or Silero VAD. Discard non-speech. Record `vad_ratio`.
3. Window into 5 s of voiced audio.

**Two parallel feature sets — build both, they are your ablation**

*A — handcrafted (~140 dims, matching the StressID baseline):* MFCCs plus Δ and ΔΔ; spectral centroid, bandwidth, contrast, flatness, roll-off; harmonic and percussive components with tonal centroid on the harmonic part; zero-crossing rate; tempogram ratio. Mean and SD over time for each.

*B — pretrained embedding:* Wav2Vec 2.0, features every 20 ms, mean-pooled to a single embedding per window. This is the one place where a large pretrained model clearly earns its keep, and the StressID authors found it beat handcrafted features on accuracy (0.66 vs 0.60).

**Quality metric** `q_voice`: `vad_ratio`, SNR estimate, clipping fraction. `q = 0` when `vad_ratio < 0.2`.

---

## 4. Label alignment

### WESAD
Labels are per-sample at 700 Hz. For a 60 s window:
- Take the modal label across the window.
- **Reject the window if the modal label covers less than 95% of it** — that is a transition window and it is contaminated.
- Reject any window whose modal label is 0, 5, 6, or 7.

### StressID
Labels are per `(subject, task)`. Every window inherits its task's label. Also carry the **raw 0–10 ratings** into the table, not just the binary — you will want the ordinal target later, and re-extracting to get it back would be painful.

### Transition trimming (both datasets)
Discard the first **30 s** of each condition block. Autonomic response to a stressor onset is not instantaneous, and the opening seconds of a "stress" block are physiologically still baseline.

For StressID's 60 s tasks this is impossible — the task *is* 60 s. Record this asymmetry in the log; it is a real limitation and a reviewer will spot it if you don't.

---

## 5. Baseline anchoring

Per §A.4 of the research log, every feature is expressed as a deviation from that subject's own calm state. The baseline source must be chosen **per dataset and per modality**.

| Dataset | Modality | Baseline source | Why |
|---|---|---|---|
| WESAD | all physio | `label == 1` (baseline block, 20 min) | Neutral reading, long, clean |
| StressID | ECG, EDA | `Breathing` task | Designated neutral, 3 min |
| StressID | **RESP** | **`Relax`** | `Breathing` is a *guided breathing exercise* — it manipulates the exact signal you'd be normalising |
| StressID | Face | `Breathing`, falling back to `Relax` for `k67g` and `h8r2` | Their Breathing video is missing |
| StressID | Voice | **none available** | No speech in any calm task. Voice cannot be baseline-anchored on this dataset — state this as a limitation |

**Use robust statistics**, not mean and SD:

```
x_anchored = (x − median_baseline) / (1.4826 × MAD_baseline + ε)
```

A 3-minute baseline yields few windows, and one artifact would destroy a mean/SD normalisation.

Write baseline profiles to `baselines/baseline_profiles.parquet` as first-class rows — they are needed at inference, and they are the calibration set for the conformal layer.

**Emit both the raw and the anchored feature in the same table** (`hr_mean_raw`, `hr_mean_anch`). The comparison between them is ablation row A1 and it is one of your main results.

---

## 6. Output schema

Identical structure for every modality table:

```
dataset          str      'wesad' | 'stressid'
subject_id       str      'S2' | '71i5'
task             str      'baseline' | 'Speaking' | ...
window_id        str      f'{dataset}_{subject_id}_{task}_W{i}'
t_start          float    seconds from session start
t_end            float
modality         str      'ecg' | 'eda' | 'resp' | 'face' | 'voice'
<features...>    float    both _raw and _anch variants
quality          float    [0,1]
quality_reason   str      '' when q == 1.0
label_binary     int      0 | 1
label_stress_raw float    StressID 0–10; null for WESAD
label_arousal    float    StressID only
label_valence    float    StressID only
label_relax      float    StressID only
label_source     str      'protocol' | 'self_report'    ← never let these mix silently
extractor_version str     git commit of the extraction code
```

`label_source` is not optional. It is the field that stops you from accidentally re-pooling two incompatible label definitions six weeks from now.

---

## 7. Verification tests — write these before the extractors

Put them in `tests/`. They must pass before any modelling.

| Test | Assertion |
|---|---|
| `test_no_subject_leakage` | No `subject_id` appears in both sides of any LOSO split |
| `test_baseline_train_only` | Baseline statistics for subject S are computed only from S's own calm windows, never from other subjects or from S's stress windows |
| `test_coverage_matches_manifest` | Subject counts per modality equal 65 / 54 / 55 for StressID and 15 for WESAD |
| `test_window_admissibility` | No frequency-domain HRV feature is non-null on a window shorter than 60 s or on a speaking task |
| `test_label_alignment` | No WESAD window has a modal-label coverage below 0.95 |
| `test_no_label_source_mixing` | Any table used for a single training run has exactly one distinct `label_source` |
| `test_quality_monotone` | Windows with `quality == 0` are excluded from every training set |
| `test_deterministic` | Running extraction twice on the same input produces byte-identical parquet |

The last one matters more than it looks. Non-determinism is how you get two reports claiming 0.9727 and 0.7694 for the same dataset.

---

## 8. Execution order

Do not parallelise this. Each step is verified before the next begins.

| Step | Work | Verification gate |
|---|---|---|
| 1 | Load WESAD `.pkl`, plot ECG + EDA for S2 with condition shading | You can *see* the TSST in the EDA trace |
| 2 | ECG extractor, WESAD only | R-peaks visually correct on 3 subjects |
| 3 | EDA extractor, WESAD only | SCR peaks visually correct on 3 subjects |
| 4 | RESP extractor, WESAD only | Breath rate is 6–21/min for every window |
| 5 | Assemble `wesad/physio_chest_windows.parquet` | ~1,900 windows at 60 s; class balance matches the protocol |
| 6 | **Reproduce the published benchmark: LOSO + LDA on chest physio** | **Land in 85–93%. This is the gate.** Below → information loss. Above → leak. |
| 7 | Port the same extractors to StressID physio | 65 subjects present in the output |
| 8 | LOSO on StressID physio | Expect 0.65–0.72; published (non-LOSO) is 0.72 |
| 9 | Face extractor → `face_windows.parquet` | 54 subjects; compare against AUs+kNN ≈ 0.69 |
| 10 | Voice extractor → `voice_windows.parquet` | 55 subjects; report the majority baseline alongside |
| 11 | Baseline profiles for every subject × modality | Anchored features have per-subject median ≈ 0 on baseline windows |
| 12 | Coverage + quality manifests | Dashboard V6 renders from them |

**Step 6 is the whole point of the first two weeks.** Until you reproduce a published number with your own code, nothing you measure afterwards means anything.

---

## 9. References

**Datasets**
- Schmidt, P. et al. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. *ICMI '18*, 400–408. https://doi.org/10.1145/3242969.3242985
- Chaptoukaev, H. et al. (2023). StressID: a Multimodal Dataset for Stress Identification. *NeurIPS 2023 Datasets and Benchmarks.* https://openreview.net/forum?id=qWsQi9DGJb
- StressID technical information (missing-modality manifest). https://project.inria.fr/stressid/files/2023/11/Technical-info_final.pdf
- StressID baseline code. https://github.com/robustml-eurecom/stressID

**HRV methodology**
- Task Force of the ESC and NASPE (1996). Heart rate variability: standards of measurement, physiological interpretation, and clinical use. *European Heart Journal* 17:354–381.
- Shaffer, F. & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health* 5:258.
- Baek, H. J. & Cho, J. (2021). Is Ultra-Short-Term Heart Rate Variability Valid in Non-static Conditions? *Frontiers in Physiology* 12:596060. https://doi.org/10.3389/fphys.2021.596060
- Munoz, M. L. et al. (2015). Validity of (Ultra-)Short Recordings for Heart Rate Variability Measurements. *PLOS ONE* 10(9):e0138921. https://doi.org/10.1371/journal.pone.0138921

**EDA**
- Boucsein, W. (2012). *Electrodermal Activity*, 2nd ed. Springer.
- Greco, A. et al. (2016). cvxEDA: A convex optimization approach to electrodermal activity processing. *IEEE TBME* 63(4):797–804.

**Emotion physiology**
- Kreibig, S. D. (2010). Autonomic nervous system activity in emotion: A review. *Biological Psychology* 84(3):394–421. — *the source of the 60-second window convention.*

**Tooling**
- Makowski, D. et al. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods* 53:1689–1696.
- Baltrušaitis, T. et al. (2018). OpenFace 2.0: Facial behavior analysis toolkit. *IEEE FG 2018*, 59–66.
- Baevski, A. et al. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *NeurIPS* 33:12449–12460.

**Stress induction**
- Kirschbaum, C., Pirke, K.-M. & Hellhammer, D. H. (1993). The Trier Social Stress Test. *Neuropsychobiology* 28(1-2):76–81.
