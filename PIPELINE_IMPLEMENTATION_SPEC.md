# Feature Extraction Pipeline — Implementation Specification

**Companion to:** `FEATURE_EXTRACTION_PROTOCOL.md` (the *what* and *why*) and `RESEARCH_LOG.md`
**This document is the *how*.** Repo layout, module contracts, function signatures, configs, run order.
**Version:** 1.0 — 2026-07-24
**Target:** a fresh, empty repository

---

## How to use this document

The signatures and contracts are specified. **The function bodies are yours to write.** That is deliberate — the point of the restart is that you can defend every line. If you paste a body in from a model without reading it, you have re-created the problem this project is recovering from.

Where something is genuinely fiddly and easy to get silently wrong (pickle encoding, label resampling, ectopic correction), the algorithm is spelled out. Where it is ordinary work, it isn't.

---

## 0. Design principles

Six rules. Everything below follows from them.

| # | Rule | Why |
|---|---|---|
| 1 | **Four stages, four materialised artifacts** | raw → clean signals → window index → features. Re-windowing must not require re-filtering. |
| 2 | **Window index built and inspected *before* features** | You validate your windowing for free, before spending an hour of compute on it. |
| 3 | **No silent NaN** | Every null carries a reason code. A NaN without a reason is a bug. |
| 4 | **Per-subject outputs** | Crashing on subject 40 must not lose subjects 1–39. Idempotent and resumable. |
| 5 | **Provenance on every row** | Config hash + git SHA. This is how you never again have two reports disagreeing about the same dataset. |
| 6 | **Deterministic** | Sorted iteration, pinned versions, fixed float precision. Run twice → byte-identical parquet. |

---

## 1. Repository layout

```
stress-research/
├── README.md
├── pyproject.toml                  # pinned deps — no ranges
├── RESEARCH_LOG.md                 # append-only
├── .gitignore                      # data/ is never committed
│
├── config/
│   ├── datasets.yaml               # paths, sampling rates, protocol structure
│   ├── windows.yaml                # window/stride per dataset × modality
│   ├── features.yaml               # feature list + admissibility rules
│   └── subjects.yaml               # coverage manifest, exclusions  ← §3
│
├── src/stressres/
│   ├── __init__.py
│   ├── provenance.py               # config hash, git SHA, versions
│   ├── admissibility.py            # is this feature valid for this window?
│   │
│   ├── io/
│   │   ├── wesad.py                # raw → RawSession
│   │   └── stressid.py             # raw → RawSession
│   │
│   ├── clean/
│   │   ├── ecg.py                  # filter + R-peaks + RR correction
│   │   ├── eda.py                  # filter + tonic/phasic decomposition
│   │   ├── resp.py                 # filter + breath segmentation
│   │   ├── face.py                 # OpenFace runner + frame table
│   │   └── voice.py                # resample + VAD
│   │
│   ├── quality/
│   │   └── sqi.py                  # one q_* function per modality
│   │
│   ├── windows/
│   │   ├── grid.py                 # window index construction
│   │   └── labels.py               # label alignment + transition trimming
│   │
│   ├── features/
│   │   ├── ecg.py  eda.py  resp.py  face.py  voice.py
│   │
│   └── baseline/
│       └── profiles.py             # calm-baseline anchoring
│
├── scripts/                        # thin CLI wrappers, no logic
│   ├── 00_inventory.py
│   ├── 01_clean_signals.py
│   ├── 02_build_window_index.py
│   ├── 03_extract_features.py
│   ├── 04_build_baseline_profiles.py
│   └── 05_anchor_features.py
│
├── tests/
├── notebooks/                      # exploration only — never imported by src/
│
└── data/                           # gitignored
    ├── raw/{wesad,stressid}/
    └── processed/
        ├── clean/                  # stage 1
        ├── index/                  # stage 2
        ├── features/               # stage 3
        ├── baselines/              # stage 4
        └── manifests/
```

**`notebooks/` must never be imported by `src/`.** Notebooks are for looking; the pipeline is for running. The moment pipeline logic lives in a notebook, reproducibility is gone.

---

## 2. Core data contracts

Define these in `src/stressres/types.py` before anything else. Every module speaks in these types.

```python
@dataclass(frozen=True)
class RawSession:
    """One subject's complete recording, as loaded from disk."""
    dataset: str                          # 'wesad' | 'stressid'
    subject_id: str
    signals: dict[str, Signal]            # 'ecg', 'eda', 'resp', 'bvp', ...
    blocks: list[Block]                   # protocol/task segmentation
    source_files: list[Path]

@dataclass(frozen=True)
class Signal:
    name: str
    data: np.ndarray                      # 1-D, float64
    fs: float                             # Hz
    t0: float = 0.0                       # seconds from session start
    unit: str = ''
    site: str = ''                        # 'chest' | 'wrist' | 'palm' | 'abdomen'

    @property
    def t(self) -> np.ndarray:            # sample times — never assume index==time
        return self.t0 + np.arange(len(self.data)) / self.fs

@dataclass(frozen=True)
class Block:
    """A labelled span of time. WESAD: condition. StressID: task."""
    name: str                             # 'baseline' | 'stress' | 'Speaking' | ...
    t_start: float
    t_end: float
    label_binary: int | None
    label_source: str                     # 'protocol' | 'self_report'
    ratings: dict[str, float]             # StressID 0-10 scores; {} for WESAD

@dataclass(frozen=True)
class WindowSpec:
    """A window, before any features exist. This is stage 2's output."""
    dataset: str
    subject_id: str
    task: str
    modality: str
    window_id: str
    t_start: float
    t_end: float
    label_binary: int
    label_source: str
    ratings: dict[str, float]
    is_baseline: bool                     # part of this subject's calm reference?
    admissible: frozenset[str]            # feature names valid at this length/state
```

`Signal.site` exists because WESAD chest EDA (abdomen) and StressID EDA (palm) are physically different measurements. Carrying the site prevents you from ever accidentally comparing their absolute values.

---

## 3. `config/subjects.yaml` — the coverage manifest

Ready to paste. **Verify it against your disk in step 0** and correct it there, not in code.

```yaml
stressid:
  # From: https://project.inria.fr/stressid/files/2023/11/Technical-info_final.pdf
  # VERIFY AGAINST DISK — the doc-derived counts (65/54/55) disagree with the
  # observed counts (65/53/54) by one in video and one in audio. Resolve in step 0
  # and record the resolution in RESEARCH_LOG.md.

  physio_only:            # no video, no audio
    [hh2e, wfsl, 37ir, hvpa, ql3b, r0a3, uyrl, qx2o, dmbd]
  physio_audio_only:      # no public video
    [u3v9, 7m3c]
  physio_video_only:      # participant silent, audio removed
    [f6q3]

  missing_tasks:
    wfsl:  [Relax]              # ended experiment early
    h7j3:  [Relax]              # ended experiment early
    7h5u:  [Video2]             # playback failure
    j1u8:  [Video1]             # playback failure
    k67g:  [Breathing, Video1]  # VIDEO ONLY — physio exists
    h8r2:  [Breathing]          # VIDEO ONLY — physio exists

  missing_labels:
    hh2e:  [arousal, valence]

  # Faces must NOT appear in any figure, paper, poster, or slide.
  no_image_publication:
    [dmbd, 45lx, cxj0, 5f7t, y9z6, u3v9, 7m3c, x1q3, 9j3o, c3m7, a1k9, h8s1]

  baseline_task:
    ecg:   Breathing
    eda:   Breathing
    resp:  Relax        # Breathing is a GUIDED BREATHING EXERCISE — contaminated
    face:  Breathing
    voice: null         # no calm speech exists — voice cannot be anchored

  baseline_task_overrides:
    k67g: {face: Relax}   # Breathing video missing
    h8r2: {face: Relax}   # Breathing video missing

wesad:
  subjects: [S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S13,S14,S15,S16,S17]  # no S1, no S12
  label_codes:
    0: transient        # DISCARD
    1: baseline         # non-stress
    2: stress           # STRESS — the positive class
    3: amusement        # non-stress
    4: meditation       # DECISION REQUIRED — see log
    5: ignore
    6: ignore
    7: ignore
  baseline_label: 1
  modalities: {physio_chest: all, physio_wrist: all, face: none, voice: none}
```

---

## 4. Stage 0 — Inventory

**`scripts/00_inventory.py`** · No feature extraction. Nothing else runs until this passes.

```python
def inventory(dataset: str, raw_root: Path) -> pd.DataFrame:
    """
    Walk the raw directory. One row per (subject, task, modality) that
    actually exists on disk, with file size and duration.
    Compares against config/subjects.yaml and reports every disagreement.
    """
```

Outputs `data/processed/manifests/inventory.parquet` and prints:

```
STRESSID INVENTORY
  physio : 65 subjects, 711 recordings
  video  : ?? subjects, ??? recordings     ← expect 54 per docs, you counted 53
  audio  : ?? subjects, ??? recordings     ← expect 55 per docs, you counted 54

  DISAGREEMENTS WITH MANIFEST
    <subject> <modality>  expected present, absent on disk
```

**Resolving the off-by-one.** Print the sorted set difference:

```python
docs_video   = all_subjects - set(physio_only) - set(physio_audio_only)
disk_video   = set(inventory.query("modality=='video'").subject_id)
print("in docs, not on disk:", sorted(docs_video - disk_video))
print("on disk, not in docs:", sorted(disk_video - docs_video))
```

That names the subject in one line. Then decide: incomplete download, or an extra exclusion not in the technical note. **Write the answer in the log and fix `subjects.yaml`.** Never carry an unexplained count difference forward — it will resurface as a confusing result in six weeks.

**Gate:** counts reconciled and explained.

---

## 5. Stage 1 — Clean signals

**`scripts/01_clean_signals.py --dataset wesad --subject S2`**

Raw → filtered signals + detected events, cached per subject. Nothing here knows about windows.

### 5.1 WESAD loader — `io/wesad.py`

```python
def load_wesad(pkl_path: Path) -> RawSession: ...
```

Three things that will bite you:

**(a) Python 2 pickle.**
```python
with open(pkl_path, 'rb') as f:
    d = pickle.load(f, encoding='latin1')   # omit encoding → UnicodeDecodeError
```

**(b) Wrist signals are not on the label axis.** `label` is 700 Hz, aligned to *chest*. BVP is 64 Hz, EDA/TEMP 4 Hz, ACC 32 Hz. Index alignment is wrong. Resample by time:
```python
t_label = np.arange(len(label)) / 700.0
t_wrist = np.arange(len(bvp))   / 64.0
label_wrist = label[np.searchsorted(t_label, t_wrist, side='right') - 1]
```

**(c) Blocks come from run-length encoding the label array.** Find contiguous runs of each code, convert sample indices to seconds, drop codes 0 and 5–7. Assert every retained block is at least 60 s — a shorter one means your RLE is wrong.

### 5.2 StressID loader — `io/stressid.py`

```python
def load_stressid(subject_id: str, raw_root: Path,
                  annotations: pd.DataFrame) -> RawSession: ...
```

One `Block` per task file, `t_start=0.0` (each task is its own recording), `t_end=duration`. Attach all four 0–10 ratings to `Block.ratings`. Set `label_source='self_report'`.

⚠ **Assert `Block.name` is one of the 11 known task names.** Silent filename-parsing drift is how subject/task mismatches happen.

### 5.3 ECG cleaning — `clean/ecg.py`

```python
def clean_ecg(sig: Signal) -> CleanECG:
    """Returns filtered signal, R-peak sample indices, corrected RR intervals,
    and per-interval ectopic flags."""
```

1. Butterworth bandpass 0.5–40 Hz, **`filtfilt`** (zero-phase — `lfilter` shifts your R-peaks).
2. R-peaks via `neurokit2.ecg_peaks(method='neurokit')`.
3. **Ectopic correction:**
   ```
   for each RR interval i:
       local = median(RR[i-2 : i+3])          # 5-neighbour window
       if |RR[i] - local| / local > 0.20:
           flag ectopic; replace by cubic interpolation from neighbours
   ```
4. Record `pct_ectopic` and `n_valid_rr`.

### 5.4 EDA cleaning — `clean/eda.py`

1. Lowpass 1 Hz for tonic work (5 Hz per WESAD if you want phasic detail — **state your choice in the log**).
2. **Downsample to 8 Hz.** StressID's 500 Hz EDA is ~60× oversampled; cvxEDA on 500 Hz × 39 h will run for hours.
3. `neurokit2.eda_phasic(method='cvxEDA')`. Fallback `'highpass'` if too slow — but record which, they give different SCR counts.
4. SCR peak detection, minimum amplitude 0.01 µS (state it).

### 5.5 RESP, face, voice

- **RESP:** bandpass 0.1–0.35 Hz, peak/trough detection for inhale/exhale segmentation.
- **Face:** ⚠ **OpenFace 2.0 is a compiled C++ binary, not a pip package.** Budget half a day for the build (or use a Docker image). Run `FeatureExtraction -f video.mp4 -out_dir ...` as a subprocess, parse the CSV, drop frames with `confidence < 0.8` or `success == 0`, downsample to 5 fps.
- **Voice:** resample 32 → 16 kHz, run VAD (Silero or amplitude-based), store voiced-segment boundaries. Do not window yet.

**Output:** `data/processed/clean/{dataset}/{subject}/{modality}.npz` + a `.json` sidecar with filter parameters, method names, and library versions.

**Gate:** For three subjects, plot R-peaks on raw ECG and SCR peaks on raw EDA, 10 s of baseline and 10 s of stress. Look at them. Save the figures. This is the single most valuable half hour in the project.

---

## 6. Stage 2 — Window index

**`scripts/02_build_window_index.py`**

Builds every `WindowSpec` for every subject × modality, **before any features exist**. This is the stage that catches windowing mistakes for free.

```python
def build_index(session: RawSession, cfg: WindowConfig) -> list[WindowSpec]: ...
```

### 6.1 Window rules — `config/windows.yaml`

```yaml
wesad:
  physio: {window_s: 60, stride_s: 5,  trim_onset_s: 30}
  face:   null
  voice:  null

stressid:
  physio:
    interactive: {window_s: 60, stride_s: null, trim_onset_s: 0}   # 1 window = 1 task
    long:        {window_s: 60, stride_s: 30,   trim_onset_s: 0}
    interactive_tasks: [Counting1, Counting2, Counting3, Math, Reading, Speaking, Stroop]
    long_tasks:        [Breathing, Video1, Video2, Relax]
  face:  {window_s: 10, stride_s: 5}
  voice: {window_s: 5,  stride_s: 2.5, voiced_only: true}
```

### 6.2 Label alignment — `windows/labels.py`

**WESAD:**
```python
modal_label, coverage = mode_and_coverage(labels[i0:i1])
if coverage < 0.95:  reject('transition_window')
if modal_label in {0,5,6,7}:  reject('ignored_label')
if t_start < block.t_start + 30:  reject('onset_trim')
```

**StressID:** the window inherits its task's label directly. Carry all four raw ratings through. Onset trimming is impossible on a 60 s task — record this asymmetry between the datasets as a known limitation.

### 6.3 Admissibility — `admissibility.py`

```python
def admissible_features(modality: str, duration_s: float,
                        task: str, dataset: str) -> frozenset[str]:
    """
    Which features are physiologically valid for this window?
    Grounded in Baek & Cho 2021 and Munoz 2015 — see FEATURE_EXTRACTION_PROTOCOL.md §2.
    """
```

```python
HRV_MIN_DURATION = {          # seconds, resting condition
    'mean_hr': 30, 'rmssd': 10, 'pnn50': 10, 'pnn20': 10, 'sdnn': 30,
    'lf_power': 30, 'hf_power': 10, 'lf_hf_ratio': 30,
    'vlf_power': 60, 'total_power': 60,
}
STATIC_ONLY = {'sdnn', 'hf_power', 'vlf_power', 'total_power',
               'lf_power', 'lf_hf_ratio', 'hf_norm', 'lf_norm'}

SPEAKING_TASKS = {
    'stressid': {'Speaking','Reading','Counting1','Counting2','Counting3','Math','Stroop'},
    'wesad':    {'stress'},          # TSST involves 5 min of speech
}
```

A feature excluded by admissibility is written `NaN` with `reason='inadmissible_speaking_task'` or `'inadmissible_duration'`. **It is never imputed.**

### 6.4 Inspect before proceeding

Print, per dataset × modality: window count, subject count, class balance, mean windows per subject, and the rejection-reason histogram.

**Gate:** StressID physio should land at roughly **1,500–2,500 windows across 65 subjects**. If you see 16,974, you are sub-windowing the 60 s tasks and have re-created the previous pipeline's inflation.

---

## 7. Stage 3 — Features

**`scripts/03_extract_features.py --dataset stressid --modality physio`**

Every extractor has the same signature:

```python
def extract(clean: CleanSignal, spec: WindowSpec) -> dict[str, float | None]:
    """Features for exactly one window. Pure. No I/O. No global state."""
```

Only features in `spec.admissible` are computed; the rest are `None` with a reason. Feature lists are in `FEATURE_EXTRACTION_PROTOCOL.md` §3.

**Output:** one parquet per dataset × modality, per §6 of the protocol document, plus:

```
extractor_version   str    git SHA of src/ at extraction time
config_hash         str    sha256 of the merged config
feature_reason      dict   {feature_name: reason} for every null
```

⚠ **Do not write a merged multi-modality table.** Ever. That was the schema bug that cost you twelve subjects.

**Gate:** `tests/` passes (§9), and per-modality subject counts equal the reconciled inventory.

---

## 8. Stages 4–5 — Baseline profiles and anchoring

**`scripts/04_build_baseline_profiles.py`** then **`scripts/05_anchor_features.py`**

```python
def build_profile(features: pd.DataFrame, subject_id: str,
                  modality: str, cfg) -> BaselineProfile:
    """Robust statistics over this subject's calm windows only."""
```

- Baseline task from `subjects.yaml`, honouring `baseline_task_overrides`.
- **Robust statistics only:** `median` and `MAD`. A 3-minute baseline yields ~5 windows; one artifact destroys a mean/SD.
- Require at least 3 valid baseline windows with `quality > 0.5`. Below that, emit `BaselineProfile(valid=False)` and **exclude that subject from anchored analyses**, recording why.
- StressID voice has no baseline. `voice.anchored = None`, permanently. State it as a limitation.

Anchoring:
```
x_anch = (x_raw − median_baseline) / (1.4826 * MAD_baseline + 1e-8)
```

**Emit both `_raw` and `_anch` in the same table.** The difference between them is ablation row A1 and one of your headline results.

**Gate:** for every subject, the median of `_anch` features over their *baseline* windows is ≈ 0. If not, anchoring is broken.

---

## 9. Tests

Write these **before** the extractors. `pytest tests/`.

| Test | Assertion |
|---|---|
| `test_no_subject_leakage` | No `subject_id` on both sides of any LOSO split |
| `test_baseline_train_only` | Subject S's baseline stats use only S's own calm windows — never other subjects, never S's stress windows |
| `test_coverage_matches_inventory` | Per-modality subject counts equal the reconciled step-0 inventory |
| `test_admissibility_enforced` | No frequency-domain HRV is non-null on a speaking task or a window under 60 s |
| `test_label_coverage` | No WESAD window has modal-label coverage below 0.95 |
| `test_single_label_source` | Any single table has exactly one distinct `label_source` |
| `test_no_unexplained_null` | Every null feature has an entry in `feature_reason` |
| `test_no_merged_modality_table` | No parquet under `features/` contains columns from two modalities |
| `test_deterministic` | Extraction run twice → identical sha256 |
| `test_window_counts_sane` | StressID physio window count is between 1,000 and 4,000 |

`test_deterministic` and `test_no_subject_leakage` are the two that would have prevented most of what went wrong before.

---

## 10. Run order and gates

Do not start step *n+1* until step *n*'s gate passes. Write a log entry at every gate.

| # | Command | Gate |
|---|---|---|
| 0 | `00_inventory.py --dataset stressid` | Counts reconciled; off-by-one **named and explained** |
| 1 | `01_clean_signals.py --dataset wesad` | R-peak and SCR plots inspected for 3 subjects |
| 2 | `02_build_window_index.py --dataset wesad` | ~1,900 physio windows, 15 subjects, plausible balance |
| 3 | `03_extract_features.py --dataset wesad --modality physio` | Tests pass |
| 4 | **LOSO + LDA on WESAD chest physio** | **85–93%.** ⚠ **THE GATE.** Below → information loss. Above → leak. Published: **93.12%** (LDA), 92.01% (RF). |
| 5 | Steps 1–3 for StressID physio | 65 subjects; 1,500–2,500 windows |
| 6 | LOSO on StressID physio | 0.65–0.72 expected (published 0.72 is *not* LOSO, so it's optimistic) |
| 7 | `04` + `05` — baseline profiles and anchoring | Anchored baseline medians ≈ 0 |
| 8 | Re-run LOSO on anchored features | **Report the delta. This is result A1.** |
| 9 | Face pipeline (OpenFace) | 53–54 subjects; compare vs AUs+kNN ≈ 0.69 |
| 10 | Voice pipeline | 54–55 subjects; report majority baseline alongside |

**Everything before step 4 is plumbing. Step 4 is the first moment you learn whether your code is correct.** Budget accordingly, and do not let yourself be tempted into building the face pipeline before the gate passes — that temptation is exactly how the previous version accumulated nine architectures on an unverified foundation.

---

## 11. Environment

Pin exactly. Ranges are how two runs silently disagree.

```toml
[project]
requires-python = "==3.11.*"
dependencies = [
  "numpy==1.26.4", "scipy==1.13.1", "pandas==2.2.2", "pyarrow==16.1.0",
  "neurokit2==0.2.7", "scikit-learn==1.5.0",
  "librosa==0.10.2", "soundfile==0.12.1", "torch==2.3.1",
  "matplotlib==3.9.0", "pyyaml==6.0.1", "pytest==8.2.2", "tqdm==4.66.4",
]
```

External, not pip-installable:
- **OpenFace 2.0** — compiled binary. Record the version and build date in `provenance.py`.
- **ffmpeg** — video/audio decoding.

`provenance.py` writes every library version, the OpenFace build, the git SHA, and the config hash into each output's sidecar. When a number surprises you in three months, this is what tells you which code produced it.

---

## 12. Open decisions — record the answer in the log

1. **Is WESAD meditation (label 4) non-stress, or discarded?** Changes class balance and therefore every published comparison.
2. **EDA lowpass at 1 Hz or 5 Hz?** WESAD used 5 Hz.
3. **cvxEDA or highpass decomposition?** Different SCR counts.
4. **SCR minimum amplitude threshold?** 0.01 vs 0.05 µS changes peak counts substantially.
5. **StressID: one window per interactive task, or sub-window with clustered bootstrap CIs?** The spec chooses correctness over sample count. Make this choice consciously.
6. **Binary label, or the raw 0–10 rating as an ordinal target?** Binarising at 5 discards most of what StressID actually measured. Both tables carry the raw ratings so you can revisit this without re-extracting.
