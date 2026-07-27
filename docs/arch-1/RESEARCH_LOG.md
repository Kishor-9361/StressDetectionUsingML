# Research Log — Multimodal Stress Detection

**Owner:** Kishor
**Started:** 2026-07-24
**Rule:** Append only. Never edit a past entry — if you were wrong, write a new entry saying so.
**Rule:** Every claim in this file must have either (a) a paper citation, or (b) a script in `verify/` that reproduces it.

---

# PART A — Dataset Dossier

This is the reference section. It gets updated only when a paper says something I misread.

---

## A.1 WESAD

**Citation**
Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI '18, Boulder, CO, pp. 400–408.
DOI: https://doi.org/10.1145/3242969.3242985
PDF: https://ubi29.informatik.uni-siegen.de/usi/pdf/ubi_icmi2018.pdf
Data: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
Mirror: UCI ML Repository, https://doi.org/10.24432/C57K5T

### Participants
- 17 recruited, **15 retained** (2 discarded for sensor malfunction)
- Mean age 27.5 ± 2.4; **12 male, 3 female**
- Graduate students at a corporate research facility
- Exclusions: pregnancy, heavy smoking, mental disorders, chronic/cardiovascular disease
- Instructed: no caffeine or tobacco 1 h before; no strenuous exercise that day

### Sensors
| Device | Placement | Signals | Rate |
|---|---|---|---|
| RespiBAN Professional | Chest | ECG, EDA, EMG, RESP, TEMP, ACC | **700 Hz (all channels)** |
| Empatica E4 | Non-dominant wrist | BVP | 64 Hz |
| | | EDA | 4 Hz |
| | | TEMP | 4 Hz |
| | | ACC | 32 Hz |

Notable placements: chest EDA is on the **rectus abdominis** (abdomen), not the hand. TEMP on the sternum. EMG on the upper trapezius, both sides. ECG is standard 3-point.

### Protocol (~2 hours total)
| Block | Duration | Content |
|---|---|---|
| Baseline | ~20 min | Sitting/standing, reading neutral magazines |
| Amusement | 392 s | 11 funny video clips, 5 s neutral gap between each |
| **Stress (TSST)** | ~10 min | 5 min public speech on personal strengths/weaknesses to a 3-person "HR panel" (cover story), 3 min prep, no notes; then 5 min mental arithmetic — count down from 2023 in steps of 17, restart on any error |
| Rest | 10 min | After TSST |
| Meditation ×2 | 7 min each | Guided breathing, eyes closed, seated |

Two protocol orders (A: baseline→amusement→medi→stress→rest→medi; B: baseline→stress→rest→medi→amusement→medi) to counterbalance order effects. Roughly half the subjects sat and half stood for baseline/amusement/stress.

### Label provenance — **protocol-based**
Self-reports were collected at 5 timepoints: PANAS (20 items), STAI (6 items), SAM (valence/arousal), SSSQ (9 items, after TSST only).

**But the labels used for the benchmark are the experimental condition, not the self-reports.** The paper states plainly that for its evaluation it treated the study protocol as ground truth. The questionnaires were used to *validate that the manipulation worked*, and the authors note the self-reports could be used for personalised models as future work.

Manipulation check (baseline → stress): STAI 10.8±1.9 → 18.5±2.0; SAM arousal 2.5±0.9 → 6.8±1.8. Confirmed by Wilcoxon signed-rank.

SSSQ breakdown after TSST: **Engagement 11.7±2.3, Worry 10.6±2.3, Distress 6.0±2.9.** Read that again — subjects were more *engaged and worried* than *distressed*. "Stress" here is closer to high-arousal task engagement than to suffering.

### Windowing used by the authors
- Physiological features: **60-second windows**, 0.25 s shift
- ACC features: 5-second windows
- ~36 min usable data per subject → ~133,000 windows total
- Class balance: 53% baseline, 30% stress, 17% amusement

### Published benchmark — MY CORRECTNESS TARGETS
LOSO cross-validation, binary (stress vs. non-stress):

| Input | Best classifier | Accuracy |
|---|---|---|
| **All chest physio** | LDA | **93.12%** |
| All chest physio | RF | 92.01% |
| All wrist physio | RF | 88.33% |
| Chest RESP alone | LDA | 88.09% |
| Chest ECG alone | LDA | 85.44% |
| Wrist BVP alone | LDA | 85.83% |
| Chest EDA alone | LDA | 81.70% |
| Chest TEMP alone | LDA | 69.49% |
| Majority-class guesser | — | 69.94% |

Three-class best: 80.34% (chest physio, AdaBoost).
Subject-specific binary accuracy ranged **82%–100%**.

> **If my LOSO physio number is not in the 85–93% band, I have a bug or a leak. Above 93% = leak. Below 85% = information loss.**

### ⚠ Confounds I must control for
1. **Speech confound.** Subjects spoke during TSST and were silent during baseline. RESP is the single strongest feature (importance 0.35 for `σ_RESP,E`), and the authors themselves flag that classifiers may have partly learned speaking vs. non-speaking. Any respiration-driven result on WESAD is suspect.
2. **Posture confound.** Standing vs. sitting varies between subjects and conditions. Affects HR, HRV, and EDA baseline.
3. **The 69.94% floor.** A model that always predicts "non-stress" scores 69.94%. Any accuracy near 70% is worthless.
4. **Only 3 female subjects.** No gender generalisation claims possible.

---

## A.2 StressID

**Citation**
Chaptoukaev, H., Strizhkova, V., Panariello, M., D'Alpaos, B., Reka, A., Manera, V., Thümmler, S., Ismailova, E., Evans, N., Bremond, F., Todisco, M., Zuluaga, M. A., & Ferrari, L. M. (2023). *StressID: a Multimodal Dataset for Stress Identification.* NeurIPS 2023 Datasets and Benchmarks Track.
PDF: https://proceedings.neurips.cc/paper_files/paper/2023/file/5f09bfe6730e9627a9f800d01a8ad5cd-Paper-Datasets_and_Benchmarks.pdf
OpenReview: https://openreview.net/forum?id=qWsQi9DGJb
Project: https://project.inria.fr/stressid/
Baseline code: https://github.com/robustml-eurecom/stressID

### Participants
- **65 participants**: 18 women, 47 men, ages 21–55 (29 ± 7)
- Mostly STEM students and workers (32% MSc/interns, 20% PhD, 48% other professions)
- All required English proficiency; experiment conducted in English
- Instructed: no smoking, caffeine, or exercise 3 h before
- **3 subjects (option B) withheld video release** → only 62 subjects have video

### Sensors
| Signal | Device / placement | Rate |
|---|---|---|
| ECG | BioSignalsPlux, 3 Ag/AgCl on ribs, non-dominant side | **500 Hz** |
| EDA | 2 Ag/AgCl on **palm of non-dominant hand** | 500 Hz |
| Respiration | Chest belt, piezoelectric | 500 Hz |
| Video | Logitech QuickCam Pro 9000, 720p | **15 fps** |
| Audio | Integrated mic | 32 kHz, 16-bit |

Note: EDA is measured on the **palm** here vs. the **abdomen** in WESAD. These are not interchangeable measurement sites — palmar EDA is far more reactive.

### Protocol (~35 min, 11 tasks in 4 blocks)
| Block | Task | Duration | Purpose |
|---|---|---|---|
| 1 | **Breathing** | 3 min | Guided breathing video — designated **neutral baseline** |
| 1 | Counting forward | 1 min | — |
| 2 | Video1 | 2 min | *There's Something About Mary* — low arousal, positive valence |
| 2 | Video2 | 3 min | *Indiana Jones and the Last Crusade* — high arousal, negative valence |
| 3 | Counting1 | 1 min | Count down from 100 by 3 |
| 3 | Counting2 | 1 min | Count down from 1011 by 7 |
| 3 | Stroop | 1 min | Stroop colour-word test |
| 3 | **Speaking** | 1 min | Strengths/weaknesses, job-interview framing (social evaluative) |
| 3 | Math | 1 min | 20 problems in 1 minute |
| 3 | Reading | 1 min | Read a text, then explain it (TSST variant) |
| 3 | Counting3 | 1 min | Count down from 1152 by 3 *while* repeating a hand movement |
| 4 | **Relax** | 2.5 min | Relaxing video |

Block 3 tasks are all strictly 1 minute, in a deliberately unexpected order. After block 3, subjects name the task they found most stressful.

### Label provenance — **self-report, per task**
After **every task**, the subject rates four things on 0–10 scales: perceived **stress**, perceived **relaxation**, **valence** (SAM), **arousal** (SAM).

- **Binary label:** stress rating ≥ 5 → stressed (1); < 5 → not stressed (0).
- **3-class label:** relaxed = valence > 5 AND arousal < 5 AND relax > 5; stressed = arousal > 5 AND valence < 5 AND stress > 5; neutral otherwise.

### Composition after task-splitting
| Modality | Annotated recordings | Hours |
|---|---|---|
| Physiological | 711 | ~19 h |
| Video | 587 | ~15 h |
| Audio | **385** | ~6 h |
| **Total** | | **>39 h** |

**Audio exists only for the 7 talking tasks** — breathing, the video clips, and relax carry no meaningful speech. 8 video/audio recordings were lost to camera malfunction.

Tasks with **all three** modalities: **370** — and these are exclusively talking tasks, which are **70% labelled stress**.

### Published benchmark — MY CORRECTNESS TARGETS
⚠ **Read this carefully: the StressID baselines are NOT subject-independent.** The authors used 10 random 80/20 splits over *tasks*, so the same subject appears in both train and test. Their numbers are therefore optimistic relative to LOSO.

Binary (2-class), all available tasks per modality:

| Baseline | F1 | Accuracy |
|---|---|---|
| Physio handcrafted + RF | 0.73 ± 0.02 | 0.72 ± 0.03 |
| Physio handcrafted + SVM | 0.71 ± 0.02 | 0.71 ± 0.02 |
| Facial AUs + kNN | 0.70 ± 0.04 | 0.69 ± 0.04 |
| Audio handcrafted + kNN | 0.67 ± 0.06 | 0.60 ± 0.05 |
| Audio Wav2Vec 2.0 | 0.70 ± 0.02 | 0.66 ± 0.03 |

Multimodal, on the 370 all-modality tasks:

| Fusion | F1 | Accuracy |
|---|---|---|
| Feature-level + SVM | 0.64 ± 0.09 | 0.56 ± 0.05 |
| Feature-level + MLP | 0.66 ± 0.04 | 0.61 ± 0.03 |
| Feature-level + DBN | 0.58 ± 0.06 | 0.52 ± 0.05 |
| **Decision-level, average rule** | **0.72 ± 0.05** | **0.65 ± 0.05** |
| Decision-level, sum rule | 0.72 ± 0.05 | 0.64 ± 0.05 |

> **Two conclusions I should not forget:**
> 1. **Decision-level fusion beat feature-level fusion on this dataset**, in the dataset authors' own experiments. Feature concatenation was the *worst* family.
> 2. Since these are not LOSO, my subject-independent numbers should land **at or below** ~0.72. A LOSO result of ~0.65–0.70 on physio is consistent with the literature, not a failure.

### ⚠ Confounds I must control for
1. **The audio subset is the stress subset.** Audio only exists for talking tasks, which are 70% stress. An audio model may be learning "is this a talking task," not "is this person stressed."
2. **Non-native English speakers.** The authors note that speaking English can itself be stress-inducing. Speaking-task labels are confounded with language anxiety.
3. **Electrode attachment may itself be stressful.** The authors flag this — the "baseline" may not be a true resting state.
4. **The breathing baseline is contaminated for respiration features.** Block 1 is a *guided breathing exercise*, which deliberately alters respiration rate and depth. Using it as the calm baseline for RRV features anchors respiration against a manipulated state. **For respiration, use `Relax` as the baseline instead of `Breathing`.** Note this as an open decision.
5. **Task duration is only 1 minute** for the 7 interactive stressors. HRV indices (SDNN especially) are marginal at 60 s. RMSSD is usable; SDNN and frequency-domain HRV are not reliable at this length.
6. **Gender imbalance** (18 F / 47 M) and a narrow STEM population.

---

## A.3 The label incompatibility — the most important thing in this file

| | WESAD | StressID |
|---|---|---|
| What "stress = 1" means | The subject **was exposed to the TSST** | The subject **rated their own stress ≥ 5/10** |
| Annotation source | Experimenter / protocol | Participant self-assessment |
| Annotation granularity | Condition block (~10–20 min) | Task (1–3 min) |
| Validated by | Questionnaires, post hoc | — (the rating *is* the label) |

These are **two different constructs wearing the same label name.**

A WESAD "stress" window can come from a subject who reported feeling fine. A StressID "not-stress" window can come from a subject whose heart rate was through the roof but who rated themselves 4/10.

This is almost certainly a large part of why my pooled `combined` dataset performs badly and why the domain classifier reached 0.9998 accuracy. **I should stop pooling them into one training set until I can justify it.** Report per dataset. If I want a combined claim, it has to be cross-dataset *transfer*, which is a different and more honest experiment.

---

## A.4 Is there a "true stress" ground truth? — **No. Answer written out so I stop wondering.**

There are three tiers of stress annotation used in the field:

| Tier | Method | Objectivity | In my datasets? |
|---|---|---|---|
| 1 | **Endocrine** — salivary cortisol, α-amylase | Highest | **Neither dataset collected it** |
| 2 | **Protocol/stressor-based** — "this block was the TSST" | Medium | WESAD |
| 3 | **Self-report** — PSS, STAI, SAM, 0–10 rating | Subjective | StressID (and WESAD's unused questionnaires) |

Even tier 1 would not solve my problem. Cortisol peaks roughly 15–25 minutes *after* stressor onset and is sampled a handful of times per session. It cannot label a 10-second window under any circumstances.

**Therefore: there is no window-level ground truth in either dataset, and there cannot be.** Every window-level label in my project is *inherited* from a block- or task-level annotation. My intuition was right — a 3-minute "stress" recording does not mean the person was stressed for all 180 seconds. It means something happened in that window of time that the protocol or the participant called stressful.

### What to do about it — four options, ranked

1. **Accept coarse labels, report honestly.** What everyone does. Fine, but say it in the limitations.
2. **Trim transitions.** Discard the first 30–60 s of each condition (onset lag) and any window overlapping a task boundary. Cheap, defensible, usually helps.
3. **Use self-report intensity as a soft/ordinal target** instead of a hard binary. StressID gives a 0–10 scale; binarising at 5 throws away most of the information. Regression or ordinal loss on the raw rating is more faithful to what was actually measured.
4. **Multiple-Instance Learning (MIL).** ← *This is the framing that matches my intuition exactly.*
   - A **bag** = one task/condition recording, which carries the label.
   - **Instances** = the windows inside it, which are unlabelled.
   - Standard MIL assumption: a bag is positive if **at least one** instance is positive.
   - An attention-based MIL pooling layer learns *which windows in the recording actually carried the stress*, and you can visualise those attention weights against the raw signal.
   - This is rare in stress detection and is a legitimate contribution. It also produces exactly the artefact a clinician would want: "here is the moment in the session where this person spiked."

**Decision (2026-07-24):** build the standard binary pipeline first as the reproducible baseline, then MIL as the research contribution. Do not start with MIL.

---

# PART B — Reading List

Tick when read and note the one thing I took from it.

### Core dataset papers
- [ ] Schmidt et al. 2018 — WESAD. https://doi.org/10.1145/3242969.3242985
- [ ] Chaptoukaev et al. 2023 — StressID. https://openreview.net/forum?id=qWsQi9DGJb

### Stress induction and measurement
- [ ] Kirschbaum, Pirke & Hellhammer 1993 — *The Trier Social Stress Test.* Neuropsychobiology 28(1-2):76–81. **The protocol behind the "stress" label in both datasets.**
- [ ] Allen et al. 2017 — *The Trier Social Stress Test: principles and practice.* Neurobiology of Stress 6:113–126.
- [ ] Bali & Jaggi 2015 — *Clinical experimental stress studies: methods and assessment.* Reviews in the Neurosciences 26(5):555–579.

### Physiological signal grounding — read before touching features
- [ ] Task Force of the ESC/NASPE 1996 — *Heart rate variability: standards of measurement, physiological interpretation, and clinical use.* Eur Heart J 17:354–381. **Defines every HRV feature I use and the minimum recording lengths.**
- [ ] Shaffer & Ginsberg 2017 — *An overview of heart rate variability metrics and norms.* Frontiers in Public Health. **Short and readable; start here, then the Task Force doc.**
- [ ] Kreibig 2010 — *Autonomic nervous system activity in emotion: a review.* Biological Psychology 84(3):394–421. **This is where WESAD's 60-second window choice comes from.**
- [ ] Kim et al. 2018 — *Stress and heart rate variability: a meta-analysis.* Psychiatry Investigation 15(3):235.
- [ ] Boucsein — *Electrodermal Activity* (2nd ed.) — reference for SCL/SCR decomposition and measurement sites.

### Method / framing
- [ ] Ilse, Tomczak & Welling 2018 — *Attention-based Deep Multiple Instance Learning.* ICML. https://arxiv.org/abs/1802.04712
- [ ] Vovk, Gammerman & Shafer — *Algorithmic Learning in a Random World* (conformal prediction), or Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*, https://arxiv.org/abs/2107.07511
- [ ] Gjoreski et al. 2016 — *Continuous stress detection using a wrist device.* UbiComp. (WESAD compares against this.)
- [ ] Hovsepian et al. 2015 — *cStress: towards a gold standard for continuous stress assessment.* UbiComp.

### Tooling
- [ ] NeuroKit2 documentation — the ECG/EDA/RSP processing I depend on. Read what `nk.ecg_process` and `nk.eda_phasic` actually do before trusting them.

---

# PART C — Dashboard Specification

The dashboard is not decoration. Its job is to answer **"is this label true for this subject?"** Build these six views, in this order.

### V1 — Protocol timeline
Per subject: a horizontal time axis with each condition/task as a coloured block, labels overlaid. WESAD and StressID side by side so the structural difference is visible at a glance.

### V2 — Raw signal viewer
ECG / EDA / RESP for one subject, full session, with condition blocks shaded behind the trace. Zoomable.
**What I'm looking for:** does EDA visibly rise during TSST? Does HR? If I can't see it here, no model will find it.

### V3 — Per-subject effect size ⭐ *the key view*
For each subject, compute Cohen's *d* between calm-condition and stress-condition windows, per feature (mean HR, RMSSD, SCR count, mean SCL, respiration rate).

Render as a heatmap: subjects on one axis, features on the other, *d* as colour.

**This directly measures label validity per subject.** A subject with *d* ≈ 0 across every feature either (a) didn't respond to the stressor, or (b) has bad data. Either way their labels are unreliable, and I will know their subject ID before I train anything. This is the principled version of the ad-hoc "diagnostics" I ran before.

### V4 — Within-task time course ⭐ *answers my "peak vs. sustained" question*
Align every stress recording to its onset (t=0) and plot the mean and IQR of EDA / HR across subjects over the task duration.

**What I'm looking for:** does the response ramp and sustain, or spike and decay? Where in the recording does it peak? The answer tells me whether transition-trimming is worth it and whether MIL is justified — and it's a figure worth putting in the paper either way.

### V5 — Label distribution
- WESAD: duration per condition per subject (shows the 53/30/17 imbalance concretely)
- StressID: histogram of the 0–10 stress ratings, per task and overall; where subjects sit relative to the ≥5 cutoff; how many subjects are near-always-stressed or near-never-stressed

**What I'm looking for:** how much information the binarisation at 5 destroys, and which subjects are outliers (e.g. `71i5` with a 74% stress ratio).

### V6 — Signal quality
Per subject per task: ECG R-peak plausibility, % EDA samples in physiological range, missing/flat segments, video frames with a detected face, VAD ratio for audio.

**What I'm looking for:** which recordings to exclude and why — decided from data, before modelling, and written down.

**Stack:** plain Python. Streamlit or Panel, matplotlib, neurokit2, pandas. Do **not** build a React app for this. It is a research instrument for one user, and it should be boring.

---

# PART D — Running Log

Append a dated entry every working session. Template:

```
## YYYY-MM-DD — <one-line title>

**Question:** what am I trying to find out?
**Predicted:** what I expect to see, WRITTEN BEFORE RUNNING. (Non-negotiable.)
**Ran:** command / script path / commit hash
**Result:** what actually happened, with numbers
**Surprised?** yes / no — and if yes, why
**Concluded:** one or two sentences
**Next:** the single next question
```

---

## 2026-07-24 — Dataset dossier compiled; project restarted

**Question:** What do WESAD and StressID actually contain, and where do their labels come from?

**Predicted:** I assumed both datasets had comparable "stress" labels and that some form of objective ground truth existed.

**Ran:** Read both source papers end to end.

**Result:**
- WESAD labels are the **experimental protocol**; the questionnaires were collected but not used as labels in the benchmark.
- StressID labels are **per-task self-report**, 0–10, binarised at 5.
- **Neither dataset collected cortisol or any endocrine measure.** There is no objective ground truth in either.
- No window-level labels exist in either dataset. All window labels are inherited from block/task annotations.
- Published LOSO binary target for WESAD chest physio: **93.12%** (LDA). My current pipeline gets 76.94%.
- StressID published baselines (0.72 physio+RF) are from **random task splits, not LOSO**, so they are optimistic — my LOSO number should be lower and that is expected, not a failure.
- StressID's own experiments found **decision-level fusion beat feature-level fusion**.

**Surprised?** Yes, three times.
1. I assumed the two datasets' labels meant the same thing. They do not, and I have been pooling them.
2. I assumed my 71% was near a ceiling. WESAD's published LOSO ceiling is 93% and I am 16 points below it.
3. I assumed feature concatenation was the standard strong approach. The StressID authors found the opposite on their own data.

**Concluded:** The project's foundations were built on an unexamined assumption that the two datasets are commensurable. They are not. Restarting per-dataset, with published baselines as correctness tests.

**Next:** Build dashboard view V3 (per-subject effect size) on WESAD. Before running it, predict how many of the 15 subjects will show Cohen's *d* > 0.8 on mean HR between baseline and TSST.
