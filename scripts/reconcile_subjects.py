"""
scripts/reconcile_subjects.py

Reconcile StressID subjects across:
1. Raw disk scan: Physiological, Videos, Audio folders and metadata CSVs
2. Official baseline feature CSVs: all_physiological_features.csv, video, audio
3. Official Technical-info_final.pdf documented manifest:
   - 65 physio
   - Physio-only: [hh2e, wfsl, 37ir, hvpa, ql3b, r0a3, uyrl, qx2o, dmbd] (9 subjects)
   - Physio+Audio only (no video): [u3v9, 7m3c] (2 subjects)
   - Physio+Video only (no audio): [f6q3] (1 subject)
   - Expected video = 65 - 9 - 2 = 54
   - Expected audio = 65 - 9 - 1 = 55
"""

import sys
from pathlib import Path
import pandas as pd
import yaml

RAW_ROOT = Path("Datasets/stressid")
BASELINE_FEAT_DIR = Path("scratch/stressID_baseline/Feature Extraction/Features")

def get_disk_subjects():
    physio_dir = RAW_ROOT / "Physiological"
    video_dir = RAW_ROOT / "Videos"
    audio_dir = RAW_ROOT / "Audio"

    physio_subs = sorted([d.name for d in physio_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    video_subs = sorted([d.name for d in video_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    audio_subs = sorted([d.name for d in audio_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    labels_df = pd.read_csv(RAW_ROOT / "labels.csv")
    label_subs = sorted(labels_df["subject/task"].apply(lambda x: x.split("_")[0]).unique().tolist())

    self_assess_df = pd.read_csv(RAW_ROOT / "self_assessments.csv", sep=";")
    self_assess_subs = sorted([c for c in self_assess_df.columns[1:] if not c.startswith("Unnamed")])

    return {
        "physio": set(physio_subs),
        "video": set(video_subs),
        "audio": set(audio_subs),
        "labels": set(label_subs),
        "self_assess": set(self_assess_subs),
    }

def get_baseline_subjects():
    baseline_physio_path = BASELINE_FEAT_DIR / "all_physiological_features.csv"
    baseline_video_path = BASELINE_FEAT_DIR / "video11tasks_aus_gaze_mean_std.csv"
    baseline_audio_path = BASELINE_FEAT_DIR / "HCfeatures.csv"

    results = {}
    if baseline_physio_path.exists():
        df = pd.read_csv(baseline_physio_path)
        col0 = df.columns[0]
        subs = set(df[col0].apply(lambda x: str(x).split("_")[0]).unique())
        results["physio"] = subs
        results["physio_rows"] = len(df)
        results["physio_cols"] = df.shape[1] - 1
    
    if baseline_video_path.exists():
        df = pd.read_csv(baseline_video_path)
        col0 = "id" if "id" in df.columns else df.columns[0]
        subs = set(df[col0].apply(lambda x: str(x).split("_")[0]).unique())
        results["video"] = subs
        results["video_rows"] = len(df)
        results["video_cols"] = df.shape[1] - 1

    if baseline_audio_path.exists():
        df = pd.read_csv(baseline_audio_path, header=None)
        subs = set(df[0].apply(lambda x: str(x).split("_")[0]).unique())
        results["audio"] = subs
        results["audio_rows"] = len(df)
        results["audio_cols"] = df.shape[1] - 1

    return results

def main():
    disk = get_disk_subjects()
    baseline = get_baseline_subjects()

    # Documented manifest from Technical information.pdf
    doc_physio_only = {"hh2e", "wfsl", "37ir", "hvpa", "ql3b", "r0a3", "uyrl", "qx2o", "dmbd"}
    doc_physio_audio_only = {"u3v9", "7m3c"}
    doc_physio_video_only = {"f6q3"}

    print("=" * 70)
    print("STRESSID SUBJECT RECONCILIATION")
    print("=" * 70)

    print("\n1. DISK RAW COUNTS:")
    print(f"   Physiological folder : {len(disk['physio'])} subjects")
    print(f"   Videos folder        : {len(disk['video'])} subjects")
    print(f"   Audio folder         : {len(disk['audio'])} subjects")
    print(f"   labels.csv           : {len(disk['labels'])} subjects (hh2e missing labels)")
    print(f"   self_assessments.csv : {len(disk['self_assess'])} subjects")

    print("\n2. BASELINE CSV COUNTS:")
    print(f"   all_physiological_features.csv: {len(baseline.get('physio', []))} subjects, {baseline.get('physio_rows')} rows, {baseline.get('physio_cols')} features")
    print(f"   video11tasks_aus_gaze...csv   : {len(baseline.get('video', []))} subjects, {baseline.get('video_rows')} rows, {baseline.get('video_cols')} features")
    print(f"   HCfeatures.csv (audio)        : {len(baseline.get('audio', []))} subjects, {baseline.get('audio_rows')} rows, {baseline.get('audio_cols')} features")

    print("\n3. CROSS-CHECK DISK vs BASELINE CSVs:")
    diff_physio_disk_base = disk["physio"] - baseline.get("physio", set())
    diff_physio_base_disk = baseline.get("physio", set()) - disk["physio"]
    print(f"   Physio Disk - Baseline : {sorted(diff_physio_disk_base) if diff_physio_disk_base else 'None (Identical)'}")
    print(f"   Physio Baseline - Disk : {sorted(diff_physio_base_disk) if diff_physio_base_disk else 'None (Identical)'}")

    diff_video_disk_base = disk["video"] - baseline.get("video", set())
    diff_video_base_disk = baseline.get("video", set()) - disk["video"]
    print(f"   Video Disk - Baseline  : {sorted(diff_video_disk_base) if diff_video_disk_base else 'None (Identical)'}")
    print(f"   Video Baseline - Disk  : {sorted(diff_video_base_disk) if diff_video_base_disk else 'None (Identical)'}")

    diff_audio_disk_base = disk["audio"] - baseline.get("audio", set())
    diff_audio_base_disk = baseline.get("audio", set()) - disk["audio"]
    print(f"   Audio Disk - Baseline  : {sorted(diff_audio_disk_base) if diff_audio_disk_base else 'None (Identical)'}")
    print(f"   Audio Baseline - Disk  : {sorted(diff_audio_base_disk) if diff_audio_base_disk else 'None (Identical)'}")

    print("\n4. MODALITY INTERSECTIONS ON DISK:")
    all_disk_subs = disk["physio"]
    observed_physio_only = all_disk_subs - disk["video"] - disk["audio"]
    observed_physio_audio = (disk["physio"] & disk["audio"]) - disk["video"]
    observed_physio_video = (disk["physio"] & disk["video"]) - disk["audio"]
    observed_tri_modal = disk["physio"] & disk["video"] & disk["audio"]

    print(f"   Tri-modal (all 3)      : {len(observed_tri_modal)} subjects")
    print(f"   Physio-only            : {len(observed_physio_only)} subjects -> {sorted(observed_physio_only)}")
    print(f"   Physio + Audio only    : {len(observed_physio_audio)} subjects -> {sorted(observed_physio_audio)}")
    print(f"   Physio + Video only    : {len(observed_physio_video)} subjects -> {sorted(observed_physio_video)}")

    print("\n5. COMPARISON WITH TECHNICAL-INFO.PDF MANIFEST:")
    print(f"   Doc Physio-only ({len(doc_physio_only)}): {sorted(doc_physio_only)}")
    print(f"   Diff Physio-only (Disk - Doc): {sorted(observed_physio_only - doc_physio_only)}")
    print(f"   Diff Physio-only (Doc - Disk): {sorted(doc_physio_only - observed_physio_only)}")

    print(f"   Doc Physio+Audio ({len(doc_physio_audio_only)}): {sorted(doc_physio_audio_only)}")
    print(f"   Diff Physio+Audio (Disk - Doc): {sorted(observed_physio_audio - doc_physio_audio_only)}")
    print(f"   Diff Physio+Audio (Doc - Disk): {sorted(doc_physio_audio_only - observed_physio_audio)}")

    print(f"   Doc Physio+Video ({len(doc_physio_video_only)}): {sorted(doc_physio_video_only)}")
    print(f"   Diff Physio+Video (Disk - Doc): {sorted(observed_physio_video - doc_physio_video_only)}")
    print(f"   Diff Physio+Video (Doc - Disk): {sorted(doc_physio_video_only - observed_physio_video)}")

    expected_video_count = len(all_disk_subs) - len(doc_physio_only) - len(doc_physio_audio_only)
    expected_audio_count = len(all_disk_subs) - len(doc_physio_only) - len(doc_physio_video_only)
    print(f"\n   Expected Video count: {expected_video_count} (Doc 65 - 9 - 2) vs Observed Disk: {len(disk['video'])}")
    print(f"   Expected Audio count: {expected_audio_count} (Doc 65 - 9 - 1) vs Observed Disk: {len(disk['audio'])}")

    # Check which specific subject accounts for any off-by-one
    if len(disk["video"]) != expected_video_count:
        expected_video_subs = all_disk_subs - doc_physio_only - doc_physio_audio_only
        print(f"   Video discrepancies: Expected - Disk = {expected_video_subs - disk['video']}, Disk - Expected = {disk['video'] - expected_video_subs}")

    if len(disk["audio"]) != expected_audio_count:
        expected_audio_subs = all_disk_subs - doc_physio_only - doc_physio_video_only
        print(f"   Audio discrepancies: Expected - Disk = {expected_audio_subs - disk['audio']}, Disk - Expected = {disk['audio'] - expected_audio_subs}")

    # Build and write config/subjects.yaml
    config_data = {
        "stressid": {
            "all_subjects": sorted(list(all_disk_subs)),
            "counts": {
                "physio": len(disk["physio"]),
                "video": len(disk["video"]),
                "audio": len(disk["audio"]),
                "tri_modal": len(observed_tri_modal),
            },
            "physio_only": sorted(list(observed_physio_only)),
            "physio_audio_only": sorted(list(observed_physio_audio)),
            "physio_video_only": sorted(list(observed_physio_video)),
            "tri_modal_subjects": sorted(list(observed_tri_modal)),
            "missing_tasks": {
                "wfsl": ["Relax"],
                "h7j3": ["Relax"],
                "7h5u": ["Video2"],
                "j1u8": ["Video1"],
                "k67g": ["Breathing", "Video1"],
                "h8r2": ["Breathing"],
            },
            "missing_labels": {
                "hh2e": ["arousal", "valence", "binary-stress", "affect3-class"]
            },
            "no_image_publication": [
                "dmbd", "45lx", "cxj0", "5f7t", "y9z6", "u3v9", "7m3c", "x1q3", "9j3o", "c3m7", "a1k9", "h8s1"
            ],
            "baseline_task": {
                "ecg": "Breathing",
                "eda": "Breathing",
                "resp": "Relax",
                "face": "Breathing",
                "voice": None
            },
            "baseline_task_overrides": {
                "k67g": {"face": "Relax"},
                "h8r2": {"face": "Relax"}
            }
        }
    }

    out_yaml = Path("config/subjects.yaml")
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.dump(config_data, f, sort_keys=False)
    print(f"\nWritten reconciled manifest to {out_yaml}")

if __name__ == "__main__":
    main()
