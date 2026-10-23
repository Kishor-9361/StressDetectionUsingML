"""
scripts/01_clean_signals.py

Stage 1 — Clean Signals
Walks all StressID physiological recordings, applies biosppy cleaning to ECG, EDA, and RESP,
saves per-task parquets under data/processed/clean/<subject>/, and produces 3-subject
visual spot-check plots of R-peak detection under docs/verification/.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


from stressres.clean.ecg import clean_ecg, detect_r_peaks
from stressres.clean.eda import clean_eda
from stressres.clean.resp import clean_resp

RAW_PHYSIO_DIR = Path("Datasets/stressid/Physiological")
OUT_CLEAN_DIR = Path("data/processed/clean")
VERIFY_DIR = Path("docs/verification")


def main():
    with open("config/subjects.yaml", "r") as f:
        config = yaml.safe_load(f)["stressid"]

    subjects = config["all_subjects"]
    OUT_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cleaning physiological signals for {len(subjects)} StressID subjects...")
    spotcheck_subjects = ["2ea4", "45lx", "8g4y"]
    spotcheck_data = {s: {} for s in spotcheck_subjects}

    total_files = 0
    saturated_eda_count = 0

    for sub in tqdm(subjects, desc="Subjects"):
        sub_dir = RAW_PHYSIO_DIR / sub
        if not sub_dir.exists():
            print(f"Warning: Missing dir {sub_dir}")
            continue

        out_sub_dir = OUT_CLEAN_DIR / sub
        out_sub_dir.mkdir(parents=True, exist_ok=True)

        txt_files = sorted([f for f in sub_dir.glob("*.txt") if not f.name.startswith(".")])

        for txt_file in txt_files:
            task = txt_file.stem.split("_")[-1]  # e.g. Baseline, Stroop, Math
            out_file = out_sub_dir / f"{task}_physio.parquet"

            if out_file.exists():
                total_files += 1
                if sub in spotcheck_subjects and task in ["Baseline", "Stroop", "Math"]:
                    if task not in spotcheck_data[sub]:
                        df_exist = pd.read_parquet(out_file)
                        clean_e = df_exist["clean_ecg"].values
                        raw_e = df_exist["raw_ecg"].values
                        peaks = detect_r_peaks(clean_e, fs=500.0)
                        spotcheck_data[sub][task] = {
                            "raw_ecg": raw_e,
                            "clean_ecg": clean_e,
                            "peaks": peaks,
                        }
                continue

            try:
                raw_df = pd.read_csv(txt_file)
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
                continue


            raw_ecg = raw_df["ECG"].values
            raw_eda = raw_df["EDA"].values
            raw_rsp = raw_df["RR"].values

            # Detect EDA saturation
            if raw_eda.std() < 1e-8:
                saturated_eda_count += 1

            clean_e = clean_ecg(raw_ecg, fs=500.0)
            clean_d = clean_eda(raw_eda, fs=500.0)
            clean_r = clean_resp(raw_rsp, fs=500.0)
            peaks = detect_r_peaks(clean_e, fs=500.0)

            # Store spot-check samples for 3 subjects
            if sub in spotcheck_subjects and task in ["Baseline", "Stroop", "Math"]:
                if task not in spotcheck_data[sub]:
                    spotcheck_data[sub][task] = {
                        "raw_ecg": raw_ecg,
                        "clean_ecg": clean_e,
                        "peaks": peaks,
                    }

            clean_df = pd.DataFrame({
                "raw_ecg": raw_ecg,
                "clean_ecg": clean_e,
                "raw_eda": raw_eda,
                "clean_eda": clean_d,
                "raw_resp": raw_rsp,
                "clean_resp": clean_r,
            })
            # Save parquet
            out_file = out_sub_dir / f"{task}_physio.parquet"
            clean_df.to_parquet(out_file, index=False)
            total_files += 1

    print(f"\nCompleted Stage 1 cleaning: {total_files} task parquets saved across {len(subjects)} subjects.")
    print(f"Railed/saturated EDA recordings noted: {saturated_eda_count}")

    # Produce 3-subject visual spot check plots
    print("Generating 3-subject R-peak visual spot checks in docs/verification/...")
    fs = 500
    sec = 10
    n_samples = fs * sec  # 10 seconds = 5000 samples

    for sub in spotcheck_subjects:
        tasks_avail = spotcheck_data[sub]
        baseline_task = "Baseline" if "Baseline" in tasks_avail else list(tasks_avail.keys())[0]
        stress_task = "Stroop" if "Stroop" in tasks_avail else ("Math" if "Math" in tasks_avail else list(tasks_avail.keys())[-1])

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharey=True)

        # Baseline window
        b_data = tasks_avail[baseline_task]
        b_clean = b_data["clean_ecg"][:n_samples]
        b_raw = b_data["raw_ecg"][:n_samples]
        # Standardize raw for visualization overlay
        b_raw_norm = (b_raw - b_raw.mean()) / (b_raw.std() + 1e-8)
        b_peaks = [p for p in b_data["peaks"] if p < n_samples]

        axes[0].plot(np.arange(len(b_raw_norm)) / fs, b_raw_norm, label="Raw ECG (Norm)", color="gray", alpha=0.6)
        axes[0].plot(np.arange(len(b_clean)) / fs, b_clean, label="Cleaned ECG (Biosppy)", color="#1f77b4", linewidth=1.2)
        axes[0].scatter(np.array(b_peaks) / fs, b_clean[b_peaks], color="red", s=40, zorder=5, label=f"R-peaks ({len(b_peaks)} detected)")
        axes[0].set_title(f"Subject {sub} — Calm Baseline ({baseline_task}) First 10s")
        axes[0].set_xlabel("Time (seconds)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Stress window
        s_data = tasks_avail[stress_task]
        s_clean = s_data["clean_ecg"][:n_samples]
        s_raw = s_data["raw_ecg"][:n_samples]
        s_raw_norm = (s_raw - s_raw.mean()) / (s_raw.std() + 1e-8)
        s_peaks = [p for p in s_data["peaks"] if p < n_samples]

        axes[1].plot(np.arange(len(s_raw_norm)) / fs, s_raw_norm, label="Raw ECG (Norm)", color="gray", alpha=0.6)
        axes[1].plot(np.arange(len(s_clean)) / fs, s_clean, label="Cleaned ECG (Biosppy)", color="#d62728", linewidth=1.2)
        axes[1].scatter(np.array(s_peaks) / fs, s_clean[s_peaks], color="black", s=40, zorder=5, label=f"R-peaks ({len(s_peaks)} detected)")
        axes[1].set_title(f"Subject {sub} — Stressor ({stress_task}) First 10s")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = VERIFY_DIR / f"rpeak_spotcheck_{sub}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved: {plot_path}")

    print("Stage 1 signal cleaning and visual verification completed successfully.")


if __name__ == "__main__":
    main()
