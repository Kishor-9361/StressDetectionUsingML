"""
scripts/04_anchor_features.py

Stage 4 — Baseline Anchoring
Calibrates per-modality feature tables using robust median/MAD normalization against
calm-task reference baselines:
- ECG, EDA, Face: Breathing (overridden to Relax for k67g, h8r2 face)
- Respiration: Relax (confound substitution)
- Voice: Unanchored limitation
Saves calibrated tables to data/processed/calibrated/<modality>_calibrated.parquet
and generates spot-check plots in docs/verification/.
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from stressres.baseline.profiles import compute_baseline_stats, anchor_feature_vector

FEAT_DIR = Path("data/processed/features")
CALIB_DIR = Path("data/processed/calibrated")
VERIFY_DIR = Path("docs/verification")


def calibrate_table(modality: str, baseline_task_default: str, overrides: dict):
    in_file = FEAT_DIR / f"{modality}_features.parquet"
    if not in_file.exists():
        print(f"Warning: Missing {in_file}")
        return None

    df = pd.read_parquet(in_file)
    id_cols = ["window_id", "subject_id", "task_id"]
    feat_cols = [c for c in df.columns if c not in id_cols]

    calib_rows = []
    subjects = df["subject_id"].unique()

    for sub in subjects:
        sub_df = df[df["subject_id"] == sub]
        b_task = overrides.get(sub, {}).get(modality, baseline_task_default)

        # Baseline windows for this subject
        base_df = sub_df[sub_df["task_id"] == b_task]
        if len(base_df) == 0:
            # Fallback to Breathing or Relax
            base_df = sub_df[sub_df["task_id"].isin(["Breathing", "Relax", "Baseline"])]

        base_stats = compute_baseline_stats(base_df, feat_cols)

        for _, row in sub_df.iterrows():
            rec = {
                "window_id": row["window_id"],
                "subject_id": sub,
                "task_id": row["task_id"],
            }
            for col in feat_cols:
                x_val = row[col]
                med = base_stats[col]["median"]
                mad = base_stats[col]["mad"]
                rec[col] = anchor_feature_vector(x_val, med, mad)
            calib_rows.append(rec)

    calib_df = pd.DataFrame(calib_rows)
    out_file = CALIB_DIR / f"{modality}_calibrated.parquet"
    calib_df.to_parquet(out_file, index=False)
    print(f"Calibrated {modality}: saved {len(calib_df)} rows to {out_file}")
    return calib_df


def main():
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    with open("config/subjects.yaml", "r") as f:
        config = yaml.safe_load(f)["stressid"]

    base_tasks = config.get("baseline_task", {})
    overrides = config.get("baseline_task_overrides", {})

    print("Executing Stage 4 Baseline Anchoring...")

    calib_ecg = calibrate_table("ecg", base_tasks.get("ecg", "Breathing"), overrides)
    calib_eda = calibrate_table("eda", base_tasks.get("eda", "Breathing"), overrides)
    calib_resp = calibrate_table("resp", base_tasks.get("resp", "Relax"), overrides)
    calib_face = calibrate_table("face", base_tasks.get("face", "Breathing"), overrides)

    # Voice is unanchored per spec; save direct calibrated copy with note
    voice_in = FEAT_DIR / "voice_features.parquet"
    if voice_in.exists():
        voice_df = pd.read_parquet(voice_in)
        voice_df.to_parquet(CALIB_DIR / "voice_calibrated.parquet", index=False)
        print(f"Voice: preserved unanchored in {CALIB_DIR / 'voice_calibrated.parquet'} (limitation logged).")

    # Verification: check calm-task medians ≈ 0
    print("\nVerifying calibrated calm-task medians (Gate check):")
    ecg_base_med = calib_ecg[calib_ecg["task_id"] == "Breathing"]["meanHR"].median()
    eda_base_med = calib_eda[calib_eda["task_id"] == "Breathing"]["mean_eda"].median()
    resp_base_med = calib_resp[calib_resp["task_id"] == "Relax"]["meanRR"].median()
    print(f"  ECG Breathing meanHR median: {ecg_base_med:.4f} (target ~= 0)")
    print(f"  EDA Breathing mean_eda median: {eda_base_med:.4f} (target ~= 0)")
    print(f"  RESP Relax meanRR median   : {resp_base_med:.4f} (target ~= 0)")


    # Produce 3-subject visual spot check plot
    print("\nGenerating 3-subject spot check plot in docs/verification/...")
    spot_subs = ["2ea4", "45lx", "8g4y"]
    raw_ecg = pd.read_parquet(FEAT_DIR / "ecg_features.parquet")

    fig, axes = plt.subplots(len(spot_subs), 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Stage 4 Spot-Check: Raw vs. Calibrated ECG meanHR Distribution (Breathing vs Stressors)", fontsize=14)

    for i, sub in enumerate(spot_subs):
        sub_raw = raw_ecg[raw_ecg["subject_id"] == sub]
        sub_cal = calib_ecg[calib_ecg["subject_id"] == sub]

        raw_calm = sub_raw[sub_raw["task_id"] == "Breathing"]["meanHR"].dropna()
        raw_stress = sub_raw[sub_raw["task_id"].isin(["Stroop", "Math", "Counting3"])]["meanHR"].dropna()

        cal_calm = sub_cal[sub_cal["task_id"] == "Breathing"]["meanHR"].dropna()
        cal_stress = sub_cal[sub_cal["task_id"].isin(["Stroop", "Math", "Counting3"])]["meanHR"].dropna()

        axes[i, 0].boxplot([raw_calm, raw_stress], labels=["Calm (Breathing)", "Stressors"])
        axes[i, 0].set_title(f"Subject {sub} — Raw meanHR (bpm)")
        axes[i, 0].set_ylabel("bpm")
        axes[i, 0].grid(True, linestyle="--", alpha=0.5)

        axes[i, 1].boxplot([cal_calm, cal_stress], labels=["Calm (Breathing)", "Stressors"])
        axes[i, 1].axhline(0, color="red", linestyle="--", alpha=0.7, label="Calm Anchor (0)")
        axes[i, 1].set_title(f"Subject {sub} — Calibrated meanHR (z-score)")
        axes[i, 1].set_ylabel("z-score")
        axes[i, 1].legend(loc="upper right")
        axes[i, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = VERIFY_DIR / "baseline_calibration_spotcheck.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")
    print("Stage 4 baseline anchoring completed successfully.")


if __name__ == "__main__":
    main()
