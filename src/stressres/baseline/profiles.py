"""
src/stressres/baseline/profiles.py

Calm-baseline profiling and robust anchoring:
z = (x - median_calm) / (1.4826 * MAD_calm + 1e-8)
- ECG, EDA, Face: anchored against 'Breathing' task (with subject overrides)
- Respiration: anchored against 'Relax' task (Breathing is guided exercise confound)
- Voice: unanchored (no calm speech exists; voice.anchored is uncalibrated)
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def compute_baseline_stats(df_baseline_windows: pd.DataFrame, feat_cols: list) -> Dict[str, Dict[str, float]]:
    """
    Compute robust median and MAD across baseline windows for a subject.
    MAD = median(|x - median(x)|)
    """
    stats_dict = {}
    for col in feat_cols:
        vals = df_baseline_windows[col].dropna().values.astype(np.float64)
        if len(vals) == 0:
            stats_dict[col] = {"median": np.nan, "mad": np.nan}
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        stats_dict[col] = {"median": med, "mad": mad}
    return stats_dict


def anchor_feature_vector(
    x_val: float,
    med: float,
    mad: float,
    eps: float = 1e-8
) -> float:
    """
    Robust baseline anchoring formula:
    z = (x - median) / (1.4826 * mad + eps)
    """
    if pd.isna(x_val) or pd.isna(med) or pd.isna(mad):
        return np.nan
    scale = 1.4826 * mad
    if scale < eps:
        scale = 1.0  # Avoid exploding if zero variance
    return float((x_val - med) / (scale + eps))
