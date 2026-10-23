"""
src/stressres/clean/eda.py

EDA signal cleaning using NeuroKit2 with biosppy method.
"""

import numpy as np
import neurokit2 as nk


def clean_eda(raw_eda: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """
    Clean EDA signal matching StressID baseline protocol.
    Standardizes signal and applies Butterworth lowpass filtering via NeuroKit2 biosppy method.
    """
    raw_eda = np.asarray(raw_eda, dtype=np.float64)
    if np.isnan(raw_eda).any():
        raw_eda = np.nan_to_num(raw_eda, nan=np.nanmedian(raw_eda))

    std_val = raw_eda.std()
    if std_val < 1e-8:
        # Flatline/saturated signal (e.g. r5s8_Counting3 railed at 65535)
        return np.zeros_like(raw_eda)

    norm_eda = (raw_eda - raw_eda.mean()) / std_val
    cleaned = nk.eda_clean(norm_eda, sampling_rate=int(fs), method="biosppy")
    return cleaned
