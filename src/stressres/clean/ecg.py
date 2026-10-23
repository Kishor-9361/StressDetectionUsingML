"""
src/stressres/clean/ecg.py

ECG signal cleaning using NeuroKit2 with biosppy method.
"""

import numpy as np
import scipy.signal as signal
import neurokit2 as nk


def clean_ecg(raw_ecg: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """
    Clean ECG signal matching StressID baseline protocol.
    Standardizes signal and applies Butterworth filtering via NeuroKit2 biosppy method.
    """
    raw_ecg = np.asarray(raw_ecg, dtype=np.float64)
    # Handle NaNs if any
    if np.isnan(raw_ecg).any():
        raw_ecg = np.nan_to_num(raw_ecg, nan=np.nanmedian(raw_ecg))

    std_val = raw_ecg.std()
    if std_val < 1e-8:
        # Flatline/saturated signal
        return np.zeros_like(raw_ecg)

    norm_ecg = (raw_ecg - raw_ecg.mean()) / std_val
    cleaned = nk.ecg_clean(norm_ecg, sampling_rate=int(fs), method="biosppy")
    return cleaned


def detect_r_peaks(cleaned_ecg: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """
    Detect R-peaks matching baseline signal.find_peaks logic with 99th quantile height.
    """
    distance = int(fs / 3.0)
    q99 = np.quantile(cleaned_ecg, 0.99)
    height = q99 / 2.0
    peaks, _ = signal.find_peaks(cleaned_ecg, distance=distance, height=height)
    return peaks
