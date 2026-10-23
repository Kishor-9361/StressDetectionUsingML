"""
src/stressres/clean/resp.py

Respiration signal cleaning using NeuroKit2 with biosppy method.
"""

import numpy as np
import neurokit2 as nk


def clean_resp(raw_resp: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """
    Clean Respiration signal matching StressID baseline protocol.
    Standardizes signal and applies Butterworth bandpass filtering via NeuroKit2 biosppy method.
    """
    raw_resp = np.asarray(raw_resp, dtype=np.float64)
    if np.isnan(raw_resp).any():
        raw_resp = np.nan_to_num(raw_resp, nan=np.nanmedian(raw_resp))

    std_val = raw_resp.std()
    if std_val < 1e-8:
        return np.zeros_like(raw_resp)

    norm_resp = (raw_resp - raw_resp.mean()) / std_val
    cleaned = nk.rsp_clean(norm_resp, sampling_rate=int(fs), method="biosppy")
    return cleaned
