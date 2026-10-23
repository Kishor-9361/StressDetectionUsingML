"""
src/stressres/features/eda.py

EDA feature extraction matching the StressID baseline protocol.
Extracts 24 features (tonic SCL, phasic SCR, statistical and peak dynamics).
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import neurokit2 as nk


def extract_eda_window_features(clean_eda: np.ndarray, fs: float = 500.0) -> dict:
    x = np.asarray(clean_eda, dtype=np.float64)
    cols = ['max_eda', 'min_eda', 'mean_eda', 'sd_eda', 'ku_eda', 'sk_eda', 'median_eda',
            'dynrange', 'scl_slope', 'max_scr', 'min_scr', 'mean_scr', 'sd_scr', 'max_scl',
            'min_scl', 'mean_scl', 'sd_scl', 'nSCR', 'aucSCR', 'meanAmpSCR', 'maxAmpSCR',
            'meanRespSCR', 'sumAmpSCR', 'sumRespSCR']

    if len(x) < 5 or np.all(x == 0.0) or np.isnan(x).all():
        return {c: np.nan for c in cols}

    try:
        # nk.eda_phasic decomposes signal into tonic and phasic
        eda_decomp = nk.eda_phasic(x, sampling_rate=int(fs))
        scr = np.asarray(eda_decomp['EDA_Phasic'])
        scl = np.asarray(eda_decomp['EDA_Tonic'])

        x_axis = np.linspace(0, len(scl) / fs, len(scl))
        slope = float(np.polyfit(x_axis, scl, 1)[0]) if len(scl) > 1 else np.nan

        min_val = float(np.nanmin(x))
        max_val = float(np.nanmax(x))
        dynrange = max_val / min_val if abs(min_val) > 1e-12 else np.nan

        stat_dict = {
            'max_eda': max_val,
            'min_eda': min_val,
            'mean_eda': float(np.nanmean(x)),
            'sd_eda': float(np.nanstd(x)),
            'ku_eda': float(stats.kurtosis(x, nan_policy="omit")),
            'sk_eda': float(stats.skew(x, nan_policy="omit")),
            'median_eda': float(np.nanquantile(x, 0.5)),
            'dynrange': dynrange,
            'scl_slope': slope,
            'max_scr': float(np.nanmax(scr)),
            'min_scr': float(np.nanmin(scr)),
            'mean_scr': float(np.nanmean(scr)),
            'sd_scr': float(np.nanstd(scr)),
            'max_scl': float(np.nanmax(scl)),
            'min_scl': float(np.nanmin(scl)),
            'mean_scl': float(np.nanmean(scl)),
            'sd_scl': float(np.nanstd(scl)),
        }

        # Peak dynamics on SCR
        try:
            _, info = nk.eda_peaks(scr, sampling_rate=int(fs))
            peaks = info.get('SCR_Peaks', [])
            amplitude = info.get('SCR_Amplitude', [])
            recovery = info.get('SCR_RecoveryTime', [])

            dur_min = (len(x) / fs) / 60.0
            n_scr = len(peaks) / dur_min if dur_min > 0 else np.nan
            auc_scr = float(np.trapz(scr))
            mean_amp = float(np.nanmean(amplitude)) if len(amplitude) > 0 else np.nan
            max_amp = float(np.nanmax(amplitude)) if len(amplitude) > 0 else np.nan
            mean_resp = float(np.nanmean(recovery)) if len(recovery) > 0 else np.nan
            sum_amp = float(np.nansum(amplitude) / dur_min) if dur_min > 0 and len(amplitude) > 0 else np.nan
            sum_resp = float(np.nansum(recovery) / dur_min) if dur_min > 0 and len(recovery) > 0 else np.nan
        except Exception:
            n_scr = auc_scr = mean_amp = max_amp = mean_resp = sum_amp = sum_resp = np.nan

        time_dict = {
            'nSCR': n_scr,
            'aucSCR': auc_scr,
            'meanAmpSCR': mean_amp,
            'maxAmpSCR': max_amp,
            'meanRespSCR': mean_resp,
            'sumAmpSCR': sum_amp,
            'sumRespSCR': sum_resp,
        }

        stat_dict.update(time_dict)
        return stat_dict
    except Exception:
        return {c: np.nan for c in cols}
