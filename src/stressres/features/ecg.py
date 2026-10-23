"""
src/stressres/features/ecg.py

ECG feature extraction matching the StressID baseline protocol.
Extracts 35 HRV features (time, freq, nonlinear) + 11 stat features = 46 features total.
Target 35 HRV features:
- Time (19): meanHR, minHR, maxHR, sdHR, modeHR, nNN, meanNN, SDSD, CVNN, SDNN, pNN50, pNN20, RMSSD, medianNN, q20NN, q80NN, minNN, maxNN, triHRV
- Freq (11): totalpower, LF, HF, ULF, VLF, VHF, LF/HF, rLF, rHF, peakLF, peakHF
- Nonlinear (5): SD1, SD2, SD1SD2, apEn, sampEn
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.integrate import trapz


def ecg_stat(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "max_ecg": float(np.nanmax(x)),
        "min_ecg": float(np.nanmin(x)),
        "mean_ecg": float(np.nanmean(x)),
        "sd_ecg": float(np.nanstd(x)),
        "ku_ecg": float(stats.kurtosis(x, nan_policy="omit")),
        "sk_ecg": float(stats.skew(x, nan_policy="omit")),
        "median_ecg": float(np.nanquantile(x, 0.5)),
        "q1_ecg": float(np.nanquantile(x, 0.25)),
        "q3_ecg": float(np.nanquantile(x, 0.75)),
        "q05_ecg": float(np.nanquantile(x, 0.05)),
        "q95_ecg": float(np.nanquantile(x, 0.95)),
    }


def ecg_time(r_peaks: np.ndarray, duration_s: float, sampling_rate: float = 500.0) -> dict:
    x = np.asarray(r_peaks, dtype=np.float64)
    if len(x) < 3:
        cols = ['meanHR', 'minHR', 'maxHR', 'sdHR', 'modeHR', 'nNN', 'meanNN', 'SDSD',
                'CVNN', 'SDNN', 'pNN50', 'pNN20', 'RMSSD', 'medianNN', 'q20NN', 'q80NN',
                'minNN', 'maxNN', 'triHRV']
        return {c: np.nan for c in cols}

    rri = np.diff(x)
    rri_ms = 1000.0 * rri / sampling_rate
    hr = 1000.0 * 60.0 / rri_ms

    diff_rri = np.diff(rri_ms)

    # HRV triangular index
    try:
        bin_width = 1000.0 / 128.0
        bins = np.arange(np.nanmin(rri_ms), np.nanmax(rri_ms) + bin_width, bin_width)
        hist, _ = np.histogram(rri_ms, bins=bins)
        tri_hrv = len(rri_ms) / np.max(hist) if np.max(hist) > 0 else np.nan
    except Exception:
        tri_hrv = np.nan

    mode_res = stats.mode(np.round(hr, 1), keepdims=True)
    mode_hr = float(mode_res.mode[0]) if len(mode_res.mode) > 0 else np.nan

    return {
        'meanHR': float(np.nanmean(hr)),
        'minHR': float(np.nanmin(hr)),
        'maxHR': float(np.nanmax(hr)),
        'sdHR': float(np.nanstd(hr)),
        'modeHR': mode_hr,
        'nNN': float(len(rri_ms)),
        'meanNN': float(np.nanmean(rri_ms)),
        'SDSD': float(np.nanstd(diff_rri)),
        'CVNN': float(np.nanstd(rri_ms) / np.nanmean(rri_ms)) if np.nanmean(rri_ms) > 0 else np.nan,
        'SDNN': float(np.nanstd(rri_ms)),
        'pNN50': float(100.0 * np.sum(np.abs(diff_rri) > 50) / len(diff_rri)) if len(diff_rri) > 0 else np.nan,
        'pNN20': float(100.0 * np.sum(np.abs(diff_rri) > 20) / len(diff_rri)) if len(diff_rri) > 0 else np.nan,
        'RMSSD': float(np.sqrt(np.nanmean(diff_rri ** 2))) if len(diff_rri) > 0 else np.nan,
        'medianNN': float(np.nanquantile(rri_ms, 0.5)),
        'q20NN': float(np.nanquantile(rri_ms, 0.2)),
        'q80NN': float(np.nanquantile(rri_ms, 0.8)),
        'minNN': float(np.nanmin(rri_ms)),
        'maxNN': float(np.nanmax(rri_ms)),
        'triHRV': float(tri_hrv),
    }


def ecg_freq(r_peaks: np.ndarray, sampling_rate: float = 500.0, upsample_rate: float = 4.0) -> dict:
    cols = ['totalpower', 'LF', 'HF', 'ULF', 'VLF', 'VHF', 'LF/HF', 'rLF', 'rHF', 'peakLF', 'peakHF']
    x = np.asarray(r_peaks, dtype=np.float64)
    if len(x) < 5:
        return {c: np.nan for c in cols}

    rri_ms = 1000.0 * np.diff(x) / sampling_rate
    time_points = np.cumsum(rri_ms) / 1000.0
    time_points -= time_points[0]
    total_time = time_points[-1]

    if total_time < 5.0:
        return {c: np.nan for c in cols}

    try:
        interp_func = interp1d(time_points, rri_ms, kind='cubic', fill_value="extrapolate")
        new_time = np.arange(0, total_time, 1.0 / upsample_rate)
        rri_resampled = interp_func(new_time)

        nperseg = min(len(rri_resampled), int(256 * upsample_rate / 4))
        if nperseg < 8:
            nperseg = len(rri_resampled)

        freqs, psd = signal.welch(rri_resampled, fs=upsample_rate, nperseg=nperseg)

        ulf_band = (freqs <= 0.0033)
        vlf_band = (freqs > 0.0033) & (freqs <= 0.04)
        lf_band = (freqs > 0.04) & (freqs <= 0.15)
        hf_band = (freqs > 0.15) & (freqs <= 0.4)
        vhf_band = (freqs > 0.4)

        def trapz_safe(y, x):
            if len(y) < 2:
                return 0.0
            return float(trapz(y, x))

        ulf = trapz_safe(psd[ulf_band], freqs[ulf_band])
        vlf = trapz_safe(psd[vlf_band], freqs[vlf_band])
        lf = trapz_safe(psd[lf_band], freqs[lf_band])
        hf = trapz_safe(psd[hf_band], freqs[hf_band])
        vhf = trapz_safe(psd[vhf_band], freqs[vhf_band])
        total_p = ulf + vlf + lf + hf + vhf

        lf_hf = lf / hf if hf > 1e-12 else np.nan
        denom = (lf + hf) if (lf + hf) > 1e-12 else np.nan
        rlf = 100.0 * lf / denom if pd.notnull(denom) else np.nan
        rhf = 100.0 * hf / denom if pd.notnull(denom) else np.nan

        peak_lf = float(freqs[lf_band][np.argmax(psd[lf_band])]) if np.any(lf_band) and len(psd[lf_band]) > 0 else np.nan
        peak_hf = float(freqs[hf_band][np.argmax(psd[hf_band])]) if np.any(hf_band) and len(psd[hf_band]) > 0 else np.nan

        return {
            'totalpower': total_p,
            'LF': lf,
            'HF': hf,
            'ULF': ulf,
            'VLF': vlf,
            'VHF': vhf,
            'LF/HF': lf_hf,
            'rLF': rlf,
            'rHF': rhf,
            'peakLF': peak_lf,
            'peakHF': peak_hf,
        }
    except Exception:
        return {c: np.nan for c in cols}


def ecg_nonlinear(r_peaks: np.ndarray, sampling_rate: float = 500.0) -> dict:
    cols = ['SD1', 'SD2', 'SD1SD2', 'apEn', 'sampEn']
    x = np.asarray(r_peaks, dtype=np.float64)
    if len(x) < 5:
        return {c: np.nan for c in cols}

    rri_ms = 1000.0 * np.diff(x) / sampling_rate
    diff_rri = np.diff(rri_ms)

    sd1 = float(np.sqrt(0.5 * np.nanvar(diff_rri)))
    var_rri = float(np.nanvar(rri_ms))
    sd2 = float(np.sqrt(max(0.0, 2.0 * var_rri - 0.5 * np.nanvar(diff_rri))))
    sd1_sd2 = sd1 / sd2 if sd2 > 1e-12 else np.nan

    # Approximate entropy & sample entropy
    def _approx_entropy(u, m=2, r=0.2):
        u = np.asarray(u)
        N = len(u)
        if N < m + 2:
            return np.nan
        r = r * np.std(u)
        def _phi(m_len):
            z = np.array([u[i:i + m_len] for i in range(N - m_len + 1)])
            C = np.sum(np.max(np.abs(z[:, None] - z[None, :]), axis=2) <= r, axis=0) / (N - m_len + 1)
            return np.sum(np.log(np.maximum(C, 1e-12))) / (N - m_len + 1)
        try:
            return float(abs(_phi(m) - _phi(m + 1)))
        except Exception:
            return np.nan

    def _sample_entropy(u, m=2, r=0.2):
        u = np.asarray(u)
        N = len(u)
        if N < m + 2:
            return np.nan
        r = r * np.std(u)
        def _count(m_len):
            z = np.array([u[i:i + m_len] for i in range(N - m_len + 1)])
            d = np.max(np.abs(z[:, None] - z[None, :]), axis=2)
            np.fill_diagonal(d, np.inf)
            return np.sum(d <= r)
        try:
            A = _count(m + 1)
            B = _count(m)
            return float(-np.log(A / B)) if A > 0 and B > 0 else np.nan
        except Exception:
            return np.nan

    apen = _approx_entropy(rri_ms)
    sampen = _sample_entropy(rri_ms)

    return {
        'SD1': sd1,
        'SD2': sd2,
        'SD1SD2': sd1_sd2,
        'apEn': apen,
        'sampEn': sampen,
    }


def extract_ecg_window_features(clean_ecg: np.ndarray, fs: float = 500.0) -> dict:
    """
    Extract full ECG features for a window.
    Returns dictionary with all 46 features.
    """
    stats_feat = ecg_stat(clean_ecg)
    # Detect R peaks in window
    distance = int(fs / 3.0)
    q99 = np.quantile(clean_ecg, 0.99)
    height = q99 / 2.0
    r_peaks, _ = signal.find_peaks(clean_ecg, distance=distance, height=height)

    duration_s = len(clean_ecg) / fs
    time_feat = ecg_time(r_peaks, duration_s=duration_s, sampling_rate=fs)
    freq_feat = ecg_freq(r_peaks, sampling_rate=fs)
    nonlin_feat = ecg_nonlinear(r_peaks, sampling_rate=fs)

    all_feat = {}
    all_feat.update(time_feat)
    all_feat.update(stats_feat)
    all_feat.update(freq_feat)
    all_feat.update(nonlin_feat)
    return all_feat
