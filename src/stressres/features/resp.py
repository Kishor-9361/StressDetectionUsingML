"""
src/stressres/features/resp.py

Respiration feature extraction matching the StressID baseline protocol.
Extracts 62 features (38 time/breathing interval, 11 statistical, 9 frequency, 4 nonlinear).
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.integrate import trapz


def rsp_stat(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        'max_rsp': float(np.nanmax(x)),
        'min_rsp': float(np.nanmin(x)),
        'mean_rsp': float(np.nanmean(x)),
        'sd_rsp': float(np.nanstd(x)),
        'ku_rsp': float(stats.kurtosis(x, nan_policy="omit")),
        'sk_rsp': float(stats.skew(x, nan_policy="omit")),
        'median_rsp': float(np.nanquantile(x, 0.5)),
        'q1_rsp': float(np.nanquantile(x, 0.25)),
        'q3_rsp': float(np.nanquantile(x, 0.75)),
        'q05_rsp': float(np.nanquantile(x, 0.05)),
        'q95_rsp': float(np.nanquantile(x, 0.95)),
    }


def rsp_time(x: np.ndarray, sampling_rate: float = 500.0) -> dict:
    cols = ['meanRR', 'minRR', 'maxRR', 'sdRR', 'modeRR', 'nBB', 'meanBB', 'SDSDb', 'CVNNb',
            'SDNNb', 'RMSSDb', 'medianBB', 'q20BB', 'q80BB', 'minBB', 'maxBB', 'meanTT', 'SDTT',
            'medianTT', 'q20TT', 'q80TT', 'minTT', 'maxTT', 'meanBA', 'SDBA', 'medianBA',
            'q20BA', 'q80BA', 'minBA', 'maxBA', 'meanBW', 'SDBW', 'medianBW', 'q20BW', 'q80BW',
            'minBW', 'maxBW']

    x = np.asarray(x, dtype=np.float64)
    if len(x) < int(sampling_rate * 5):
        return {c: np.nan for c in cols}

    # Peaks (inhalation peaks)
    distance = int(sampling_rate * 3)  # healthy person takes <= 20 breaths/min
    height = float(x.std() / 4.0)
    r_peaks, _ = signal.find_peaks(x, distance=distance, height=height)

    # Troughs (exhalation troughs)
    r_troughs, _ = signal.find_peaks(-x, distance=distance, height=height)

    if len(r_peaks) < 2:
        return {c: np.nan for c in cols}

    bbi = np.diff(r_peaks)
    bbi_ms = 1000.0 * bbi / sampling_rate
    dbbi = np.diff(bbi_ms)
    rr = 1000.0 * 60.0 / bbi_ms

    dur_min = (len(x) / sampling_rate) / 60.0
    n_bb = len(bbi_ms) / dur_min if dur_min > 0 else np.nan

    mean_rr = float(np.nanmean(rr))
    min_rr = float(np.nanmin(rr))
    max_rr = float(np.nanmax(rr))
    sd_rr = float(np.nanstd(rr))
    mode_rr = max_rr - min_rr

    mean_bb = float(np.nanmean(bbi_ms))
    sdsd_b = float(np.nanstd(dbbi)) if len(dbbi) > 0 else np.nan
    cvnn_b = float(np.nanstd(bbi_ms))
    sdnn_b = cvnn_b / mean_bb if mean_bb > 0 else np.nan
    rmssd_b = float(np.sqrt(np.nanmean(dbbi ** 2))) if len(dbbi) > 0 else np.nan

    med_bb = float(np.nanquantile(bbi_ms, 0.5))
    q20_bb = float(np.nanquantile(bbi_ms, 0.2))
    q80_bb = float(np.nanquantile(bbi_ms, 0.8))
    min_bb = float(np.nanmin(bbi_ms))
    max_bb = float(np.nanmax(bbi_ms))

    # TT intervals (troughs)
    if len(r_troughs) >= 2:
        tti = 1000.0 * np.diff(r_troughs) / sampling_rate
        mean_tt = float(np.nanmean(tti))
        sd_tt = float(np.nanstd(tti))
        med_tt = float(np.nanquantile(tti, 0.5))
        q20_tt = float(np.nanquantile(tti, 0.2))
        q80_tt = float(np.nanquantile(tti, 0.8))
        min_tt = float(np.nanmin(tti))
        max_tt = float(np.nanmax(tti))
    else:
        mean_tt = sd_tt = med_tt = q20_tt = q80_tt = min_tt = max_tt = np.nan

    # Breath amplitudes
    if len(r_peaks) > 0 and len(r_troughs) > 0:
        ba = np.subtract(x[r_peaks], np.mean(x[r_troughs]))
        mean_ba = float(np.nanmean(ba))
        sd_ba = float(np.nanstd(ba))
        med_ba = float(np.nanquantile(ba, 0.5))
        q20_ba = float(np.nanquantile(ba, 0.2))
        q80_ba = float(np.nanquantile(ba, 0.8))
        min_ba = float(np.nanmin(ba))
        max_ba = float(np.nanmax(ba))

        # Breath widths
        bw = np.sort(np.concatenate((r_peaks, r_troughs)))
        bw_diff = 1000.0 * np.diff(bw) / sampling_rate
        mean_bw = float(np.nanmean(bw_diff))
        sd_bw = float(np.nanstd(bw_diff))
        med_bw = float(np.nanquantile(bw_diff, 0.5))
        q20_bw = float(np.nanquantile(bw_diff, 0.2))
        q80_bw = float(np.nanquantile(bw_diff, 0.8))
        min_bw = float(np.nanmin(bw_diff))
        max_bw = float(np.nanmax(bw_diff))
    else:
        mean_ba = sd_ba = med_ba = q20_ba = q80_ba = min_ba = max_ba = np.nan
        mean_bw = sd_bw = med_bw = q20_bw = q80_bw = min_bw = max_bw = np.nan

    return {
        'meanRR': mean_rr, 'minRR': min_rr, 'maxRR': max_rr, 'sdRR': sd_rr, 'modeRR': mode_rr,
        'nBB': n_bb, 'meanBB': mean_bb, 'SDSDb': sdsd_b, 'CVNNb': cvnn_b, 'SDNNb': sdnn_b,
        'RMSSDb': rmssd_b, 'medianBB': med_bb, 'q20BB': q20_bb, 'q80BB': q80_bb,
        'minBB': min_bb, 'maxBB': max_bb, 'meanTT': mean_tt, 'SDTT': sd_tt, 'medianTT': med_tt,
        'q20TT': q20_tt, 'q80TT': q80_tt, 'minTT': min_tt, 'maxTT': max_tt,
        'meanBA': mean_ba, 'SDBA': sd_ba, 'medianBA': med_ba, 'q20BA': q20_ba, 'q80BA': q80_ba,
        'minBA': min_ba, 'maxBA': max_ba, 'meanBW': mean_bw, 'SDBW': sd_bw, 'medianBW': med_bw,
        'q20BW': q20_bw, 'q80BW': q80_bw, 'minBW': min_bw, 'maxBW': max_bw,
    }


def rsp_freq(x: np.ndarray, sampling_rate: float = 500.0, upsample_rate: float = 6.0,
             freqband_limits=(0.01, 0.05, 0.15, 0.5, 0.8)) -> dict:
    cols = ['totalpower_rsp', 'LF_rsp', 'HF_rsp', 'VLF_rsp', 'VHF_rsp', 'LF/HF_rsp',
            'rLF_rsp', 'rHF_rsp', 'peakLF_rsp', 'peakHF_rsp']

    x = np.asarray(x, dtype=np.float64)
    distance = int(sampling_rate * 3)
    height = float(x.std() / 4.0)
    r_peaks, _ = signal.find_peaks(x, distance=distance, height=height)

    if len(r_peaks) < 4:
        return {c: np.nan for c in cols}

    try:
        bb = np.diff(r_peaks)
        bb_time = np.cumsum(bb) / sampling_rate
        interp_f = interp1d(bb_time, bb, kind='cubic', fill_value="extrapolate")
        new_time = np.arange(1.0, bb_time.max(), 1.0 / upsample_rate)
        if len(new_time) < 8:
            return {c: np.nan for c in cols}

        rri = interp_f(new_time)
        freq, power = signal.welch(x=rri, fs=upsample_rate, nperseg=min(len(rri), 256))

        lim_vlf = (freq >= freqband_limits[0]) & (freq < freqband_limits[1])
        lim_lf = (freq >= freqband_limits[1]) & (freq < freqband_limits[2])
        lim_hf = (freq >= freqband_limits[2]) & (freq < freqband_limits[3])
        lim_vhf = (freq >= freqband_limits[3]) & (freq < freqband_limits[4])

        def trapz_safe(y, x_ax):
            if len(y) < 2:
                return 0.0
            return float(trapz(y, x_ax))

        vlf = trapz_safe(power[lim_vlf], freq[lim_vlf])
        lf = trapz_safe(power[lim_lf], freq[lim_lf])
        hf = trapz_safe(power[lim_hf], freq[lim_hf])
        vhf = trapz_safe(power[lim_vhf], freq[lim_vhf])

        total_p = vlf + lf + hf + vhf
        lf_hf = lf / hf if hf > 1e-12 else np.nan
        denom = (lf + hf) if (lf + hf) > 1e-12 else np.nan
        rlf = 100.0 * lf / denom if pd.notnull(denom) else np.nan
        rhf = 100.0 * hf / denom if pd.notnull(denom) else np.nan

        peak_lf = float(freq[lim_lf][np.argmax(power[lim_lf])]) if np.any(lim_lf) and len(power[lim_lf]) > 0 else np.nan
        peak_hf = float(freq[lim_hf][np.argmax(power[lim_hf])]) if np.any(lim_hf) and len(power[lim_hf]) > 0 else np.nan

        return {
            'totalpower_rsp': total_p, 'LF_rsp': lf, 'HF_rsp': hf, 'VLF_rsp': vlf, 'VHF_rsp': vhf,
            'LF/HF_rsp': lf_hf, 'rLF_rsp': rlf, 'rHF_rsp': rhf, 'peakLF_rsp': peak_lf, 'peakHF_rsp': peak_hf
        }
    except Exception:
        return {c: np.nan for c in cols}


def rsp_nonlinear(x: np.ndarray, sampling_rate: float = 500.0) -> dict:
    cols = ['SD1_rrv', 'SD2_rrv', 'SD1SD2_rrv', 'apEn_rrv']
    x = np.asarray(x, dtype=np.float64)
    distance = int(sampling_rate * 3)
    height = float(x.std() / 4.0)
    r_peaks, _ = signal.find_peaks(x, distance=distance, height=height)

    if len(r_peaks) < 4:
        return {c: np.nan for c in cols}

    bbi_ms = 1000.0 * np.diff(r_peaks) / sampling_rate
    diff_bbi = np.diff(bbi_ms)

    sd1 = float(np.sqrt(0.5 * np.nanvar(diff_bbi)))
    var_bbi = float(np.nanvar(bbi_ms))
    sd2 = float(np.sqrt(max(0.0, 2.0 * var_bbi - 0.5 * np.nanvar(diff_bbi))))
    sd1_sd2 = sd1 / sd2 if sd2 > 1e-12 else np.nan

    # Approximate entropy
    def _apen(u, m=2, r=0.2):
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

    apen = _apen(bbi_ms)

    return {
        'SD1_rrv': sd1,
        'SD2_rrv': sd2,
        'SD1SD2_rrv': sd1_sd2,
        'apEn_rrv': apen,
    }


def extract_resp_window_features(clean_resp: np.ndarray, fs: float = 500.0) -> dict:
    """
    Extract full Respiration features for a window.
    Returns dictionary with all 62 features.
    """
    stat_feat = rsp_stat(clean_resp)
    time_feat = rsp_time(clean_resp, sampling_rate=fs)
    freq_feat = rsp_freq(clean_resp, sampling_rate=fs)
    nonlin_feat = rsp_nonlinear(clean_resp, sampling_rate=fs)

    all_feat = {}
    all_feat.update(time_feat)
    all_feat.update(stat_feat)
    all_feat.update(freq_feat)
    all_feat.update(nonlin_feat)
    return all_feat
