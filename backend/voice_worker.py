import numpy as np
import librosa
import io
import tempfile
import os
import subprocess
import sys

def extract_f0_yin(y, sr, f0_min=65, f0_max=450, frame_len=1024, hop_len=160):
    """
    YIN algorithm for F0 extraction.
    frame_len=1024 ensures accurate detection of low pitches down to 50 Hz without period truncation.
    """
    try:
        f0_yin = librosa.yin(
            y, 
            fmin=f0_min, 
            fmax=f0_max, 
            sr=sr,
            frame_length=frame_len,
            hop_length=hop_len
        )
        # Compute RMS energy per frame
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        if len(rms) < len(f0_yin):
            rms = np.pad(rms, (0, len(f0_yin) - len(rms)), mode='edge')
        elif len(rms) > len(f0_yin):
            rms = rms[:len(f0_yin)]
            
        # Voice activity threshold: ambient room noise is typically 0.005 - 0.020.
        # Human speech into mic typically has RMS >= 0.028.
        speech_thresh = max(0.025, float(np.median(rms) * 1.5))
        voiced_flag = (rms > speech_thresh) & (f0_yin > f0_min) & (f0_yin < f0_max)
        
        f0_track = np.copy(f0_yin)
        f0_track[~voiced_flag] = np.nan
        
        return f0_track, voiced_flag
    except Exception as e:
        print(f"librosa.yin failed: {e}")
        num_frames = 1 + (len(y) - frame_len) // hop_len
        if num_frames < 1: num_frames = 1
        return np.full(num_frames, np.nan), np.zeros(num_frames, dtype=bool)

def extract_voice_stress_indicators(audio_bytes, sr_target=16000, f0_min=75, f0_max=400):
    """
    Extract 12 acoustic stress biomarkers from a raw audio chunk.
    Designed for 1-3 second chunks. Fast, lightweight, generalizable.
    
    Accepts raw audio bytes (wav, webm, ogg, mp3) from Flask request.
    Returns: dict with 12 named indicators + numpy array for model input
    """
    y, sr = None, None
    EPS = 1e-10

    # Try loading directly via scipy.io.wavfile first (fast and deadlock-free on Windows)
    try:
        from scipy.io import wavfile
        from scipy import signal
        audio_buf = io.BytesIO(audio_bytes)
        sr_orig, y_orig = wavfile.read(audio_buf)
        
        # Convert to float32 normalized to [-1.0, 1.0]
        if y_orig.dtype == np.int16:
            y_float = y_orig.astype(np.float32) / 32768.0
        elif y_orig.dtype == np.int32:
            y_float = y_orig.astype(np.float32) / 2147483648.0
        elif y_orig.dtype == np.uint8:
            y_float = (y_orig.astype(np.float32) - 128.0) / 128.0
        else:
            y_float = y_orig.astype(np.float32)
        
        # Convert stereo to mono
        if len(y_float.shape) > 1:
            y_float = np.mean(y_float, axis=1)
            
        # Resample if sample rate doesn't match target
        if sr_orig != sr_target:
            num_samples = int(len(y_float) * sr_target / sr_orig)
            y = signal.resample(y_float, num_samples)
            sr = sr_target
        else:
            y = y_float
            sr = sr_target
            
        # Truncate to maximum 3.0 seconds duration
        max_samples = int(sr * 3.0)
        if len(y) > max_samples:
            y = y[:max_samples]
            
        print("Successfully loaded WAV audio using scipy.io.wavfile first", flush=True)
    except Exception as e_scipy:
        try:
            # Direct in-process load with soundfile or librosa (fast, memory-based)
            import soundfile as sf
            audio_buf = io.BytesIO(audio_bytes)
            y_orig, sr_orig = sf.read(audio_buf, dtype='float32')
            if len(y_orig.shape) > 1:
                y_orig = np.mean(y_orig, axis=1)
            if sr_orig != sr_target:
                from scipy import signal
                num_samples = int(len(y_orig) * sr_target / sr_orig)
                y = signal.resample(y_orig, num_samples)
            else:
                y = y_orig
            sr = sr_target
            max_samples = int(sr * 3.0)
            if len(y) > max_samples:
                y = y[:max_samples]
            print("Successfully loaded audio using in-process soundfile fallback", flush=True)
        except Exception as e_sf:
            print(f"soundfile fallback failed: {e_sf}. Trying librosa.load...", flush=True)
            try:
                temp_in_path = None
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_tmp:
                    f_tmp.write(audio_bytes)
                    temp_in_path = f_tmp.name
                y, sr = librosa.load(temp_in_path, sr=sr_target, mono=True, duration=3.0)
                if temp_in_path and os.path.exists(temp_in_path):
                    os.remove(temp_in_path)
                print("Successfully loaded audio using librosa.load", flush=True)
            except Exception as e_lib:
                print(f"librosa.load failed: {e_lib}", flush=True)
                y = None
    if y is None:
        print("voice_worker returning None: y is None")
        return None
    if len(y) < sr_target * 0.25:  # Relaxed to 0.25s
        print(f"voice_worker returning None: len(y)={len(y)} < {sr_target * 0.25} (duration {len(y)/sr_target:.2f}s)")
        return None
    # Removed peak amplitude check since app.py handles silence detection
    # Ensure loaded audio is normalized to float [-1.0, 1.0] and clamped
    if y.dtype != np.float32 and y.dtype != np.float64:
        y = y.astype(np.float32) / 32768.0
    y = np.clip(y, -1.0, 1.0)

    indicators = {}

    frame_len = int(sr * 0.025)  # 25ms frames
    hop_len   = int(sr * 0.010)  # 10ms hop
    
    # Check harmonic speech periodicity via normalized autocorrelation
    # Human voiced speech has strong periodic peaks (> 0.38) at pitch lags (50-450Hz).
    # Ambient room noise and laptop fans have chaotic, flat autocorrelation (< 0.25).
    try:
        ac_full = np.correlate(y, y, mode='full')[len(y) - 1:]
        ac_norm = ac_full / (ac_full[0] + EPS)
        min_lag = max(1, int(sr / max(f0_max, 450)))
        max_lag = min(len(ac_norm) - 1, int(sr / max(f0_min, 50)))
        harmonic_peak = float(np.max(ac_norm[min_lag:max_lag])) if max_lag > min_lag else 0.0
    except Exception:
        harmonic_peak = 0.0

    if harmonic_peak < 0.35:
        # Definitively silent room or non-speech background noise
        indicators['f0_mean'] = 0.0
        indicators['f0_std'] = 0.0
        indicators['f0_range'] = 0.0
        indicators['jitter_percent'] = 0.0
        indicators['jitter_reliable'] = True
        indicators['shimmer_db'] = 0.0
        indicators['speaking_rate_proxy'] = 0.0
        indicators['voiced_fraction'] = 0.0
        indicators['is_silent'] = True
    else:
        try:
            f0_track, voiced_flag = extract_f0_yin(y, sr, f0_min, f0_max, 1024, hop_len)
            f0_voiced = f0_track[~np.isnan(f0_track)]
            voiced_frac = float(np.sum(voiced_flag) / len(voiced_flag)) if len(voiced_flag) > 0 else 0.0
            indicators['voiced_fraction'] = voiced_frac

            # If less than 15% of frames contain voiced speech or fewer than 5 voiced frames: SILENCE / UNVOICED
            if len(f0_voiced) < 5 or voiced_frac < 0.15:
                indicators['f0_mean'] = 0.0
                indicators['f0_std'] = 0.0
                indicators['f0_range'] = 0.0
                indicators['jitter_percent'] = 0.0
                indicators['jitter_reliable'] = True
                indicators['shimmer_db'] = 0.0
                indicators['speaking_rate_proxy'] = 0.0
                indicators['is_silent'] = True
            else:
                indicators['is_silent'] = False
                indicators['f0_mean']  = float(np.mean(f0_voiced))
                indicators['f0_std']   = float(np.std(f0_voiced))
                indicators['f0_range'] = float(np.ptp(f0_voiced))
                
                import scipy.signal
                f0_smoothed = scipy.signal.medfilt(f0_voiced, kernel_size=3)
                periods = sr / (f0_smoothed + 1e-10)
                period_diffs = np.abs(np.diff(periods))
                jitter_rap = float(np.mean(period_diffs) / (np.mean(periods) + 1e-10)) * 100
                indicators['jitter_percent']  = float(np.clip(jitter_rap, 0.0, 5.0))
                indicators['jitter_reliable'] = bool(jitter_rap < 3.0)
                
                # Shimmer: amplitude variation between consecutive voiced frames
                rms_all = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
                voiced_rms = rms_all[:len(voiced_flag)][voiced_flag]
                if len(voiced_rms) >= 3:
                    voiced_rms_smoothed = scipy.signal.medfilt(voiced_rms, kernel_size=3)
                    amp_ratios = voiced_rms_smoothed[1:] / (voiced_rms_smoothed[:-1] + 1e-10)
                    shimmer_db = float(np.mean(np.abs(20 * np.log10(amp_ratios + 1e-10))))
                    indicators['shimmer_db'] = float(np.clip(shimmer_db, 0.0, 5.0))
                else:
                    indicators['shimmer_db'] = 0.0
        except Exception as e:
            print(f"Feature extraction error: {e}")
            indicators['f0_mean'] = indicators['f0_std'] = indicators['f0_range'] = 0.0
            indicators['jitter_percent'] = 0.0
            indicators['jitter_reliable'] = True
            indicators['shimmer_db'] = 0.0
            indicators['voiced_fraction'] = 0.0
            indicators['is_silent'] = True
        
    # 6: HNR approximation via autocorrelation
    try:
        if indicators.get('is_silent', False):
            indicators['hnr'] = 0.0
        else:
            ac_full = np.correlate(y, y, mode='full')[len(y) - 1:]
            ac_norm = ac_full / (ac_full[0] + EPS)
            # Find first zero crossing to determine search range for fundamental period
            zc = np.where(ac_norm < 0)[0]
            first_zc = zc[0] if len(zc) > 0 else len(ac_norm)
            
            if first_zc < len(ac_norm):
                peak_val = np.max(ac_norm[first_zc:])
                peak_val = np.clip(peak_val, 1e-10, 0.9999) # prevent log10 domain errors
                hnr = 10 * np.log10(peak_val / (1 - peak_val + EPS) + EPS)
            else:
                hnr = 0.0
            indicators['hnr'] = float(np.clip(hnr, -20, 30))
    except Exception:
        indicators['hnr'] = 0.0

    # 7: Speaking rate proxy (ZCR)
    try:
        if indicators.get('is_silent', False):
            indicators['speaking_rate_proxy'] = 0.0
        else:
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
            indicators['speaking_rate_proxy'] = float(np.mean(zcr))
    except Exception:
        indicators['speaking_rate_proxy'] = 0.0

    # 8: Voice intensity (reports actual audio RMS even in silence)
    try:
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        indicators['voice_intensity'] = float(np.mean(rms))
    except Exception:
        indicators['voice_intensity'] = 0.0
    except Exception:
        indicators['voice_intensity'] = 0.0
        rms = np.array([0.0])

    # 9: High frequency ratio (stress elevates high-freq content)
    try:
        stft = np.abs(librosa.stft(y, n_fft=512, hop_length=hop_len))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        high_mask = freqs >= 3000
        total_energy = np.sum(stft) + EPS
        indicators['high_freq_ratio'] = float(np.sum(stft[high_mask]) / total_energy)
    except Exception:
        indicators['high_freq_ratio'] = 0.0
        stft = np.zeros((257, 1))

    # 10: Spectral flux
    try:
        spectral_flux = np.mean(np.diff(stft, axis=1) ** 2) if stft.shape[1] > 1 else 0.0
        indicators['spectral_flux'] = float(np.clip(spectral_flux, 0, 1))
    except Exception:
        indicators['spectral_flux'] = 0.0

    # 11: Pause ratio (near-silent frames)
    try:
        silence_thresh = 0.01 * np.max(np.abs(y))
        pause_frames = np.sum(rms < silence_thresh)
        indicators['pause_ratio'] = float(pause_frames / (len(rms) + EPS))
    except Exception:
        indicators['pause_ratio'] = 0.0

    # Feature vector for model (fixed order, 12 features)
    feature_vec = np.array([
        indicators['f0_mean'],
        indicators['f0_std'],
        indicators['f0_range'],
        indicators['jitter_percent'],
        indicators['shimmer_db'],
        indicators['hnr'],
        indicators['speaking_rate_proxy'],
        indicators['voice_intensity'],
        indicators['high_freq_ratio'],
        indicators['spectral_flux'],
        indicators['pause_ratio'],
        indicators['voiced_fraction'],
    ], dtype=np.float32)

    return {'indicators': indicators, 'features': feature_vec}
