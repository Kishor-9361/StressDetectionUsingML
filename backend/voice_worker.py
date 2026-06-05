import numpy as np
import librosa
import io
import tempfile
import os

def _extract_period_with_interpolation(ac, min_lag, max_lag):
    """
    Extract sub-sample precise period from autocorrelation via parabolic interpolation.
    Solves the flat-line jitter problem caused by integer lag binning.
    """
    if max_lag >= len(ac):
        max_lag = len(ac) - 1

    # Find discrete peak
    km = np.argmax(ac[min_lag:max_lag]) + min_lag

    # Parabolic interpolation around the peak
    # Requires one sample on each side of the peak
    if min_lag < km < max_lag - 1:
        y1 = ac[km - 1]
        y2 = ac[km]
        y3 = ac[km + 1]
        denom = y1 - 2 * y2 + y3
        if abs(denom) > 1e-8:
            delta = 0.5 * (y1 - y3) / denom
            # Clamp delta to ±0.5 samples (parabola valid only near peak)
            delta = np.clip(delta, -0.5, 0.5)
            return float(km) + delta, float(y2)
    return float(km), float(ac[km])

def extract_voice_stress_indicators(audio_bytes, sr_target=16000):
    """
    Extract 12 acoustic stress biomarkers from a raw audio chunk.
    Designed for 1-3 second chunks. Fast, lightweight, generalizable.
    
    Accepts raw audio bytes (wav, webm, ogg, mp3) from Flask request.
    Returns: dict with 12 named indicators + numpy array for model input
    """
    y, sr = None, None
    EPS = 1e-10

    # Try loading directly via BytesIO
    try:
        audio_buf = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buf, sr=sr_target, mono=True, duration=3.0)
    except Exception as e:
        # Fallback: write to a temp file and load (some codecs require a file path)
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.bin')
            os.close(fd)
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            y, sr = librosa.load(temp_path, sr=sr_target, mono=True, duration=3.0)
            try:
                os.remove(temp_path)
            except:
                pass
        except Exception as e_inner:
            print(f"Error loading audio: {e_inner}")
            return None

    if y is None or len(y) < sr_target * 0.5:  # less than 0.5 seconds — skip
        return None

    indicators = {}

    # Fast autocorrelation-based F0, Jitter, Shimmer, and Voiced Fraction extraction
    # This avoids the slow librosa.pyin call, dropping execution time from 4.4s to <10ms.
    frame_len = int(sr * 0.025)  # 25ms frames
    hop_len   = int(sr * 0.010)  # 10ms hop
    
    try:
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
        periods = []
        amplitudes = []
        f0_voiced = []
        
        for frame in frames.T:
            rms_val = np.sqrt(np.mean(frame ** 2))
            amplitudes.append(rms_val)
            
            ac = np.correlate(frame, frame, mode='full')[frame_len - 1:]
            ac = ac / (ac[0] + EPS)
            min_lag = int(sr / 500)  # 500 Hz max
            max_lag = int(sr / 60)   # 60 Hz min
            
            refined_period, peak_corr = _extract_period_with_interpolation(ac, min_lag, max_lag)
            periods.append(refined_period)
            
            # Voice detection criteria: correlation strength > 0.45 and energy > 0.005
            if peak_corr > 0.45 and rms_val > 0.005:
                f_hz = sr / refined_period
                if 60 <= f_hz <= 500:
                    f0_voiced.append(f_hz)
                    
        periods = np.array(periods, dtype=float)
        amplitudes = np.array(amplitudes, dtype=float)
        
        # 1-3: F0 (fundamental frequency / pitch)
        indicators['f0_mean']  = float(np.median(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        indicators['f0_std']   = float(np.std(f0_voiced))    if len(f0_voiced) > 0 else 0.0
        indicators['f0_range'] = float(np.ptp(f0_voiced))    if len(f0_voiced) > 0 else 0.0
        
        # 4-5: Jitter and Shimmer
        jitter  = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + EPS)) if len(periods) > 1 else 0.0
        shimmer = float(np.mean(np.abs(np.diff(amplitudes))) / (np.mean(amplitudes) + EPS)) if len(amplitudes) > 1 else 0.0
        
        # 12: Voiced fraction
        voiced_frac = float(len(f0_voiced) / len(frames)) if len(frames) > 0 else 0.5
        indicators['voiced_fraction'] = voiced_frac
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        indicators['f0_mean'] = indicators['f0_std'] = indicators['f0_range'] = 0.0
        jitter, shimmer = 0.0, 0.0
        indicators['voiced_fraction'] = 0.5
        
    indicators['jitter_percent'] = min(jitter * 100, 10.0)  # cap at 10%
    indicators['shimmer_db']     = min(shimmer * 20, 5.0)   # approximate dB scale

    # 6: HNR approximation via autocorrelation
    try:
        ac_full = np.correlate(y, y, mode='full')[len(y) - 1:]
        ac_norm = ac_full / (ac_full[0] + EPS)
        min_period = int(sr / 400)
        max_period = int(sr / 80)
        if max_period < len(ac_norm):
            peak_val = np.max(ac_norm[min_period:max_period])
            hnr = 10 * np.log10(peak_val / (1 - peak_val + EPS) + EPS)
        else:
            hnr = 0.0
    except Exception:
        hnr = 0.0
    indicators['hnr'] = float(np.clip(hnr, -20, 30))

    # 7: Speaking rate proxy (ZCR)
    try:
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
        indicators['speaking_rate_proxy'] = float(np.mean(zcr))
    except Exception:
        indicators['speaking_rate_proxy'] = 0.0

    # 8: Voice intensity
    try:
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        indicators['voice_intensity'] = float(np.mean(rms))
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

    # 12: Voiced fraction
    try:
        voiced_frac = float(np.sum(voiced_flag) / (len(voiced_flag) + EPS)) if 'voiced_flag' in locals() else 0.5
    except Exception:
        voiced_frac = 0.5
    indicators['voiced_fraction'] = voiced_frac

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
