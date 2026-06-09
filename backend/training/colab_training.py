"""
Google Colab Multimodal Stress Detection Expert Training Script.
This script is self-contained. Copy-paste it into a Google Colab notebook cell
or run it directly in Colab after mounting Google Drive.

Expected Drive Directory Structure:
/content/drive/MyDrive/Multimodal_stress_Detection/
  ├── facesData/
  │     ├── train/
  │     │     ├── stress/
  │     │     └── nostress/
  │     └── test/
  │           ├── stress/
  │           └── nostress/
  └── StressID/
        └── StressID Dataset/
              ├── labels.csv
              └── Audio/
                    ├── Subject01/
                    ├── Subject02/
                    └── ... (nested subject directories)
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
import io
import sys
import csv
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Check if running in Google Colab environment by checking the directory path
# We assume Google Drive has already been mounted by the user in a previous cell.
if os.path.exists('/content/drive'):
    print("Google Drive mount detected. Using Colab paths...")
    BASE_DRIVE_DIR = '/content/drive/MyDrive/Multimodal_stress_Detection'
else:
    print("Google Drive mount not detected. Running in local environment...")
    BASE_DRIVE_DIR = r'f:\Multimodal_stress_Detection'

# Configure paths based on BASE_DRIVE_DIR
FACES_ROOT = os.path.join(BASE_DRIVE_DIR, 'facesData')
STRESSID_ROOT = os.path.join(BASE_DRIVE_DIR, 'StressID', 'StressID Dataset')
STRESSID_AUDIO = os.path.join(STRESSID_ROOT, 'Audio')
OUTPUT_DIR = os.path.join(BASE_DRIVE_DIR, 'expert_models')

# Setup Outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)
FACE_TRAIN_CSV = os.path.join(BASE_DRIVE_DIR, 'face_indicators_train.csv')
FACE_TEST_CSV  = os.path.join(BASE_DRIVE_DIR, 'face_indicators_test.csv')
FACE_STRESSID_CSV = os.path.join(BASE_DRIVE_DIR, 'face_indicators_stressid.csv')
VOICE_STRESSID_CSV = os.path.join(BASE_DRIVE_DIR, 'voice_indicators_stressid.csv')

FACE_MODEL_PATH = os.path.join(OUTPUT_DIR, 'face_expert_lightweight.pkl')
FACE_SCALER_PATH = os.path.join(OUTPUT_DIR, 'face_scaler_lightweight.pkl')

VOICE_MODEL_PATH = os.path.join(OUTPUT_DIR, 'voice_expert_lightweight.pkl')
VOICE_SCALER_PATH = os.path.join(OUTPUT_DIR, 'voice_scaler_lightweight.pkl')

# -----------------------------------------------------------------------------
# 1. STANDALONE FEATURE EXTRACTORS
# -----------------------------------------------------------------------------

# --- Face Feature Extractor (using mediapipe) ---
MEDIAPIPE_AVAILABLE = False
USE_LEGACY_MEDIAPIPE = False

try:
    import cv2
    import mediapipe as mp
    # Check if we can use legacy solutions
    try:
        import mediapipe.solutions.face_mesh as mp_face_mesh
        USE_LEGACY_MEDIAPIPE = True
        MEDIAPIPE_AVAILABLE = True
        print("Using Legacy MediaPipe solutions API.")
    except (ImportError, AttributeError):
        # Fallback to Tasks API
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        USE_LEGACY_MEDIAPIPE = False
        MEDIAPIPE_AVAILABLE = True
        print("Using Modern MediaPipe Tasks API fallback.")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"Warning: mediapipe or opencv-python import failed: {e}. Face mesh feature extraction will not work.")

# --- FaceMeshWrapper to support both Legacy solutions and Modern Tasks API ---
class FaceMeshWrapper:
    def __init__(self, static_mode=True):
        self.static_mode = static_mode
        self.use_tasks = not USE_LEGACY_MEDIAPIPE
        self.fm = None
        self.detector = None
        
        if self.use_tasks:
            self.model_path = "face_landmarker.task"
            if not os.path.exists(self.model_path):
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                print("Downloading Face Landmarker model asset for Tasks API...")
                urllib.request.urlretrieve(url, self.model_path)
            
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        else:
            import mediapipe.solutions.face_mesh as mp_face_mesh
            self.fm = mp_face_mesh.FaceMesh(
                static_image_mode=self.static_mode,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5
            )

    def process(self, rgb_image):
        if self.use_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            res = self.detector.detect(mp_image)
            
            class LegacyLandmarkResult:
                def __init__(self, landmarks):
                    self.landmark = landmarks
                    
            class LegacyResult:
                def __init__(self, face_landmarks):
                    self.multi_face_landmarks = [LegacyLandmarkResult(face_landmarks[0])]
                    
            class LegacyResultEmpty:
                multi_face_landmarks = None

            if res and res.face_landmarks and len(res.face_landmarks) > 0:
                return LegacyResult(res.face_landmarks)
            else:
                return LegacyResultEmpty()
        else:
            return self.fm.process(rgb_image)

    def close(self):
        if self.fm:
            self.fm.close()
        if self.detector:
            self.detector.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def dist(pts, a, b):
    return float(np.linalg.norm(pts[a] - pts[b]))

def compute_18_face_indicators(image_path):
    """Extract 18 face landmarks from an image path using Python MediaPipe"""
    if not MEDIAPIPE_AVAILABLE:
        raise ImportError("MediaPipe not installed.")
        
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    with FaceMeshWrapper(static_mode=True) as fm:
        res = fm.process(rgb)

    if not res.multi_face_landmarks:
        return None

    lm  = res.multi_face_landmarks[0].landmark
    pts = np.array([[l.x * w, l.y * h] for l in lm])

    faceH = dist(pts, 10, 152) + 1e-6
    faceW = dist(pts, 234, 454) + 1e-6
    iod   = dist(pts, 33, 263) + 1e-6

    earL = (dist(pts, 159, 145) + dist(pts, 158, 153)) / (2 * dist(pts, 33, 133) + 1e-6)
    earR = (dist(pts, 386, 374) + dist(pts, 385, 380)) / (2 * dist(pts, 362, 263) + 1e-6)
    avgEAR = (earL + earR) / 2

    return [
        earL,
        earR,
        avgEAR,
        0.0,                                                    # blink_velocity — static image, set 0
        dist(pts, 55, 159) / faceH,                             # brow_descent_left
        dist(pts, 285, 386) / faceH,                            # brow_descent_right
        abs(dist(pts, 55, 159) - dist(pts, 285, 386)) / faceH,    # brow_asymmetry
        dist(pts, 13, 14) / (dist(pts, 61, 291) + 1e-6),        # lip_compression
        dist(pts, 4, 152) / iod,                                # jaw_displacement (dynamic nose-to-chin distance)
        (dist(pts, 61, 4) + dist(pts, 291, 4)) / (2 * faceH),   # mouth_corner_pull
        dist(pts, 10, 151) / faceH,                             # forehead_tension
        faceH / iod,                                            # face_height_norm
        0.0,                                                    # head_tilt — static image, set 0
        0.0,                                                    # temporal_x_var — static image, set 0
        0.0,                                                    # temporal_y_var — static image, set 0
        avgEAR,                                                 # eye_openness_ratio
        0.9,                                                    # landmark_confidence (detected)
        dist(pts, 4, 50) / faceH,                               # nose_wrinkle
    ]

# --- Voice Feature Extractor (using librosa) ---
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa is not installed. Voice feature extraction will fail.")

def extract_voice_stress_indicators(audio_bytes, sr_target=16000):
    """
    Extract 12 acoustic stress biomarkers from raw audio bytes.
    These features represent the absolute gold standard for speech-based stress detection,
    mirroring how MediaPipe represents universal geometric facial patterns:
    
    1-3. F0 (Pitch) Mean, Std, & Range: Autonomic nervous system arousal increases 
         laryngeal muscle tension, elevating fundamental frequency (pitch) and its variance.
    4-5. Jitter & Shimmer: Micro-tremors in vocal cords during stress disrupt period stability 
         (frequency/jitter) and energy stability (amplitude/shimmer). Highly generalized.
    6. Harmonics-to-Noise Ratio (HNR): Measures voice breathiness/strain. Stress shifts
       vocal production to a more tense/turbulent state, decreasing harmonic strength.
    7. ZCR (Zero Crossing Rate proxy): Correlates with articulation velocity/speaking rate.
    8. Voice Intensity (RMS energy): Represents physical vocal effort and volume variations.
    9. High Frequency Ratio (energy >= 3000 Hz): Stress shifts spectral energy to higher 
       frequencies due to rapid glottal closure, producing a "tighter", sharper voice quality.
    10. Spectral Flux: Measures transition speed and stability of spectral frames.
    11-12. Pause Ratio & Voiced Fraction: Cognitive loading and anxiety manifest physically 
           as increased micro-pauses and hesitations (silence fraction) and reduced voicing.
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa not installed.")
        
    try:
        audio_buf = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buf, sr=sr_target, mono=True, duration=3.0)
    except Exception:
        return None

    if len(y) < sr_target * 0.5:
        return None

    indicators = {}
    EPS = 1e-10

    # 1-3: F0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr, frame_length=2048
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        indicators['f0_mean']  = float(np.mean(f0_voiced))  if len(f0_voiced) > 0 else 0.0
        indicators['f0_std']   = float(np.std(f0_voiced))   if len(f0_voiced) > 0 else 0.0
        indicators['f0_range'] = float(np.ptp(f0_voiced))   if len(f0_voiced) > 0 else 0.0
    except Exception:
        indicators['f0_mean'] = indicators['f0_std'] = indicators['f0_range'] = 0.0

    # 4-5: Jitter and Shimmer
    frame_len = int(sr * 0.025)
    hop_len   = int(sr * 0.010)
    
    try:
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
        periods = []
        amplitudes = []
        for frame in frames.T:
            ac = np.correlate(frame, frame, mode='full')[frame_len - 1:]
            ac = ac / (ac[0] + EPS)
            min_lag = int(sr / 500)
            max_lag = int(sr / 60)
            if max_lag < len(ac):
                peak_idx = np.argmax(ac[min_lag:max_lag]) + min_lag
                periods.append(peak_idx)
            amplitudes.append(np.sqrt(np.mean(frame ** 2)))

        periods = np.array(periods, dtype=float)
        amplitudes = np.array(amplitudes, dtype=float)

        jitter  = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + EPS)) if len(periods) > 1 else 0.0
        shimmer = float(np.mean(np.abs(np.diff(amplitudes))) / (np.mean(amplitudes) + EPS)) if len(amplitudes) > 1 else 0.0
    except Exception:
        jitter, shimmer = 0.0, 0.0

    indicators['jitter_percent'] = min(jitter * 100, 10.0)
    indicators['shimmer_db']     = min(shimmer * 20, 5.0)

    # 6: HNR
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

    # 7: Speaking rate
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

    # 9: High frequency ratio
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

    # 11: Pause ratio
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


# -----------------------------------------------------------------------------
# 2. OFFLINE DATA EXTRACTION
# -----------------------------------------------------------------------------

def run_face_extraction():
    print("\n" + "="*50)
    print("STAGE 1: Face Landmark Feature Extraction")
    print("="*50)
    
    if not os.path.exists(FACES_ROOT):
        print(f"Error: facesData root directory not found at: {FACES_ROOT}")
        print("Please check your Drive paths.")
        return False

    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe is not installed or failed to import.")
        print("Please run '!pip install mediapipe opencv-python' in Colab before executing this script.")
        return False

    def process_split(split_name, output_csv):
        if os.path.exists(output_csv):
            print(f"Cache found for Face {split_name} split: {output_csv}. Skipping extraction...")
            return True
            
        rows = []
        for label_name, label_val in [('stress', 1), ('nostress', 0)]:
            folder = os.path.join(FACES_ROOT, split_name, label_name)
            if not os.path.exists(folder):
                print(f"Warning: {folder} not found. Skipping...")
                continue
                
            images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Processing {len(images)} images in {split_name}/{label_name}...")
            
            for i, img_name in enumerate(images):
                path = os.path.join(folder, img_name)
                try:
                    indicators = compute_18_face_indicators(path)
                    if indicators is not None:
                        rows.append(indicators + [label_val])
                except Exception as e:
                    pass
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i+1}/{len(images)} images...")

        if len(rows) == 0:
            print(f"Error: No features extracted for {split_name} split.")
            return False

        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
                'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
                'lip_compression', 'jaw_tension', 'mouth_corner_pull',
                'forehead_tension', 'face_height_norm', 'head_tilt',
                'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
                'landmark_confidence', 'nose_wrinkle', 'label',
            ]
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Saved {len(rows)} samples to {output_csv}")
        return True

    print("Extracting training split...")
    train_success = process_split('train', FACE_TRAIN_CSV)
    print("Extracting testing split...")
    test_success = process_split('test', FACE_TEST_CSV)
    
    return train_success and test_success

def run_face_extraction_stressid_videos():
    print("\n" + "="*50)
    print("STAGE 1: Face Landmark Feature Extraction from StressID Videos")
    print("="*50)
    
    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe is not installed or failed to import.")
        print("Please run '!pip install mediapipe opencv-python' in Colab before executing this script.")
        return False
    
    labels_csv = os.path.join(STRESSID_ROOT, 'labels.csv')
    if not os.path.exists(labels_csv):
        labels_csv_fallback = os.path.join(STRESSID_ROOT, '._labels.csv')
        if os.path.exists(labels_csv_fallback):
            labels_csv = labels_csv_fallback
            
    if not os.path.exists(labels_csv):
        print(f"Error: labels.csv not found at {labels_csv}")
        return False
        
    try:
        labels_df = pd.read_csv(labels_csv)
    except Exception as e:
        print(f"Error reading labels CSV: {e}")
        return False
        
    output_csv = FACE_STRESSID_CSV
    if os.path.exists(output_csv):
        print(f"Cache found for StressID Face features: {output_csv}. Skipping extraction...")
        return True
        
    rows = []
    print(f"Processing videos from {STRESSID_ROOT} based on labels...")
    
    with FaceMeshWrapper(static_mode=False) as fm:
        for idx, row in labels_df.iterrows():
            subject_task = row.get('subject/task')
            if not subject_task:
                continue
            subject = subject_task.split('_')[0]
            label_val = row.get('binary-stress')
            if label_val is None or pd.isna(label_val):
                continue
                
            video_path = os.path.join(STRESSID_ROOT, 'Videos', subject, f"{subject_task}.mp4")
            if not os.path.exists(video_path):
                continue
                
            print(f"Processing: {subject_task}.mp4 ...")
            
            # Extract features from video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
                
            frame_indicators = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample 1 frame per second (assuming 30fps)
                if frame_count % 30 == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    res = fm.process(rgb)
                    if res.multi_face_landmarks:
                        lm = res.multi_face_landmarks[0].landmark
                        pts = np.array([[l.x * w, l.y * h] for l in lm])
                        
                        faceH = dist(pts, 10, 152) + 1e-6
                        faceW = dist(pts, 234, 454) + 1e-6
                        iod   = dist(pts, 33, 263) + 1e-6
                        
                        earL = (dist(pts, 159, 145) + dist(pts, 158, 153)) / (2 * dist(pts, 33, 133) + 1e-6)
                        earR = (dist(pts, 386, 374) + dist(pts, 385, 380)) / (2 * dist(pts, 362, 263) + 1e-6)
                        avgEAR = (earL + earR) / 2
                        
                        indicators = [
                            earL,
                            earR,
                            avgEAR,
                            0.0,                                                    # blink_velocity
                            dist(pts, 55, 159) / faceH,                             # brow_descent_left
                            dist(pts, 285, 386) / faceH,                            # brow_descent_right
                            abs(dist(pts, 55, 159) - dist(pts, 285, 386)) / faceH,    # brow_asymmetry
                            dist(pts, 13, 14) / (dist(pts, 61, 291) + 1e-6),        # lip_compression
                            faceW / faceH,                                          # jaw_tension
                            (dist(pts, 61, 4) + dist(pts, 291, 4)) / (2 * faceH),   # mouth_corner_pull
                            dist(pts, 10, 151) / faceH,                             # forehead_tension
                            faceH / iod,                                            # face_height_norm
                            0.0,                                                    # head_tilt
                            0.0,                                                    # temporal_x_var
                            0.0,                                                    # temporal_y_var
                            avgEAR,                                                 # eye_openness_ratio
                            0.9,                                                    # landmark_confidence
                            dist(pts, 4, 50) / faceH,                               # nose_wrinkle
                        ]
                        frame_indicators.append(indicators)
                frame_count += 1
                
            cap.release()
            if frame_indicators:
                avg_indicators = list(np.mean(frame_indicators, axis=0))
                rows.append(avg_indicators + [int(label_val)])
                print(f"  > Success: extracted {len(frame_indicators)} frames")
                
    if len(rows) == 0:
        print("Error: No facial features extracted from videos.")
        return False
        
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
            'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
            'lip_compression', 'jaw_tension', 'mouth_corner_pull',
            'forehead_tension', 'face_height_norm', 'head_tilt',
            'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
            'landmark_confidence', 'nose_wrinkle', 'label',
        ]
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved {len(rows)} samples to {output_csv}")
    return True

# -----------------------------------------------------------------------------
# 3. TRAINING MODELS
# -----------------------------------------------------------------------------

def train_face_expert():
    print("\n" + "="*50)
    print("STAGE 2: Training Lightweight Face Expert")
    print("="*50)
    
    if os.path.exists(FACE_STRESSID_CSV):
        df = pd.read_csv(FACE_STRESSID_CSV)
        print(f"Loaded StressID Face features dataset: {FACE_STRESSID_CSV}")
    elif os.path.exists(FACE_TRAIN_CSV) and os.path.exists(FACE_TEST_CSV):
        df_train = pd.read_csv(FACE_TRAIN_CSV)
        df_test  = pd.read_csv(FACE_TEST_CSV)
        df = pd.concat([df_train, df_test], ignore_index=True)
        print("Loaded facesData (train/test) dataset")
    else:
        print("Error: Extracted face CSV files missing. Run extraction first.")
        return

    FEATURES = [
        'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
        'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
        'lip_compression', 'jaw_tension', 'mouth_corner_pull',
        'forehead_tension', 'face_height_norm', 'head_tilt',
        'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
        'landmark_confidence', 'nose_wrinkle',
    ]

    X = df[FEATURES].values
    y = df['label'].values

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f'Train size: {len(X_train)}  |  Test size: {len(X_test)}')
    print(f'Class balance: {dict(zip(*np.unique(y_train, return_counts=True)))}')

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled  = scaler.transform(X_test)

    # Gradient Boosting Classifier
    print("Fitting Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train_res)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    print('\nTest Results:')
    print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train_res, cv=cv, scoring='f1')
    print(f'5-Fold CV F1-Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

    # Save expert pickle models back to Drive
    with open(FACE_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(FACE_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved: {FACE_MODEL_PATH}")
    print(f"Saved: {FACE_SCALER_PATH}")


def train_voice_expert():
    print("\n" + "="*50)
    print("STAGE 3: Training Lightweight Voice Expert")
    print("="*50)

    FEATURE_NAMES = [
        'f0_mean', 'f0_std', 'f0_range', 'jitter_percent', 'shimmer_db',
        'hnr', 'speaking_rate_proxy', 'voice_intensity', 'high_freq_ratio',
        'spectral_flux', 'pause_ratio', 'voiced_fraction', 'label',
    ]

    # Check for cached voice features CSV in Google Drive
    if os.path.exists(VOICE_STRESSID_CSV):
        print(f"Cache found for StressID Voice features: {VOICE_STRESSID_CSV}. Loading cached data...")
        df = pd.read_csv(VOICE_STRESSID_CSV)
    else:
        if not os.path.exists(STRESSID_AUDIO):
            print(f"Error: StressID audio path not found at: {STRESSID_AUDIO}")
            print("Please check your Drive dataset paths.")
            return

        # Load labels CSV if available for ground-truth matching
        labels_csv = os.path.join(STRESSID_ROOT, 'labels.csv')
        if not os.path.exists(labels_csv):
            labels_csv_fallback = os.path.join(STRESSID_ROOT, '._labels.csv')
            if os.path.exists(labels_csv_fallback):
                labels_csv = labels_csv_fallback
                
        use_csv_labels = False
        label_map = {}
        if os.path.exists(labels_csv):
            try:
                labels_df = pd.read_csv(labels_csv)
                for _, row in labels_df.iterrows():
                    st = row.get('subject/task')
                    lbl = row.get('binary-stress')
                    if st and lbl is not None and not pd.isna(lbl):
                        label_map[str(st).strip().lower()] = int(lbl)
                use_csv_labels = True
                print(f"Loaded {len(label_map)} labels from {labels_csv} for audio mapping.")
            except Exception as e:
                print(f"Warning: Could not read labels CSV for audio mapping: {e}")

        # StressID Labels fallback keywords definition
        STRESS_CONDITIONS = ['public_speaking', 'mental_math', 'stroop', 'math', 'speaking', 'stress']
        CALM_CONDITIONS   = ['rest', 'baseline', 'relax', 'calm', 'breathing', 'reading', 'video', 'nostress']

        rows = []
        print("Walking audio folders to extract 12 vocal biomarkers...")
        
        # Walk directory to locate all audio files recursively
        audio_files = []
        for root, dirs, files in os.walk(STRESSID_AUDIO):
            for fname in files:
                if fname.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    audio_files.append((root, fname))

        print(f"Found {len(audio_files)} audio samples. Commencing extraction...")
        
        for idx, (root, fname) in enumerate(audio_files):
            fpath = os.path.join(root, fname)
            base_name = os.path.splitext(fname)[0].strip().lower()
            
            # Deduce binary label (CSV lookup first, then fallback to filename checks)
            label = None
            if use_csv_labels and base_name in label_map:
                label = label_map[base_name]
            else:
                # Fallback to filename keywords (never use parent path/root folder name to avoid 'stress' in path conflicts)
                for sc in STRESS_CONDITIONS:
                    if sc in base_name:
                        label = 1
                        break
                if label is None:
                    for cc in CALM_CONDITIONS:
                        if cc in base_name:
                            label = 0
                            break

            try:
                with open(fpath, 'rb') as f:
                    audio_bytes = f.read()
                result = extract_voice_stress_indicators(audio_bytes)
                if result is not None:
                    rows.append(list(result['features']) + [label])
            except Exception as e:
                pass
                
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(audio_files)} audio files...")

        if len(rows) == 0:
            print("Error: No voice features extracted. Check your datasets.")
            return

        df = pd.DataFrame(rows, columns=FEATURE_NAMES)
        df.to_csv(VOICE_STRESSID_CSV, index=False)
        print(f"Saved {len(df)} extracted voice samples to cached CSV: {VOICE_STRESSID_CSV}")

    print(f'Using voice dataset with {len(df)} samples.')
    print(df['label'].value_counts())

    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # Scale Features
    scaler = StandardScaler()
    X_res_scaled  = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)

    # Fit Model
    print("Fitting Voice Gradient Boosting Expert...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_res_scaled, y_res)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print('\nTest Results:')
    print(classification_report(y_test, y_pred, target_names=['Calm', 'Stress']))

    # Save models directly back to Drive
    with open(VOICE_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VOICE_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved: {VOICE_MODEL_PATH}")
    print(f"Saved: {VOICE_SCALER_PATH}")


# -----------------------------------------------------------------------------
# 4. SCRIPT RUNNER
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print("="*60)
    print("GOOGLE COLAB / DRIVE MULTIMODAL STRESS EXPERTS TRAINING PIPELINE")
    print("="*60)
    
    start_time = time.time()
    
    # Check if face expert models already exist in Drive
    face_model_exists = os.path.exists(FACE_MODEL_PATH) and os.path.exists(FACE_SCALER_PATH)
    if face_model_exists:
        print("\n>>> Face expert model and scaler already exist in Google Drive. Skipping face extraction and training.")
    else:
        # 1. Run face landmark extraction and save CSV
        print("\nAttempting extraction from StressID video files...")
        face_extracted = run_face_extraction_stressid_videos()
        
        if not face_extracted:
            print("StressID Videos extraction skipped or failed. Falling back to facesData images...")
            face_extracted = run_face_extraction()
            
        # 2. Train face expert model
        if face_extracted:
            train_face_expert()
            
    # Check if voice expert models already exist in Drive
    voice_model_exists = os.path.exists(VOICE_MODEL_PATH) and os.path.exists(VOICE_SCALER_PATH)
    if voice_model_exists:
        print("\n>>> Voice expert model and scaler already exist in Google Drive. Skipping voice extraction and training.")
    else:
        # 3. Extract and train voice expert model
        train_voice_expert()
    
    total_m = (time.time() - start_time) / 60.0
    print("\n" + "="*50)
    print(f"Pipeline executed successfully in {total_m:.2f} minutes.")
    print(f"Trained models saved in Google Drive folder: {OUTPUT_DIR}")
    print("Please download this folder and copy all pkl files to the local 'backend/expert_models/' folder.")
    print("="*50)
