"""
Fixed model.py with correct feature dimensions matching training data
"""

import numpy as np
import cv2
import librosa
import pickle
import os
import mediapipe as mp

# Check if we can use legacy solutions or modern Tasks API
USE_LEGACY_MEDIAPIPE = False
try:
    # pyrefly: ignore [missing-import]
    import mediapipe.solutions.face_mesh as mp_face_mesh
    USE_LEGACY_MEDIAPIPE = True
except (ImportError, AttributeError):
    USE_LEGACY_MEDIAPIPE = False

# --- FaceMeshWrapper to support both Legacy solutions and Modern Tasks API ---
class FaceMeshWrapper:
    def __init__(self, static_mode=True):
        self.static_mode = static_mode
        self.use_tasks = not USE_LEGACY_MEDIAPIPE
        self.fm = None
        self.detector = None
        
        if self.use_tasks:
            self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
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
            # pyrefly: ignore [missing-import]
            import mediapipe.solutions.face_mesh as mp_face_mesh
            self.fm = mp_face_mesh.FaceMesh(
                static_image_mode=self.static_mode,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
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


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class MultimodalStressDetector:
    def __init__(self):
        self.facial_model = None
        self.voice_model = None
        self.phys_model = None
        
        self.facial_scaler = None
        self.voice_scaler = None
        self.phys_scaler = None
        
        self.is_trained = False
        
        # Load Haar Cascades (Legacy Backup)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        # Advanced MediaPipe Initialization
        self.face_mesh = FaceMeshWrapper(static_mode=False)
    
    def load_model(self, base_path='.'):
        """Load the 3 specialized expert models"""
        try:
            # 1. Facial Expert (Trained on facesData - Clean Images)
            face_path = os.path.join(base_path, 'facial_expert_model.pkl')
            if os.path.exists(face_path):
                with open(face_path, 'rb') as f:
                    self.facial_model = pickle.load(f)
                print("Loaded Facial Expert Model")
            
            # 2. Voice Expert (Trained on StressID - Real Audio)
            voice_path = os.path.join(base_path, 'voice_expert_model.pkl')
            if os.path.exists(voice_path):
                with open(voice_path, 'rb') as f:
                    data = pickle.load(f)
                    self.voice_model = data['model']
                    self.voice_scaler = data['scaler']
                print("Loaded Voice Expert Model")

            # 3. Physio Expert (Trained on StressID - Real Biosignals)
            phys_path = os.path.join(base_path, 'physio_expert_model.pkl')
            if os.path.exists(phys_path):
                with open(phys_path, 'rb') as f:
                    data = pickle.load(f)
                    self.phys_model = data['model']
                    self.phys_scaler = data['scaler']
                print("Loaded Physio Expert Model")
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def detect_smile(self, image_path):
        """Heuristic to detect if a person is smiling."""
        try:
            img = cv2.imread(image_path)
            if img is None: return 0.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                if len(smiles) > 0: return 1.0
            return 0.0
        except: return 0.0

    def extract_facial_features(self, image_path):
        """Advanced MediaPipe-based Feature Extraction"""
        try:
            img = cv2.imread(image_path)
            if img is None: return None, None
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img, _ = img.shape
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_img)
            
            if not results.multi_face_landmarks:
                # Fallback to Haar for bounding box if MediaPipe fails to find mesh
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    return np.zeros(84), None
                x, y, w, h = faces[0]
                return np.zeros(84), (int(x), int(y), int(w), int(h))

            # 1. High Level Geometric Features
            landmarks = results.multi_face_landmarks[0].landmark
            
            def get_dist(p1_idx, p2_idx):
                p1 = landmarks[p1_idx]
                p2 = landmarks[p2_idx]
                return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

            # --- Derived Stress Indicators ---
            # Eye Aspect Ratio (EAR) - Dilation/Squinting
            ear_l = (get_dist(160, 144) + get_dist(158, 153)) / (2 * get_dist(33, 133) + 1e-6)
            ear_r = (get_dist(385, 380) + get_dist(387, 373)) / (2 * get_dist(362, 263) + 1e-6)
            avg_ear = (ear_l + ear_r) / 2
            
            # Brow Tension (Distance between inner brows and eye)
            brow_dist = get_dist(9, 168) # Bridge of nose to brow center
            brow_lowering = (get_dist(52, 159) + get_dist(282, 386)) / 2 
            
            # Mouth Tension (Lip compression/Lip width)
            mouth_width = get_dist(61, 291)
            lip_height = get_dist(13, 14)
            mar = lip_height / mouth_width if mouth_width > 0 else 0
            
            # Head Pose (Tilt/Rotation proxy)
            face_width = get_dist(234, 454)
            
            geom_features = [
                avg_ear, brow_dist, brow_lowering, mouth_width, lip_height, mar, face_width,
                landmarks[1].z, # Nose depth (tells us if they lean in)
                get_dist(10, 152), # Face length
                get_dist(10, 1),   # Upper face
                get_dist(1, 152)   # Lower face
            ]
            
            # 2. Add some robust histogram data from the face center
            # Find bounding box from landmarks
            x_coords = [p.x for p in landmarks]
            y_coords = [p.y for p in landmarks]
            x, y = int(min(x_coords) * w_img), int(min(y_coords) * h_img)
            w, h = int((max(x_coords) - min(x_coords)) * w_img), int((max(y_coords) - min(y_coords)) * h_img)
            
            face_roi = cv2.cvtColor(img[max(0,y):y+h, max(0,x):x+w], cv2.COLOR_BGR2GRAY)
            if face_roi.size > 0:
                face_roi = cv2.resize(face_roi, (50, 50))
                hist = cv2.calcHist([face_roi], [0], None, [16], [0, 256]).flatten() / face_roi.size
                geom_features.extend(hist)
            else:
                geom_features.extend([0]*16)

            # Pad to 84 features for model compatibility
            features = np.array(geom_features)
            if len(features) < 84:
                features = np.pad(features, (0, 84 - len(features)), 'constant')
            
            return features[:84], (x, y, w, h)
            
        except Exception as e:
            print(f"Error extracting facial features: {e}")
            return np.zeros(84), None

    def extract_voice_features(self, audio_path):
        """Extract 140 audio features from a WAV file using librosa."""
        try:
            # Fix: duration=None to read full temp file (which is specifically cut to buffer length)
            y, sr = librosa.load(audio_path, duration=None)
            
            # Simple Silence Removal / Noise Gate
            if len(y) == 0 or np.max(np.abs(y)) < 0.005: # Silence threshold
                return np.zeros(140)

            features = []
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features.extend([
                np.mean(spectral_centroids), np.std(spectral_centroids), np.median(spectral_centroids), np.max(spectral_centroids), np.min(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff), np.median(spectral_rolloff), np.max(spectral_rolloff), np.min(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth), np.median(spectral_bandwidth), np.max(spectral_bandwidth), np.min(spectral_bandwidth)
            ])
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([np.mean(zcr), np.std(zcr), np.median(zcr), np.max(zcr), np.min(zcr)])
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            if np.ndim(tempo) > 0: tempo = tempo[0]
            features.extend([tempo, len(beats), np.std(np.diff(beats)) if len(beats) > 1 else 0, np.mean(np.diff(beats)) if len(beats) > 1 else 0, np.var(y)])
            rms = librosa.feature.rms(y=y)[0]
            features.extend([np.mean(rms), np.std(rms), np.median(rms), np.max(rms), np.min(rms), np.percentile(rms, 25), np.percentile(rms, 75), np.var(rms), np.max(rms) - np.min(rms), np.mean(np.abs(np.diff(rms)))])
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=8)
            for i in range(8): features.extend([np.mean(mel_spec[i]), np.std(mel_spec[i])])
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=5)
            features.extend(np.mean(contrast, axis=1))
            features.extend(np.std(contrast, axis=1))
            features = np.array(features[:140])
            if len(features) < 140: features = np.pad(features, (0, 140 - len(features)), 'constant')
            return features
        except Exception as e:
            print(f"Voice feature extraction error: {e}")
            return np.zeros(140)

    def extract_physiological_features(self, eeg_data=None, gsr_data=None):
        features = []
        if eeg_data is not None and len(eeg_data) > 0:
            features.extend([np.mean(eeg_data), np.std(eeg_data), np.median(eeg_data), np.max(eeg_data), np.min(eeg_data), np.var(eeg_data), np.percentile(eeg_data, 25), np.percentile(eeg_data, 75), np.max(eeg_data) - np.min(eeg_data), np.mean(np.abs(np.diff(eeg_data)))])
            while len(features) < 66: features.append(0)
        else: features.extend([0] * 66)
        
        if gsr_data is not None and len(gsr_data) > 0:
            gsr_features = [np.mean(gsr_data), np.std(gsr_data), np.median(gsr_data), np.max(gsr_data), np.min(gsr_data), np.var(gsr_data), np.percentile(gsr_data, 25), np.percentile(gsr_data, 75), np.max(gsr_data) - np.min(gsr_data), np.mean(np.abs(np.diff(gsr_data)))]
            features.extend(gsr_features)
            while len(features) < 132: features.append(0)
        else: 
            features.extend([0] * 66)
        return np.array(features[:132])

    def predict(self, facial_features=None, voice_features=None, phys_features=None, temp_image_path=None, sensitivity=0.5):
        if not self.is_trained: return {'error': 'Models not loaded'}
        
        probs = []
        preds = {'facial': None, 'voice': None, 'physiological': None}
        
        # 1. Facial Expert
        if facial_features is not None and self.facial_model:
            try:
                ff = np.array(facial_features).reshape(1, -1)
                f_prob = self.facial_model.predict_proba(ff)[0][1]
                preds['facial'] = f_prob
                
                if temp_image_path:
                    smile = self.detect_smile(temp_image_path)
                    if smile > 0.5:
                        f_prob = max(0.0, f_prob - 0.4) 
                
                probs.append(f_prob)
            except Exception as e: print(f"Facial pred error: {e}")
            
        # 2. Voice Expert
        if voice_features is not None and self.voice_model:
            try:
                vf = np.array(voice_features).reshape(1, -1)
                vf_scaled = self.voice_scaler.transform(vf)
                v_prob = self.voice_model.predict_proba(vf_scaled)[0][1]
                preds['voice'] = v_prob
                probs.append(v_prob)
            except Exception as e: print(f"Voice pred error: {e}")

        # 3. Physio Expert
        if phys_features is not None and self.phys_model:
            try:
                pf = np.array(phys_features).reshape(1, -1)
                pf_scaled = self.phys_scaler.transform(pf)
                p_prob = self.phys_model.predict_proba(pf_scaled)[0][1]
                preds['physiological'] = p_prob
                probs.append(p_prob)
            except Exception as e: print(f"Physio pred error: {e}")

        if not probs: return {'error': 'No valid predictions'}

        # Simple Average Fusion
        avg_prob = np.mean(probs)
        
        # Sensitivity Thresholding
        threshold = 0.6 + (0.5 - sensitivity) * 0.4
        final_pred = 1 if avg_prob > threshold else 0
        
        stress_level = "High" if avg_prob > 0.7 else "Moderate" if avg_prob > 0.4 else "Low"

        return {
            'status': 'success',
            'predicted_class': 'Stress' if final_pred else 'No Stress',
            'stress_probability': float(avg_prob),
            'no_stress_probability': float(1 - avg_prob),
            'confidence': float(max(avg_prob, 1 - avg_prob)),
            'stress_level': stress_level,
            'percentage': float(avg_prob * 100),
            'individual_predictions': preds
        }


# --- Module-Level Physiological Feature Extractors and Fusion ---

def extract_eeg_features(eeg_data, fs=256):
    import numpy as np
    from scipy import signal
    
    if eeg_data is None or len(eeg_data) == 0:
        return np.zeros(42)
        
    eeg_arr = np.array(eeg_data)
    
    # Compute PSD using Welch's method
    nperseg = min(len(eeg_arr), 256)
    if nperseg < 8:
        f, psd = np.array([0]), np.array([0])
    else:
        f, psd = signal.welch(eeg_arr, fs, nperseg=nperseg)
        
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    feats = np.zeros(42)
    band_means = {}
    for i, (band_name, (low, high)) in enumerate(bands.items()):
        idx = np.where((f >= low) & (f <= high))[0]
        if len(idx) > 0:
            bp_mean = float(np.mean(psd[idx]))
            bp_std = float(np.std(psd[idx]))
        else:
            bp_mean = 0.0
            bp_std = 0.0
            
        band_means[band_name] = bp_mean
        feats[2 * i] = bp_mean
        feats[2 * i + 1] = bp_std
        
    # Stress index: feats[11] = (beta + gamma) / (alpha + theta)
    denom = band_means['alpha'] + band_means['theta']
    if denom > 1e-10:
        feats[11] = (band_means['beta'] + band_means['gamma']) / denom
    else:
        feats[11] = 0.0
        
    return feats

def extract_gsr_features(gsr_data, fs=4):
    import numpy as np
    if gsr_data is None or len(gsr_data) == 0:
        return np.zeros(9)
        
    gsr_arr = np.array(gsr_data)
    
    # Smooth gsr to avoid noise peaks
    if len(gsr_arr) > 5:
        gsr_smooth = np.convolve(gsr_arr, np.ones(5)/5, mode='valid')
    else:
        gsr_smooth = gsr_arr
        
    diffs = np.diff(gsr_smooth)
    peaks = 0
    in_peak = False
    for val in diffs:
        if val > 0.05:
            if not in_peak:
                peaks += 1
                in_peak = True
        else:
            in_peak = False
            
    feats = np.zeros(9)
    feats[0] = float(np.mean(gsr_arr))
    feats[1] = float(np.std(gsr_arr))
    feats[2] = float(np.max(gsr_arr) - np.min(gsr_arr))
    feats[3] = float(peaks)
    
    return feats

def extract_physiological_features(eeg_data=None, gsr_data=None, pad_to_132=True):
    import numpy as np

    if pad_to_132:
        features = []
        if eeg_data is not None and len(eeg_data) > 0:
            eeg_arr = np.array(eeg_data)
            features.extend([
                np.mean(eeg_arr), np.std(eeg_arr), np.median(eeg_arr), np.max(eeg_arr), np.min(eeg_arr),
                np.var(eeg_arr), np.percentile(eeg_arr, 25), np.percentile(eeg_arr, 75),
                np.max(eeg_arr) - np.min(eeg_arr), np.mean(np.abs(np.diff(eeg_arr)))
            ])
            while len(features) < 66:
                features.append(0.0)
        else:
            features.extend([0.0] * 66)

        if gsr_data is not None and len(gsr_data) > 0:
            gsr_arr = np.array(gsr_data)
            gsr_features = [
                np.mean(gsr_arr), np.std(gsr_arr), np.median(gsr_arr), np.max(gsr_arr), np.min(gsr_arr),
                np.var(gsr_arr), np.percentile(gsr_arr, 25), np.percentile(gsr_arr, 75),
                np.max(gsr_arr) - np.min(gsr_arr), np.mean(np.abs(np.diff(gsr_arr)))
            ]
            features.extend(gsr_features)
            while len(features) < 132:
                features.append(0.0)
        else:
            features.extend([0.0] * 66)

        return np.array(features[:132])
    else:
        # Standard 42 EEG + 9 GSR features (total 51)
        eeg_feats = extract_eeg_features(eeg_data)
        gsr_feats = extract_gsr_features(gsr_data)
        return np.concatenate([eeg_feats, gsr_feats])

def fuse_predictions(probs, confs, fusion_mode='reliability'):
    active_modes = list(probs.keys())
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low', 'weights': {}, 'modality_weights': {}}
        
    base_weights = {'face': 0.371, 'voice': 0.474, 'physio': 0.338}
    
    active_modes = [m for m in active_modes if m in base_weights]
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low', 'weights': {}, 'modality_weights': {}}
        
    raw_weights = {}
    for m in active_modes:
        raw_weights[m] = base_weights[m] * confs.get(m, 1.0)
        
    w_sum = sum(raw_weights.values())
    if w_sum > 0:
        norm_weights = {m: raw_weights[m] / w_sum for m in active_modes}
    else:
        norm_weights = {m: 1.0 / len(active_modes) for m in active_modes}
        
    rounded_weights = {m: round(w, 3) for m, w in norm_weights.items()}
    fused_score = sum(probs[m] * norm_weights[m] for m in active_modes)
    level = "High" if fused_score > 0.7 else "Moderate" if fused_score > 0.4 else "Low"
    
    return {
        'fused_score': fused_score,
        'stress_level': level,
        'weights': rounded_weights,
        'modality_weights': rounded_weights
    }

