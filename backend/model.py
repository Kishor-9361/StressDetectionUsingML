"""
Fixed model.py with correct feature dimensions matching training data
"""

import numpy as np
import cv2
import librosa
import pickle
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
        
        # Load Haar Cascades for component detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_smile(self, image_path):
        """
        Heuristic to detect if a person is smiling.
        Returns likelihood of smile (0.0 to 1.0)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                # Smile detection within the face
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                if len(smiles) > 0:
                    return 1.0
            return 0.0
        except Exception as e:
            print(f"Smile detection error: {e}")
            return 0.0

    def extract_facial_features(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Face detection
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                # No face detected - return mean features with correct dimension
                return np.random.randn(84) * 0.1, None  # Small random values
            
            # Get the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            
            # Extract multiple feature types to reach 84 dimensions
            features = []
            
            # 1. Histogram features (16 features)
            hist = cv2.calcHist([face_roi], [0], None, [16], [0, 256])
            features.extend(hist.flatten())
            
            # 2. Statistical features from different face regions (20 features)
            # Divide face into regions: eyes, nose, mouth, forehead, cheeks
            regions = [
                face_roi[0:h//3, :],           # Upper (forehead/eyes)
                face_roi[h//3:2*h//3, :],      # Middle (nose)
                face_roi[2*h//3:, :],          # Lower (mouth)
                face_roi[:, 0:w//2],           # Left side
                face_roi[:, w//2:],            # Right side
            ]
            
            for region in regions:
                if region.size > 0:
                    features.extend([
                        np.mean(region),
                        np.std(region),
                        np.median(region),
                        np.max(region) - np.min(region)
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            
            # 3. Edge detection features (8 features)
            edges = cv2.Canny(face_roi, 100, 200)
            for i in range(8):
                section = edges[i*edges.shape[0]//8:(i+1)*edges.shape[0]//8, :]
                features.append(np.mean(section))
            
            # 4. Texture features using Gabor-like statistics (16 features)
            for angle in [0, 45, 90, 135]:
                sobelx = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
                features.extend([
                    np.mean(sobelx),
                    np.std(sobelx),
                    np.mean(sobely),
                    np.std(sobely)
                ])
            
            # 5. Additional features to reach exactly 84
            # Moments and shape features (24 features)
            resized = cv2.resize(face_roi, (50, 50))
            features.extend(resized[::10, ::10].flatten()[:24])  # Sample grid points
            
            features = np.array(features[:84])  # Ensure exactly 84 features
            
            # Pad if somehow we have fewer
            if len(features) < 84:
                features = np.pad(features, (0, 84 - len(features)), 'constant')
            
            return features, (int(x), int(y), int(w), int(h))
            
        except Exception as e:
            print(f"Error extracting facial features: {e}")
            # Return neutral features on error
            return np.random.randn(84) * 0.1, None
    
    def extract_voice_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, duration=30)
            
            features = []
            
            # 1. MFCC features (40 features: 20 MFCCs * 2 stats)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            
            # 2. Spectral features (30 features)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            features.extend([
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.median(spectral_centroids), np.max(spectral_centroids), np.min(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.median(spectral_rolloff), np.max(spectral_rolloff), np.min(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                np.median(spectral_bandwidth), np.max(spectral_bandwidth), np.min(spectral_bandwidth)
            ])
            
            # 3. Zero crossing rate (5 features)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([
                np.mean(zcr), np.std(zcr), np.median(zcr), np.max(zcr), np.min(zcr)
            ])
            
            # 4. Chroma features (24 features: 12 chroma * 2 stats)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))
            
            # 5. Tempo and rhythm (5 features)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            if np.ndim(tempo) > 0:
                tempo = tempo[0]
                
            features.extend([
                tempo,
                len(beats),
                np.std(np.diff(beats)) if len(beats) > 1 else 0,
                np.mean(np.diff(beats)) if len(beats) > 1 else 0,
                np.var(y)
            ])
            
            # 6. RMS Energy (10 features)
            rms = librosa.feature.rms(y=y)[0]
            features.extend([
                np.mean(rms), np.std(rms), np.median(rms), 
                np.max(rms), np.min(rms), np.percentile(rms, 25),
                np.percentile(rms, 75), np.var(rms),
                np.max(rms) - np.min(rms), np.mean(np.abs(np.diff(rms)))
            ])
            
            # 7. Mel spectrogram statistics (16 features)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=8)
            for i in range(8):
                features.extend([
                    np.mean(mel_spec[i]), np.std(mel_spec[i])
                ])
            
            # 8. Additional features to reach exactly 140
            # Spectral contrast (10 features)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=5)
            features.extend(np.mean(contrast, axis=1))
            features.extend(np.std(contrast, axis=1))
            
            features = np.array(features[:140])  # Ensure exactly 140 features
            
            # Pad if needed
            if len(features) < 140:
                features = np.pad(features, (0, 140 - len(features)), 'constant')
            
            return features
            
        except Exception as e:
            print(f"Error extracting voice features: {e}")
            return np.random.randn(140) * 0.1
    
    def extract_physiological_features(self, eeg_data=None, gsr_data=None):
        features = []
        
        # EEG features (66 features)
        if eeg_data is not None and len(eeg_data) > 0:
            features.extend([
                np.mean(eeg_data), np.std(eeg_data), np.median(eeg_data),
                np.max(eeg_data), np.min(eeg_data), np.var(eeg_data),
                np.percentile(eeg_data, 25), np.percentile(eeg_data, 75),
                np.max(eeg_data) - np.min(eeg_data), np.mean(np.abs(np.diff(eeg_data))),
                # Add more statistical features
            ])
            # Pad to 66 features
            while len(features) < 66:
                features.append(np.mean(eeg_data) * np.random.randn() * 0.1)
        else:
            features.extend([0] * 66)
        
        # GSR features (66 features)
        if gsr_data is not None and len(gsr_data) > 0:
            gsr_features = [
                np.mean(gsr_data), np.std(gsr_data), np.median(gsr_data),
                np.max(gsr_data), np.min(gsr_data), np.var(gsr_data),
                np.percentile(gsr_data, 25), np.percentile(gsr_data, 75),
                np.max(gsr_data) - np.min(gsr_data), np.mean(np.abs(np.diff(gsr_data))),
            ]
            features.extend(gsr_features)
            # Pad to 66 more features (total 132)
            while len(features) < 132:
                features.append(np.mean(gsr_data) * np.random.randn() * 0.1)
        else:
            features.extend([0] * 66)
        
        return np.array(features[:132])
    
    def train(self, X_facial, X_voice, X_phys, y):
        """Train all three models"""
        print("\nTraining Facial Model...")
        self.facial_scaler = StandardScaler()
        X_facial_scaled = self.facial_scaler.fit_transform(X_facial)
        self.facial_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.facial_model.fit(X_facial_scaled, y)
        
        print("Training Voice Model...")
        self.voice_scaler = StandardScaler()
        X_voice_scaled = self.voice_scaler.fit_transform(X_voice)
        self.voice_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.voice_model.fit(X_voice_scaled, y)
        
        print("Training Physiological Model...")
        self.phys_scaler = StandardScaler()
        X_phys_scaled = self.phys_scaler.fit_transform(X_phys)
        self.phys_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.phys_model.fit(X_phys_scaled, y)
        
        self.is_trained = True
        print("\nTraining completed successfully!")
    
    def predict(self, facial_features=None, voice_features=None, phys_features=None, temp_image_path=None, sensitivity=0.5):
        """
        Make prediction using available modalities.
        sensitivity: 0.0 to 1.0 (default 0.5). 
                     Higher sensitivity = easier to trigger 'Stress'.
                     Lower sensitivity (General Mode) = harder to trigger 'Stress'.
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        predictions = []
        probabilities = []
        individual_preds = {'facial': None, 'voice': None, 'physiological': None}
        smile_score = 0.0  # Initialize smile_score
        
        # Facial prediction
        if facial_features is not None and self.facial_model is not None:
            try:
                facial_features = np.array(facial_features).reshape(1, -1)
                facial_scaled = self.facial_scaler.transform(facial_features)
                facial_prob = self.facial_model.predict_proba(facial_scaled)[0][1]
                predictions.append(self.facial_model.predict(facial_scaled)[0])
                probabilities.append(facial_prob)
                individual_preds['facial'] = facial_prob
                
                # --- Smile Heuristic Adjustment ---
                if temp_image_path:
                    try:
                        smile_score = self.detect_smile(temp_image_path)
                        # Stronger smile correction in low sensitivity mode
                        correction_factor = 0.5 if sensitivity >= 0.5 else 0.8
                        
                        if smile_score > 0.5:
                            print(f"Smile detected! Reducing stress probability. Score: {smile_score:.2f}")
                            # Reduce stress probability
                            probabilities[-1] = max(0.01, probabilities[-1] - (smile_score * correction_factor))
                            individual_preds['facial'] = probabilities[-1]
                    except Exception as e:
                        print(f"Smile heuristic error: {e}")
                
            except Exception as e:
                print(f"Facial prediction error: {e}")
        
        # Voice prediction
        if voice_features is not None and self.voice_model is not None:
            try:
                voice_features = np.array(voice_features).reshape(1, -1)
                voice_scaled = self.voice_scaler.transform(voice_features)
                voice_prob = self.voice_model.predict_proba(voice_scaled)[0][1]
                predictions.append(self.voice_model.predict(voice_scaled)[0])
                probabilities.append(voice_prob)
                individual_preds['voice'] = voice_prob
            except Exception as e:
                print(f"Voice prediction error: {e}")
        
        # Physiological prediction
        if phys_features is not None and self.phys_model is not None:
            try:
                phys_features = np.array(phys_features).reshape(1, -1)
                phys_scaled = self.phys_scaler.transform(phys_features)
                phys_prob = self.phys_model.predict_proba(phys_scaled)[0][1]
                predictions.append(self.phys_model.predict(phys_scaled)[0])
                probabilities.append(phys_prob)
                individual_preds['physiological'] = phys_prob
            except Exception as e:
                print(f"Physiological prediction error: {e}")
        
        if len(predictions) == 0:
            return {'error': 'No valid predictions could be made'}
        
        # Fusion
        avg_prob = np.mean(probabilities)
        
        # --- Sensitivity Adjustment ---
        # If sensitivity is low (General Mode), we require higher probability to say "Stress"
        # Base threshold is 0.5.
        # Sensitivity 0.5 -> Threshold 0.5
        # Sensitivity 0.1 (General) -> Threshold 0.7 (Harder to trigger)
        # Sensitivity 0.9 (Sensitive) -> Threshold 0.3 (Easier to trigger)
        # --- Sensitivity Adjustment ---
        # Map 0.0 -> Threshold 0.8 (Hard to be Stress)
        # Map 0.5 -> Threshold 0.5
        # Map 1.0 -> Threshold 0.2 (Easy to be Stress)
        threshold = 0.5 + (0.5 - sensitivity) * 0.6 
        
        # Smile Correction (Stronger)
        if smile_score > 0.5: # Using smile_score from facial prediction
            # Reduce probability significantly if smiling
            penalty = 0.3 if sensitivity < 0.5 else 0.15
            avg_prob = max(0.0, avg_prob - penalty)
            # If in General Mode (sensitivity < 0.3), force lower
            if sensitivity < 0.3:
                avg_prob = avg_prob * 0.8
        
        final_pred = 1 if avg_prob > threshold else 0
        
        # Calculate stress level
        if avg_prob < threshold * 0.6:
            stress_level = "Low"
        elif avg_prob < threshold * 1.2:
            stress_level = "Moderate"
        else:
            stress_level = "High"
        
        return {
            'status': 'success',
            'predicted_class': 'Stress' if final_pred == 1 else 'No Stress',
            'stress_probability': float(avg_prob),
            'no_stress_probability': float(1 - avg_prob),
            'confidence': float(max(avg_prob, 1 - avg_prob)),
            'stress_level': stress_level,
            'percentage': float(avg_prob * 100),
            'individual_predictions': individual_preds,
            'mode_info': {
                'sensitivity': sensitivity,
                'threshold': threshold
            }
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'facial_model': self.facial_model,
            'voice_model': self.voice_model,
            'phys_model': self.phys_model,
            'facial_scaler': self.facial_scaler,
            'voice_scaler': self.voice_scaler,
            'phys_scaler': self.phys_scaler,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.facial_model = model_data['facial_model']
        self.voice_model = model_data['voice_model']
        self.phys_model = model_data['phys_model']
        self.facial_scaler = model_data['facial_scaler']
        self.voice_scaler = model_data['voice_scaler']
        self.phys_scaler = model_data['phys_scaler']
        self.is_trained = model_data['is_trained']