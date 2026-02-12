
import numpy as np
import os
import tempfile
import soundfile as sf
import base64
from io import BytesIO
from PIL import Image
import cv2
from model import MultimodalStressDetector

class StressStreamProcessor:
    def __init__(self):
        self.model = MultimodalStressDetector()
        
        # Load the model if available
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, 'multimodal_stress_model.pkl')
        
        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_model(MODEL_PATH)
                print("Realtime Processor: Model loaded successfully")
            except Exception as e:
                print(f"Realtime Processor Error: Could not load model: {e}")
        
        # Session storage for audio buffering
        # { session_id: { 'audio': np.array([]), 'sr': 44100 } }
        self.sessions = {}
        
        # Constants
        self.MAX_AUDIO_BUFFER_SECONDS = 5.0
        self.MIN_AUDIO_FOR_PREDICTION = 1.0
        
    def initialize_session(self, session_id):
        self.sessions[session_id] = {
            'audio': np.array([]),
            'sr': 44100  # Default, updates on first chunk
        }
        
    def remove_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def process_video_frame(self, image_data_base64, sensitivity=0.5):
        """
        Process a single video frame.
        Expects base64 encoded string (data:image/jpeg;base64,...)
        """
        try:
            # Decode base64
            if ',' in image_data_base64:
                image_data = image_data_base64.split(',')[1]
            else:
                image_data = image_data_base64
                
            image_bytes = base64.b64decode(image_data)
            
            # Save to temp file (since model expects path)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
                temp_img.write(image_bytes)
                temp_path = temp_img.name
            
            # Extract features
            features, face_coords = self.model.extract_facial_features(temp_path)
            
            # Predict (Pass temp_path for smile detection)
            result = self.model.predict(facial_features=features, temp_image_path=temp_path, sensitivity=sensitivity)
            
            # Add face coordinates to result
            if face_coords:
                result['face_box'] = face_coords
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
                
            return result
            
        except Exception as e:
            print(f"Video processing error: {e}")
            return {'error': str(e)}

    def process_audio_chunk(self, session_id, audio_blob, sample_rate=44100, sensitivity=0.5):
        """
        Process an audio chunk. 
        Expects raw bytes (e.g., WebM/WAV from MediaRecorder) OR float32 array.
        """
        if session_id not in self.sessions:
            self.initialize_session(session_id)
            
        session = self.sessions[session_id]
        
        try:
            new_audio = None
            
            # Case 1: Raw Bytes (from MediaRecorder)
            if isinstance(audio_blob, bytes):
                # Save to temp file to decode
                with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_in:
                    temp_in.write(audio_blob)
                    temp_in_path = temp_in.name
                
                try:
                    # Load with librosa (handles resampling automatically)
                    # We load at 22050Hz or 44100Hz usually, but let's stick to default or model's need
                    # Model expects features extracted from file, so maybe we don't need to load raw yet?
                    # BUT we need to buffer it. So we load it.
                    y, sr = librosa.load(temp_in_path, sr=sample_rate)
                    new_audio = y
                    os.remove(temp_in_path)
                except Exception as e:
                    print(f"Error decoding audio bytes: {e}")
                    if os.path.exists(temp_in_path):
                        os.remove(temp_in_path)
                    return {'error': 'Audio decode failed'}
            
            # Case 2: List/Array (Legacy/ScriptProcessor)
            else:
                new_audio = np.array(audio_blob, dtype=np.float32)
            
            
            # --- DEBUG LOGGING ---
            if len(new_audio) > 0:
                amplitude = np.max(np.abs(new_audio))
                if amplitude > 0.01:
                    print(f"Audio received: {len(new_audio)} samples, Max Amp: {amplitude:.4f}")
            
            # Append to buffer
            session['audio'] = np.concatenate((session['audio'], new_audio))
            
            # Maintain sliding window (keep last N seconds)
            max_samples = int(self.MAX_AUDIO_BUFFER_SECONDS * sample_rate)
            if len(session['audio']) > max_samples:
                session['audio'] = session['audio'][-max_samples:]
            
            # Only predict if we have enough data
            min_samples = int(self.MIN_AUDIO_FOR_PREDICTION * sample_rate)
            if len(session['audio']) < min_samples:
                return {'status': 'buffering', 'message': f'Collecting audio samples... ({len(session["audio"])}/{min_samples})'}
            
            # Save buffer to temp wav file for feature extraction
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                sf.write(temp_audio.name, session['audio'], sample_rate)
                temp_path = temp_audio.name
                
            # Extract features
            features = self.model.extract_voice_features(temp_path)
            
            # Predict
            result = self.model.predict(voice_features=features, sensitivity=sensitivity)
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
                
            return result
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            return {'error': str(e)}
