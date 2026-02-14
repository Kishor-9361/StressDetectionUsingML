import numpy as np
import os
import tempfile
import soundfile as sf
import base64
import cv2
from model import MultimodalStressDetector

class StressStreamProcessor:
    def __init__(self):
        self.model = MultimodalStressDetector()
        
        # Load the model if available
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        if self.model.load_model(BASE_DIR):
             print("Realtime Processor: Expert Models loaded successfully")
        else:
             print("Realtime Processor Error: Could not load Expert Models")
        
        # Session storage for audio buffering
        self.sessions = {}
        
        # Constants
        self.MAX_AUDIO_BUFFER_SECONDS = 3.0
        self.MIN_AUDIO_FOR_PREDICTION = 0.5 
        
    def initialize_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'audio': np.array([], dtype=np.float32),
                'sr': 44100 
            }
        
    def remove_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def process_video_frame(self, image_data_base64, sensitivity=0.5):
        try:
            if ',' in image_data_base64:
                image_data = image_data_base64.split(',')[1]
            else:
                image_data = image_data_base64
                
            image_bytes = base64.b64decode(image_data)
            
            fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            
            with open(temp_path, 'wb') as f:
                f.write(image_bytes)
            
            features, face_coords = self.model.extract_facial_features(temp_path)
            
            if features is None:
                 try: os.remove(temp_path) 
                 except: pass
                 return {'error': 'No face detected'}

            # Predict
            result = self.model.predict(facial_features=features, temp_image_path=temp_path, sensitivity=sensitivity)
            
            if face_coords:
                result['face_box'] = face_coords
            
            try: os.remove(temp_path)
            except: pass
                
            return result
            
        except Exception as e:
            print(f"Video processing error: {e}")
            return {'error': str(e)}

    def process_audio_chunk(self, session_id, audio_blob, sample_rate=44100, sensitivity=0.5):
        if session_id not in self.sessions:
            self.initialize_session(session_id)
            
        session = self.sessions[session_id]
        
        try:
            new_audio = np.array(audio_blob, dtype=np.float32)
            session['audio'] = np.concatenate((session['audio'], new_audio))
            
            max_samples = int(self.MAX_AUDIO_BUFFER_SECONDS * sample_rate)
            if len(session['audio']) > max_samples:
                session['audio'] = session['audio'][-max_samples:]
            
            min_samples = int(self.MIN_AUDIO_FOR_PREDICTION * sample_rate)
            if len(session['audio']) < min_samples:
                return {'status': 'buffering', 'message': f'Collecting audio...'}
            
            fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            sf.write(temp_path, session['audio'], sample_rate)
            features = self.model.extract_voice_features(temp_path)
            
            result = self.model.predict(voice_features=features, sensitivity=sensitivity)
            
            try: os.remove(temp_path)
            except: pass
                
            return result
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            return {'error': str(e)}
