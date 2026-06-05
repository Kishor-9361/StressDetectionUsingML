import multiprocessing

# Only monkey patch standard libraries in the main process to prevent deadlocks in subprocesses
if multiprocessing.current_process().name == 'MainProcess':
    import eventlet
    eventlet.monkey_patch()

import sys
import builtins

# Force all print statements to flush immediately to avoid buffering in standard terminals/IDE logs
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    builtins.print(*args, **kwargs)

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from realtime_core import StressStreamProcessor
import os
import numpy as np
import threading
from collections import deque
from werkzeug.utils import secure_filename
import tempfile
import cv2
import librosa
from model import MultimodalStressDetector

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25, max_http_buffer_size=100000000)

# Initialize global stream processor
stream_processor = StressStreamProcessor()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the model
model = MultimodalStressDetector()

# Try to load pre-trained expert models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The load_model method now expects a directory containing the 3 expert .pkl files
if model.load_model(BASE_DIR):
    print("Pre-trained expert models loaded successfully!")
else:
    print("Could not load one or more expert models.")
    print("Please ensure 'facial_expert_model.pkl', 'voice_expert_model.pkl', and 'physio_expert_model.pkl' exist in the backend folder.")

# --- Load Lightweight Expert Models ---
import pickle
import time
import json
from voice_worker import extract_voice_stress_indicators
from score_buffer import score_buffer

def load_expert(filename):
    # Try backend/expert_models/ first, then backend/
    path1 = os.path.join(BASE_DIR, 'expert_models', filename)
    path2 = os.path.join(BASE_DIR, filename)
    path = path1 if os.path.exists(path1) else path2
    if not os.path.exists(path):
        print(f"Warning: {filename} not found at {path1} or {path2}")
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

face_expert = load_expert('face_expert_lightweight.pkl')
face_scaler = load_expert('face_scaler_lightweight.pkl')
voice_expert = load_expert('voice_expert_lightweight.pkl')
voice_scaler = load_expert('voice_scaler_lightweight.pkl')

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models_loaded': {
            'face_expert': face_expert is not None,
            'voice_expert': voice_expert is not None,
            'physio_expert': (model.phys_model is not None if hasattr(model, 'phys_model') else False)
        },
        'server': 'eventlet'
    })

@app.route('/api/multimodal/analyze', methods=['POST'])
def analyze_multimodal():
    """
    Multimodal stress analysis endpoint
    Accepts: image file, audio file, EEG data, GSR data
    """
    print("[HTTP] POST /api/multimodal/analyze - Starting multimodal analysis...")
    try:
        # Initialize feature holders
        facial_features = None
        voice_features = None
        phys_features = None
        
        # Process facial image if provided
        if 'face_image' in request.files:
            face_file = request.files['face_image']
            if face_file and allowed_file(face_file.filename, ALLOWED_IMAGE_EXTENSIONS):
                filename = secure_filename(face_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                face_file.save(filepath)
                
                # Extract facial features
                facial_features, _ = model.extract_facial_features(filepath)
                
                # Clean up
                os.remove(filepath)
        
        # Process voice audio if provided
        if 'voice_audio' in request.files:
            audio_file = request.files['voice_audio']
            if audio_file and allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
                filename = secure_filename(audio_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(filepath)
                
                # Extract voice features
                voice_features = model.extract_voice_features(filepath)
                
                # Clean up
                os.remove(filepath)
        
        # Process physiological data if provided
        eeg_data = request.form.get('eeg_data')
        gsr_data = request.form.get('gsr_data')
        
        if eeg_data or gsr_data:
            eeg_array = np.fromstring(eeg_data, sep=',') if eeg_data else None
            gsr_array = np.fromstring(gsr_data, sep=',') if gsr_data else None
            phys_features = model.extract_physiological_features(eeg_array, gsr_array)
        
        # Check if at least one modality is provided
        if facial_features is None and voice_features is None and phys_features is None:
            return jsonify({
                'status': 'error',
                'message': 'Please provide at least one input (image, audio, or physiological data)'
            }), 400
        
        # Make prediction
        result = model.predict(
            facial_features=facial_features,
            voice_features=voice_features,
            phys_features=phys_features
        )
        
        if 'error' in result:
            print(f"[HTTP] POST /api/multimodal/analyze - Failed: {result['error']}")
            return jsonify({
                'status': 'error',
                'message': result['error']
            }), 400
        
        print(f"[HTTP] POST /api/multimodal/analyze - Success: stress_level={result.get('stress_level')}, percentage={result.get('percentage')}%")
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/face/upload', methods=['POST'])
def analyze_face():
    """Facial stress analysis endpoint"""
    print("[HTTP] POST /api/face/upload - Starting facial analysis...")
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Please upload an image (PNG, JPG, JPEG)'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features and predict
        facial_features, _ = model.extract_facial_features(filepath)
        result = model.predict(facial_features=facial_features)
        
        # Clean up
        os.remove(filepath)
        
        print(f"[HTTP] POST /api/face/upload - Success: stress_level={result.get('stress_level')}, percentage={result.get('percentage')}%")
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/voice/upload', methods=['POST'])
def analyze_voice():
    """Voice stress analysis endpoint"""
    print("[HTTP] POST /api/voice/upload - Starting voice analysis...")
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Please upload an audio file (WAV, MP3)'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features and predict
        voice_features = model.extract_voice_features(filepath)
        result = model.predict(voice_features=voice_features)
        
        # Clean up
        os.remove(filepath)
        
        print(f"[HTTP] POST /api/voice/upload - Success: stress_level={result.get('stress_level')}, percentage={result.get('percentage')}%")
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/voice/record', methods=['POST'])
def record_voice():
    """Voice recording endpoint (simulated)"""
    # This is a placeholder - actual recording would be done client-side
    return jsonify({
        'status': 'error',
        'message': 'Please use the upload feature instead of recording'
    }), 501

@app.route('/api/webcam/capture', methods=['POST'])
def capture_webcam():
    """Webcam capture endpoint"""
    try:
        # Get base64 image data from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No image data provided'
            }), 400
        
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Save temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_webcam.jpg')
        image.save(temp_path)
        
        # Extract features and predict
        facial_features, _ = model.extract_facial_features(temp_path)
        result = model.predict(facial_features=facial_features)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# -----------------------------------------------------------------------------
# PHASE 1 REAL-TIME LIGHTWEIGHT MULTIMODAL STRESS DETECTION ENDPOINTS (SSE)
# -----------------------------------------------------------------------------

def fuse_predictions(probs, confs, fusion_mode='reliability'):
    """
    Fuse predictions from active modalities.
    If only one modality is active, returns that score.
    If multiple, fuses them using the reliability weight algorithm.
    """
    active_modes = list(probs.keys())
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low'}
        
    if len(active_modes) == 1:
        score = probs[active_modes[0]]
        level = "High" if score > 0.7 else "Moderate" if score > 0.4 else "Low"
        return {'fused_score': score, 'stress_level': level}
        
    # Reliability weights: base confidence on MediaPipe landmark confidence or fixed defaults
    # face: 0.5, voice: 0.5
    base_weights = {'face': 0.5, 'voice': 0.5}
    
    # Filter active modalities to only include face and voice
    active_modes = [m for m in active_modes if m in base_weights]
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low'}
        
    # Normalize weights for active modalities
    w_sum = sum(base_weights[m] for m in active_modes)
    norm_weights = {m: base_weights[m] / w_sum for m in active_modes}
    
    # Adjust weights based on confidence values if present
    if 'face' in active_modes and confs.get('face', 1.0) < 0.5:
        norm_weights['face'] *= max(confs['face'], 0.1)
        w_sum_new = sum(norm_weights.values())
        norm_weights = {m: w / w_sum_new for m, w in norm_weights.items()}
        
    fused_score = sum(probs[m] * norm_weights[m] for m in active_modes)
    level = "High" if fused_score > 0.7 else "Moderate" if fused_score > 0.4 else "Low"
    return {
        'fused_score': fused_score,
        'stress_level': level,
        'weights': norm_weights
    }

# Rolling histories and locks for multi-user face score smoothing
_face_histories = {}
_face_hist_lock = threading.Lock()

def get_face_history(user_id):
    with _face_hist_lock:
        if user_id not in _face_histories:
            _face_histories[user_id] = deque(maxlen=4)
        return _face_histories[user_id]

@app.route('/api/stream/face', methods=['POST'])
def stream_face():
    """
    Receive 18 browser-extracted facial indicators and update ScoreBuffer.
    """
    if face_expert is None or face_scaler is None:
        return jsonify({'error': 'Face expert model not loaded'}), 500
        
    data = request.json or {}
    indicators = data.get('indicators', {})
    user_id = data.get('user_id', 'default')

    if not indicators:
        return jsonify({'score': None}), 200

    try:
        from calibration import get_or_create
        cal = get_or_create(user_id)
        
        # Apply personal baseline normalization if complete
        if cal.is_complete:
            indicators = cal.normalize_face_indicators(indicators)
            cal.add_face_sample(indicators)  # continue updating baseline slowly

        # Extract features
        raw_left_ear = indicators.get('left_ear', 0.3)
        raw_right_ear = indicators.get('right_ear', 0.3)
        raw_avg_ear = indicators.get('avg_ear', 0.3)
        raw_brow_l = indicators.get('brow_descent_left', 0.1)
        raw_brow_r = indicators.get('brow_descent_right', 0.1)
        raw_jaw = indicators.get('jaw_displacement', 1.6)
        
        left_ear = raw_left_ear
        right_ear = raw_right_ear
        avg_ear = raw_avg_ear
        brow_l = raw_brow_l
        brow_r = raw_brow_r
        # A baseline linear mapping for jaw displacement -> jaw tension proxy:
        # 0.68 * (jaw_displacement / 1.6)
        jaw_tension = 0.68 * (raw_jaw / 1.6) if raw_jaw > 0 else 0.68
        eye_openness = indicators.get('eye_openness_ratio', avg_ear)
        
        if cal.is_complete:
            means = face_scaler.mean_
            scales = face_scaler.scale_
            
            # Map normalized Z-scores back using scaler parameters
            # so that face_scaler.transform() returns exactly the user Z-scores
            z_ear = indicators.get('avg_ear_normalized', 0.0)
            left_ear = means[0] + z_ear * scales[0]
            right_ear = means[1] + z_ear * scales[1]
            avg_ear = means[2] + z_ear * scales[2]
            eye_openness = means[15] + z_ear * scales[15]
            
            z_brow_l = indicators.get('brow_descent_left_normalized', 0.0)
            z_brow_r = indicators.get('brow_descent_right_normalized', 0.0)
            brow_l = means[4] + z_brow_l * scales[4]
            brow_r = means[5] + z_brow_r * scales[5]
            
            z_jaw = indicators.get('jaw_displacement_normalized', 0.0)
            jaw_tension = means[8] + z_jaw * scales[8]

        feature_vec = np.array([
            left_ear,
            right_ear,
            avg_ear,
            indicators.get('blink_velocity', 0.0),
            brow_l,
            brow_r,
            indicators.get('brow_asymmetry', 0.0),
            indicators.get('lip_compression', 0.2),
            jaw_tension,
            indicators.get('mouth_corner_pull', 0.3),
            indicators.get('forehead_tension', 0.1),
            indicators.get('face_height_norm', 1.5),
            indicators.get('head_tilt', 0.0),
            indicators.get('temporal_x_var', 0.0),
            indicators.get('temporal_y_var', 0.0),
            eye_openness,
            indicators.get('landmark_confidence', 0.9),
            indicators.get('nose_wrinkle', 0.1),
        ], dtype=np.float32).reshape(1, -1)

        # Scale features
        feature_scaled = face_scaler.transform(feature_vec)
        raw_score = float(face_expert.predict_proba(feature_scaled)[0][1])

        # Smile/laughter dampening
        smile_score = float(indicators.get('smile_score', 0.0))
        if smile_score > 0.3:
            dampening = smile_score * 0.4
            raw_score = max(0.0, raw_score - dampening)

        # Temporal smoothing: 4-sample rolling median per user
        hist = get_face_history(user_id)
        hist.append(raw_score)
        smoothed_score = float(np.median(list(hist)))

        # Write to buffer
        score_buffer.write('face', smoothed_score, indicators)
        print(f"[Face Expert] Processed frame. Score: {smoothed_score:.3f} (raw: {raw_score:.3f}, smile: {smile_score:.2f})")
        return jsonify({'score': smoothed_score, 'raw_score': raw_score, 'smile_detected': smile_score > 0.3})
    except Exception as e:
        print(f"Face streaming error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/voice', methods=['POST'])
def stream_voice():
    """
    Receive 2-second audio blob (WAV), extract 12 vocal biomarkers in eventlet's OS thread pool,
    and update ScoreBuffer.
    """
    if voice_expert is None or voice_scaler is None:
        return jsonify({'error': 'Voice expert model not loaded'}), 500
        
    audio_bytes = request.data
    if not audio_bytes or len(audio_bytes) < 1000:
        return jsonify({'score': None, 'reason': 'too_short_or_empty'}), 200

    user_id = request.args.get('user_id', 'default')

    # Check silence first — fast, no librosa needed (using numpy from audio_bytes)
    try:
        import io
        import soundfile as sf
        y, _ = sf.read(io.BytesIO(audio_bytes))
        rms = float(np.sqrt(np.mean(y ** 2)))
        
        from calibration import get_or_create
        cal = get_or_create(user_id)
        
        # Adjust silence threshold based on noise floor if calibrated
        threshold = 0.015
        if cal.is_complete and cal.noise_floor is not None:
            threshold = max(0.015, cal.noise_floor * 1.5)
            
        if rms < threshold:
            score_buffer.clear('voice')
            print(f"[Voice Expert] Silence/Ambient noise detected (Intensity: {rms:.4f} < threshold: {threshold:.4f}). Clearing voice buffer.")
            mock_indicators = {
                'f0_mean': 0.0,
                'jitter_percent': 0.0,
                'shimmer_db': 0.0,
                'speaking_rate_proxy': 0.0,
                'voice_intensity': rms
            }
            return jsonify({'score': None, 'reason': 'silence_detected', 'indicators': mock_indicators})
    except Exception as e:
        print(f"Error checking silence: {e}")
        pass

    try:
        # Run CPU-heavy librosa processing in eventlet's built-in OS thread pool
        # This keeps the main greenlet loop completely free without spawning subprocesses on Windows
        import eventlet.tpool
        result = eventlet.tpool.execute(extract_voice_stress_indicators, audio_bytes)
        
        if result is None:
            return jsonify({'score': None, 'reason': 'audio_too_short_or_invalid'}), 200

        # Extract features and indicators
        indicators = result['indicators']
        
        # Filter out silence or non-speech hum/noise post-feature-extraction
        if indicators.get('f0_mean', 0.0) == 0.0 or indicators.get('voiced_fraction', 0.0) < 0.05:
            score_buffer.clear('voice')
            print(f"[Voice Expert] Silence/Unvoiced audio detected in features (F0: {indicators.get('f0_mean', 0.0):.1f} Hz, Voiced Frac: {indicators.get('voiced_fraction', 0.0):.3f}). Clearing voice buffer.")
            return jsonify({'score': None, 'reason': 'unvoiced_or_silence', 'indicators': indicators})

        features = result['features']
        
        from calibration import get_or_create
        cal = get_or_create(user_id)
        if cal.is_complete:
            features = cal.normalize_voice_features(features, voice_scaler)
            cal.add_voice_sample(indicators)  # continue updating baseline slowly

        features = features.reshape(1, -1)
        features_scaled = voice_scaler.transform(features)
        score = float(voice_expert.predict_proba(features_scaled)[0][1])

        score_buffer.write('voice', score, {
            'landmark_confidence': 0.9  # high confidence for voice if extracted successfully
        })
        print(f"[Voice Expert] Processed audio chunk in OS thread pool. Score: {score:.3f}")
        return jsonify({'score': score, 'indicators': indicators})
    except Exception as e:
        print(f"Voice streaming error: {e}")
        return jsonify({'error': str(e)}), 500

# -----------------------------------------------------------------------------
# USER CALIBRATION ENDPOINTS
# -----------------------------------------------------------------------------

@app.route('/api/calibrate/silence', methods=['POST'])
def calibrate_silence():
    """Phase 1: Record ambient noise floor from 15 seconds of silence."""
    data        = request.json or {}
    user_id     = data.get('user_id', 'default')
    noise_rms   = float(data.get('noise_rms', 0.01))

    from calibration import get_or_create
    cal = get_or_create(user_id)
    cal.noise_floor = noise_rms
    cal.phase = 'silence_done'
    print(f"[Calibration] Saved noise floor for user {user_id}: {noise_rms:.4f}")
    return jsonify({'status': 'ok', 'noise_floor': noise_rms})

@app.route('/api/calibrate/voice_sample', methods=['POST'])
def calibrate_voice_sample():
    """Phase 2: Receive voice indicators during neutral speech calibration."""
    data     = request.json or {}
    user_id  = data.get('user_id', 'default')
    indicators = data.get('indicators', {})

    from calibration import get_or_create
    cal = get_or_create(user_id)
    cal.phase = 'voice_calibrating'
    cal.add_voice_sample(indicators)
    return jsonify({'status': 'ok', 'samples': len(cal.samples_voice)})

@app.route('/api/calibrate/face_sample', methods=['POST'])
def calibrate_face_sample():
    """Phase 3: Receive face indicators during neutral face calibration."""
    data       = request.json or {}
    user_id    = data.get('user_id', 'default')
    indicators = data.get('indicators', {})

    from calibration import get_or_create
    cal = get_or_create(user_id)
    cal.phase = 'face_calibrating'
    cal.add_face_sample(indicators)
    return jsonify({'status': 'ok', 'samples': len(cal.samples_face)})

@app.route('/api/calibrate/finalize', methods=['POST'])
def calibrate_finalize():
    """Compute final baseline statistics from all collected samples."""
    data    = request.json or {}
    user_id = data.get('user_id', 'default')

    from calibration import get_or_create
    cal = get_or_create(user_id)
    voice_ok = cal.finalize_voice()
    face_ok  = cal.finalize_face()

    print(f"[Calibration] Finalized for user {user_id}. Voice ok: {voice_ok}, Face ok: {face_ok}")
    return jsonify({
        'status':      'complete' if (voice_ok and face_ok) else 'partial',
        'calibration': cal.to_dict(),
    })

@app.route('/api/calibrate/status', methods=['GET'])
def calibrate_status():
    user_id = request.args.get('user_id', 'default')
    from calibration import get_or_create
    cal = get_or_create(user_id)
    return jsonify(cal.to_dict())

@app.route('/api/stream/fused', methods=['GET'])
def stream_fused():
    """
    SSE stream emitting fused predictions every 2 seconds.
    """
    request.environ['eventlet.minimum_write_chunk_size'] = 1
    request.environ['eventlet.minimum_chunk_size'] = 1
    print("[Fusion Engine] Client connected to fused SSE stream.")

    def generate():
        yield ":" + " " * 4096 + "\n\n"
        while True:
            all_scores = score_buffer.read_all()

            if not all_scores:
                data = json.dumps({'status': 'waiting', 'modalities_active': 0})
                print("[Fusion Engine] Waiting for active modalities...")
            else:
                probs = {k: v['score'] for k, v in all_scores.items()}
                confs = {k: v['indicators'].get('landmark_confidence', 0.7)
                         for k, v in all_scores.items()}

                fused = fuse_predictions(probs, confs, fusion_mode='reliability')
                fused['status'] = 'active'
                fused['modalities_active'] = len(all_scores)
                fused['per_modality'] = {
                    k: {'score': round(v['score'], 3)} for k, v in all_scores.items()
                }
                data = json.dumps(fused)
                print(f"[Fusion Engine] Fused Level: {fused['stress_level']} ({fused['fused_score']:.3f}) | Active: {list(probs.keys())}")

            yield f"data: {data}\n\n"
            eventlet.sleep(1)

    return app.response_class(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        }
    )

# ---------------------------
# Real-Time WebSocket Events
# ---------------------------

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    stream_processor.initialize_session(request.sid)
    emit('status', {'msg': 'Connected to streaming server'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    stream_processor.remove_session(request.sid)

@socketio.on('stream_audio')
def handle_audio_stream(data):
    """
    Handle incoming audio buffer stream.
    Expects data = {'audio': [float array]}
    """
    if 'audio' not in data:
        return
        
    session_id = request.sid
    # Get sensitivity from data (default 0.5)
    sensitivity = data.get('sensitivity', 0.5)
    
    result = stream_processor.process_audio_chunk(
        session_id, 
        data['audio'], 
        data.get('sr', 44100),
        sensitivity=sensitivity
    )
    
    emit('stress_update', {'type': 'audio', 'result': result})

@socketio.on('stream_video')
def handle_video_stream(data):
    """
    Handle incoming video frame.
    Expects data = {'image': 'base64string'}
    """
    try:
        if 'image' not in data:
            return
            
        # Get sensitivity from data (default 0.5)
        sensitivity = data.get('sensitivity', 0.5)
            
        result = stream_processor.process_video_frame(
            data['image'],
            sensitivity=sensitivity
        )
        
        if 'error' not in result:
            emit('stress_update', {
                'type': 'video', 
                'result': result
            })
    except Exception as e:
        print(f"Stream Video Error: {e}")

if __name__ == '__main__':
    print("Starting Multimodal Stress Detection API...")
    print(f"Model trained: {model.is_trained}")
    
    # Waitress does not support WebSockets/Socket.IO, so we use eventlet via socketio.run
    print("Starting SocketIO server on http://localhost:5000...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False, minimum_chunk_size=1)
