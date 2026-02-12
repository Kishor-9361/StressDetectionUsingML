# import streamlit as st
# import os
# from PIL import Image
# import numpy as np
# import plotly.graph_objects as go
# from tensorflow.keras.models import load_model as keras_load_model

# # ---------------------------
# # StressDetectionModel Wrapper
# # ---------------------------
# class StressDetectionModel:
#     def __init__(self, img_size=(128, 128)):
#         self.img_size = img_size
#         self.model = None

#     def predict_single_image(self, image_path):
#         """Predict stress/no-stress from a single image using self.model"""
#         if self.model is None:
#             return {"error": "Model not loaded"}

#         try:
#             img = Image.open(image_path).convert('RGB')
#             img = img.resize(self.img_size)
#             img_array = np.array(img) / 255.0
#             img_array = np.expand_dims(img_array, axis=0)

#             pred = self.model.predict(img_array)[0][0]

#             return {
#                 "stress_probability": float(pred),
#                 "no_stress_probability": float(1 - pred),
#                 "predicted_class": "Stress" if pred > 0.5 else "No Stress",
#                 "confidence": float(max(pred, 1 - pred))
#             }

#         except Exception as e:
#             return {"error": str(e)}

# # ---------------------------
# # Streamlit App
# # ---------------------------

# # Page configuration
# st.set_page_config(
#     page_title="Stress Detection Model",
#     page_icon="😊",
#     layout="wide"
# )

# # Initialize session state
# if 'model_loaded' not in st.session_state:
#     st.session_state.model_loaded = False
# if 'model' not in st.session_state:
#     st.session_state.model = None

# @st.cache_resource
# def load_trained_model(model_path):
#     """Load trained Keras model"""
#     if os.path.exists(model_path):
#         model = keras_load_model(model_path)
#         return model
#     else:
#         st.error(f"Model file not found: {model_path}")
#         st.info("Please train the model first by running: python stress_model.py")
#         return None

# # ---------------------------
# # Main app
# # ---------------------------
# def main():
#     st.title("😊 Stress Detection Model")
    
#     page = st.sidebar.selectbox("Choose a page", ["Model Testing", "Model Training", "About"])
    
#     if page == "Model Testing":
#         model_testing_page()
#     elif page == "Model Training":
#         model_training_page()
#     else:
#         about_page()

# # ---------------------------
# # Model Testing Page
# # ---------------------------
# def model_testing_page():
#     st.header("🧠 Model Testing")
    
#     # Load model
#     if not st.session_state.model_loaded:
#         with st.spinner("Loading model..."):
#             model_path = r'E:\Sem 5\SDP_15_oct\ml model\final_stress_detection_model.h5'
#             st.session_state.model = load_trained_model(model_path)
#             if st.session_state.model is not None:
#                 st.session_state.model_loaded = True
#                 st.success("Model loaded successfully!")
#             else:
#                 st.error("Failed to load model.")
#                 return
    
#     uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    
#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
        
#         # Save temporarily
#         temp_path = "temp_image.jpg"
#         image.save(temp_path)
        
#         # Predict using wrapper
#         wrapper_model = StressDetectionModel(img_size=(128,128))
#         wrapper_model.model = st.session_state.model
#         result = wrapper_model.predict_single_image(temp_path)
        
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
        
#         if "error" in result:
#             st.error(result["error"])
#         else:
#             stress_prob = result['stress_probability']
#             no_stress_prob = result['no_stress_probability']
#             predicted_class = result['predicted_class']
#             confidence = result['confidence']
            
#             # Prediction display
#             if predicted_class == "Stress":
#                 st.markdown(f"<h2 style='color:red'>🚨 {predicted_class} Detected</h2>", unsafe_allow_html=True)
#             else:
#                 st.markdown(f"<h2 style='color:green'>😊 {predicted_class} Detected</h2>", unsafe_allow_html=True)
            
#             st.metric("Confidence", f"{confidence:.2%}")
            
#             # Probability bar chart
#             fig = go.Figure(data=[go.Bar(
#                 x=['No Stress', 'Stress'],
#                 y=[no_stress_prob, stress_prob],
#                 marker_color=['#4caf50', '#f44336'],
#                 text=[f'{no_stress_prob:.2%}', f'{stress_prob:.2%}'],
#                 textposition='auto'
#             )])
#             fig.update_layout(title="Stress Detection Probabilities", yaxis=dict(range=[0, 1]))
#             st.plotly_chart(fig, use_container_width=True)

# # ---------------------------
# # Model Training Page
# # ---------------------------
# def model_training_page():
#     st.header("🏋️ Model Training")
#     st.info("Training code runs via stress_model.py. It may take several hours.")
#     if st.button("🚀 Train Model"):
#         st.warning("⚠️ Ensure your dataset is ready and sufficient resources are available.")
#         import subprocess, sys
#         subprocess.run([sys.executable, "stress_model.py"])

# # ---------------------------
# # About Page
# # ---------------------------
# def about_page():
#     st.header("📖 About")
#     st.write("This project detects stress from facial images using a trained deep learning model.")
#     st.write("Upload an image in 'Model Testing' to see predictions.")

# # ---------------------------
# # Run the app
# # ---------------------------
# if __name__ == "__main__":
#     main()

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from realtime_core import StressStreamProcessor
import os
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import cv2
import librosa
from model import MultimodalStressDetector

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

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

# Try to load pre-trained model if it exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'multimodal_stress_model.pkl')
if os.path.exists(MODEL_PATH):
    try:
        model.load_model(MODEL_PATH)
        print("Pre-trained model loaded successfully!")
    except Exception as e:
        print(f"Could not load pre-trained model: {e}")
        print("Please train the model first using train_model.py")

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': model.is_trained
    })

@app.route('/api/multimodal/analyze', methods=['POST'])
def analyze_multimodal():
    """
    Multimodal stress analysis endpoint
    Accepts: image file, audio file, EEG data, GSR data
    """
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
            return jsonify({
                'status': 'error',
                'message': result['error']
            }), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/face/upload', methods=['POST'])
def analyze_face():
    """Facial stress analysis endpoint"""
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
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/voice/upload', methods=['POST'])
def analyze_voice():
    """Voice stress analysis endpoint"""
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
    if not model.is_trained:
        print("\n[WARNING]: Model not trained!")
        print("Please run train_model.py first to train the model.\n")
    
    # Use socketio.run instead of app.run
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader = False)