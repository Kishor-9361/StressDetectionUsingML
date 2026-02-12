from flask import Flask, jsonify, request
from flask_cors import CORS
from voice import predict_stress_from_file, record_and_predict, load_model, LabelEncoder
import os
import numpy as np
import logging
import librosa
from scipy.io import wavfile
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load voice model and label encoder
try:
    voice_model = load_model('stress_detection_model.h5')
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)
    logger.info("Voice model and label encoder loaded successfully")
except Exception as e:
    logger.error(f"Failed to load voice model or label encoder: {str(e)}")
    raise

# Load face model
try:
    face_model = keras_load_model('final_stress_detection_model.h5')
    logger.info("Face model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load face model: {str(e)}")
    raise

# Map voice stress labels to friendly text and percentages
voice_stress_label_map = {
    'non_stress': {'display': 'No Stress', 'percentage': 10},
    'low_stress': {'display': 'Low Stress', 'percentage': 30},
    'moderate_stress': {'display': 'Moderate Stress', 'percentage': 50},
    'high_stress': {'display': 'High Stress', 'percentage': 80}
}

# Map face stress labels to friendly text and percentages
face_stress_label_map = {
    'No Stress': {'display': 'No Stress', 'percentage': 10},
    'Stress': {'display': 'Stress', 'percentage': 70}
}

class StressDetectionModel:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.model = face_model

    def predict_single_image(self, image_path):
        """Predict stress/no-stress from a single image"""
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            img = Image.open(image_path).convert('RGB').resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = self.model.predict(img_array)[0][0]

            predicted_class = "Stress" if pred > 0.5 else "No Stress"
            confidence = float(pred if pred > 0.5 else 1 - pred)
            percentage = face_stress_label_map[predicted_class]['percentage']

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "percentage": percentage
            }
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'Welcome to the Stress Detection API. Use /api/voice/record, /api/voice/upload (WAV, MP3), or /api/face/upload (JPG, JPEG, PNG).'
    })

@app.route('/api/voice/record', methods=['POST'])
def voice_record():
    try:
        duration = request.json.get('duration', 3)
        logger.info(f"Starting voice recording for {duration} seconds")
        stress_level = record_and_predict(duration=duration, model=voice_model, label_encoder=label_encoder)
        if stress_level:
            mapped_label = voice_stress_label_map.get(stress_level, {'display': stress_level, 'percentage': 50})
            logger.info(f"Voice recording analysis successful: {stress_level}")
            return jsonify({
                'status': 'success',
                'stress_level': mapped_label['display'],
                'percentage': mapped_label['percentage']
            })
        else:
            logger.error("Failed to analyze voice recording")
            return jsonify({'status': 'error', 'message': 'Failed to analyze voice recording'}), 500
    except Exception as e:
        logger.error(f"Error in voice_record: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/voice/upload', methods=['POST'])
def voice_upload():
    temp_path = None
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        filename = file.filename.lower()
        if not (filename.endswith('.wav') or filename.endswith('.mp3')):
            logger.error(f"Invalid file format: {filename}")
            return jsonify({'status': 'error', 'message': 'File must be WAV or MP3 format'}), 400
        
        temp_path = 'temp_upload.wav'
        temp_input_path = 'temp_input_audio' + os.path.splitext(filename)[1]
        file.save(temp_input_path)
        logger.info(f"Saved uploaded file to {temp_input_path}")

        if filename.endswith('.wav'):
            os.rename(temp_input_path, temp_path)
            logger.info(f"Renamed WAV file to {temp_path}")
        else:
            try:
                y, sr = librosa.load(temp_input_path, sr=None)
                y = y / np.max(np.abs(y))
                wavfile.write(temp_path, sr, (y * 32767).astype(np.int16))
                logger.info(f"Converted MP3 to WAV and saved to {temp_path}")
            except Exception as e:
                logger.error(f"Error converting MP3: {str(e)}")
                return jsonify({'status': 'error', 'message': f'Failed to process MP3 file: {str(e)}'}), 500
            finally:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                    logger.info(f"Removed temporary input file {temp_input_path}")

        stress_level = predict_stress_from_file(temp_path, voice_model, label_encoder)
        if stress_level:
            mapped_label = voice_stress_label_map.get(stress_level, {'display': stress_level, 'percentage': 50})
            logger.info(f"File analysis successful: {stress_level}")
            return jsonify({
                'status': 'success',
                'stress_level': mapped_label['display'],
                'percentage': mapped_label['percentage']
            })
        else:
            logger.error("Failed to analyze audio file")
            return jsonify({'status': 'error', 'message': 'Failed to analyze audio file'}), 500
    except Exception as e:
        logger.error(f"Error in voice_upload: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Removed temporary file {temp_path}")

@app.route('/api/face/upload', methods=['POST'])
def face_upload():
    temp_path = None
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'status': 'error', 'message': 'No image uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'status': 'error', 'message': 'No image selected'}), 400
        
        filename = file.filename.lower()
        if not (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')):
            logger.error(f"Invalid file format: {filename}")
            return jsonify({'status': 'error', 'message': 'File must be JPG, JPEG, or PNG format'}), 400
        
        temp_path = 'temp_image.jpg'
        file.save(temp_path)
        logger.info(f"Saved uploaded image to {temp_path}")

        face_detector = StressDetectionModel(img_size=(128, 128))
        result = face_detector.predict_single_image(temp_path)
        
        if "error" in result:
            logger.error(f"Error in face prediction: {result['error']}")
            return jsonify({'status': 'error', 'message': result['error']}), 500
        
        logger.info(f"Face analysis successful: {result['predicted_class']}")
        return jsonify({
            'status': 'success',
            'stress_level': result['predicted_class'],
            'percentage': result['percentage']
        })
    except Exception as e:
        logger.error(f"Error in face_upload: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Removed temporary image file {temp_path}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)