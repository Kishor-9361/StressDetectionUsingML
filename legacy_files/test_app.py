from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': False,
        'message': 'Test server running'
    })

@app.route('/api/multimodal/analyze', methods=['POST'])
def analyze_multimodal():
    """Test endpoint that returns fake data"""
    return jsonify({
        'status': 'success',
        'predicted_class': 'No Stress',
        'confidence': 0.85,
        'stress_probability': 0.35,
        'no_stress_probability': 0.65,
        'percentage': 35.0,
        'stress_level': 'Low',
        'individual_predictions': {
            'facial': 0.30,
            'voice': 0.40,
            'physiological': 0.35
        }
    })

if __name__ == '__main__':
    print("🧪 Test Server Starting...")
    print("This is a test server that returns fake data")
    print("Starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)