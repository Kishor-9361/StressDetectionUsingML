# Multimodal Stress Detection System

An intelligent stress detection system for workspace professionals using facial expressions, voice patterns, and physiological signals (EEG, GSR) through advanced multimodal fusion techniques.

## Features

- **Multimodal Analysis**: Combines facial (webcam/upload), vocal, and physiological data
- **Real-time Detection**: Instant webcam capture and analysis
- **Decision-Level Fusion**: SVM-based fusion with multiple strategies (Average, Sum, Product, Maximum)
- **Individual Modality Insights**: View predictions from each modality separately
- **Flexible Input**: Works with any combination of 1, 2, or 3 modalities
- **RESTful API**: Flask backend with CORS support
- **Modern Dashboard**: Responsive React interface

## Installation

### Backend Setup

```bash
# Clone repository
git clone https://github.com/yourusername/stress-detection.git
cd stress-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install flask flask-cors numpy pandas scikit-learn scipy opencv-python librosa soundfile imbalanced-learn joblib

# Train model
python train_model.py
# Choose option 2 for demo model or option 1 for dataset training

# Start backend
python app.py
```

### Frontend Setup

```bash
# Install Node.js dependencies
npm install lucide-react

# Start development server
npm start
```

## Usage

### Quick Start

1. Start Flask backend: `python app.py`
2. Open dashboard at `http://localhost:3000`
3. Upload face image, voice recording, or enter EEG/GSR data
4. Click "Analyze Stress Level"
5. View results with overall stress percentage and individual modality predictions

### API Example

```bash
curl -X POST http://localhost:5000/api/multimodal/analyze \
  -F "face_image=@photo.jpg" \
  -F "voice_audio=@audio.wav" \
  -F "eeg_data=0.5,0.7,0.6,0.8" \
  -F "gsr_data=2.1,2.3,2.2,2.4"
```

### Python Example

```python
from model import MultimodalStressDetector

model = MultimodalStressDetector()
model.load_model('multimodal_stress_model.pkl')

facial_features = model.extract_facial_features('photo.jpg')
voice_features = model.extract_voice_features('audio.wav')
phys_features = model.extract_physiological_features(
    eeg_data=[0.5, 0.7, 0.6, 0.8],
    gsr_data=[2.1, 2.3, 2.2, 2.4]
)

result = model.predict(
    facial_features=facial_features,
    voice_features=voice_features,
    phys_features=phys_features,
    fusion='average'
)

print(f"Stress Level: {result['stress_level']} ({result['percentage']:.1f}%)")
```

## Model Details

### Architecture
- **Algorithm**: Support Vector Machines (SVM) with RBF kernel
- **Fusion**: Decision-level fusion using Average rule
- **Preprocessing**: StandardScaler + PCA (95% variance)
- **Class Balancing**: SMOTE resampling

### Methodology

**Feature Extraction:**
- **Facial Analysis**: OpenCV Haar Cascades for face detection, extracting facial dimensions, texture features (Sobel gradients), eye detection, and color statistics
- **Voice Analysis**: Librosa for audio processing, extracting MFCCs, spectral features, pitch, and energy patterns
- **Physiological Analysis**: Statistical feature extraction from EEG and GSR time-series data

**Machine Learning Pipeline:**
1. **Individual Modality Training**: Three separate SVM classifiers trained independently on facial, voice, and physiological features
2. **Feature Engineering**: StandardScaler normalization followed by PCA dimensionality reduction (95% variance retained)
3. **Classification Models Compared**:
   - Support Vector Machines (SVM) - **Selected (Best Performance)**
   - Random Forest Classifier
   - Convolutional Neural Networks (CNN) - Explored for image-based features
   - K-Nearest Neighbors (KNN)
   - Multi-Layer Perceptron (MLP)
4. **Decision-Level Fusion**: Probability outputs from each SVM combined using fusion rules (Average, Sum, Product, Maximum)
5. **Final Prediction**: Weighted combination of modality predictions produces overall stress classification

**Why SVM?**
- Superior performance on limited datasets (62-68% F1-score)
- Robust to overfitting with RBF kernel
- Efficient training and inference times
- Better generalization compared to CNN on our dataset size

### Features
- **Facial (20 dims)**: Face dimensions, intensity stats, texture, eye detection, color features
- **Voice (36 dims)**: MFCCs, spectral centroid/rolloff, zero crossing rate, RMS energy, pitch
- **Physiological (12 dims)**: EEG and GSR statistics (mean, std, min, max, skewness, kurtosis)

### API Endpoints

```
GET  /api/health                  # Health check
POST /api/multimodal/analyze      # Multimodal analysis
POST /api/face/upload             # Face-only analysis
POST /api/voice/upload            # Voice-only analysis
POST /api/webcam/capture          # Webcam capture
```

## Project Structure

```
stress-detection/
├── model.py              # Core ML model
├── app.py                # Flask API
├── train_model.py        # Training script
├── requirements.txt      # Dependencies
├── frontend/
│   └── src/
│       └── Dashboard.js  # React UI
└── multimodal_stress_model.pkl
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m 'Add YourFeature'`
4. Push to branch: `git push origin feature/YourFeature`
5. Submit pull request

## License

MIT License - see LICENSE file for details.

## Contributors
 Saathviga B, Kaviya R