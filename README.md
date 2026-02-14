# Multimodal Stress Detection System

A comprehensive stress detection platform analyzing facial expressions, vocal patterns, and physiological data (EEG/GSR) to provide real-time stress assessments.

## 🚀 Quick Start (Easiest Method)

1. **Double-click `run.bat`** in the project root folder.
   - This will automatically launch both the Backend (Python) and Frontend (React) servers in separate windows.
   - The application will open at `http://localhost:3000`.

## 📂 Project Structure

- **`backend/`**: Python Flask API & Machine Learning Models
  - `app.py`: Main API server
  - `model.py`: Core logic for multimodal fusion
  - `train_model.py`: Script to train expert models
  - `expert_models/`: Contains the trained `.pkl` files for Face, Voice, and Physio experts
- **`frontend/`**: React Web Application
  - `src/`: Source code for UI components and pages
  - `public/`: Static assets
- **`datasets/`** (Excluded from Git):
  - `facesData`: Facial expression dataset
  - `StressID`: multimodal stress dataset

## 🛠️ Manual Installation & Run

### Prerequisites
- Python 3.8+
- Node.js & npm

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
# Server runs on http://localhost:5000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm start
# App runs on http://localhost:3000
```

## 🧠 Model Architecture
The system uses a **Late Fusion** strategy with three specialized "Expert" models:
1. **Facial Expert**: Random Forest trained on `facesData` (CK+ / FER2013 style images).
2. **Voice Expert**: Random Forest trained on `StressID` audio features (MFCC, Spectral, etc.).
3. **Physiological Expert**: Random Forest trained on `StressID` biosignals (EEG/GSR).

Predictions from these experts are combined using a weighted average to produce a final robust stress assessment.

## 📊 Features
- **Real-time Monitoring**: Live analysis from webcam and microphone.
- **File Upload**: Analyze pre-recorded videos, audio files, or CSV physio data.
- **Dashboard**: Interactive visualization of stress levels and modality breakdowns.