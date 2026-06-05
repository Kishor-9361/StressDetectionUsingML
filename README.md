# Multimodal Stress Detection System

A real-time stress detection system that uses three physiological modalities — **facial expressions**, **voice acoustics**, and **physiological signals** — to detect and quantify stress levels using machine learning.

## Quick Start

```bash
# Start both backend and frontend
run.bat
```

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

## Project Structure

```
StressDetectionUsingML/
│
├── run.bat                       # One-click launcher (starts backend + frontend)
├── README.md
├── TEST_GUIDE.md                 # Comprehensive testing guide
│
├── backend/                      # Flask + SocketIO API server
│   ├── app.py                    # Main application entry point (API routes, SSE, WebSocket)
│   ├── model.py                  # MultimodalStressDetector: feature extraction + inference
│   ├── realtime_core.py          # StressStreamProcessor: real-time session management
│   ├── voice_worker.py           # High-speed vocal feature extraction (autocorrelation pitch)
│   ├── calibration.py            # Per-user baseline calibration engine
│   ├── score_buffer.py           # Rolling score buffer with smoothing
│   ├── requirements.txt          # Python dependencies
│   │
│   ├── expert_models/            # Production lightweight expert models (~1-2 MB each)
│   │   ├── face_expert_lightweight.pkl
│   │   ├── face_scaler_lightweight.pkl
│   │   ├── voice_expert_lightweight.pkl
│   │   ├── voice_scaler_lightweight.pkl
│   │   ├── physio_expert.pkl
│   │   └── physio_scaler.pkl
│   │
│   ├── face_landmarker.task      # MediaPipe Tasks face landmarker model
│   ├── facial_expert_model.pkl   # Legacy full-size facial model (multimodal endpoint)
│   ├── voice_expert_model.pkl    # Legacy full-size voice model (multimodal endpoint)
│   ├── physio_expert_model.pkl   # Legacy physio model (multimodal endpoint)
│   │
│   ├── run_and_verify_all.py     # Automated 23-test verification suite
│   ├── tests/                    # Manual and API test scripts
│   │   ├── test_api_endpoints.py
│   │   ├── test_health.py
│   │   ├── test_realtime_performance.py
│   │   ├── test_socket_stream.py
│   │   └── test_voice_model.py
│   │
│   ├── uploads/                  # Temporary file uploads (auto-cleaned)
│   └── training/                 # Offline training scripts & datasets (not for production)
│       ├── train_model.py        # Train voice + physio experts
│       ├── train_face_expert.py  # Train facial expression expert
│       ├── colab_training.py     # Google Colab training notebook (standalone)
│       ├── integrate_dataset.py  # StressID dataset integration
│       ├── extract_face_indicators_offline.py
│       ├── Dataset/              # labels.csv
│       └── Feature Extraction/   # Pre-computed feature CSVs
│
├── frontend/                     # React application
│   ├── public/
│   │   ├── index.html
│   │   ├── facePostWorker.js     # Web Worker: offloads face POST requests off main thread
│   │   └── mediapipe/            # MediaPipe WASM bundles (face landmarker)
│   └── src/
│       ├── App.js                # Root: renders Dashboard directly
│       ├── index.js
│       ├── index.css
│       ├── theme.css             # Global design tokens & CSS variables
│       ├── pages/
│       │   └── Dashboard.js     # Main page: upload analysis + real-time monitor tabs
│       └── components/
│           ├── RealtimeMonitor.jsx   # Live telemetry dashboard with SSE fusion stream
│           ├── CalibrationWizard.jsx # 3-phase personal baseline calibration wizard
│           ├── FaceStream.jsx        # Webcam capture + MediaPipe landmark streaming
│           └── WaveformRecorder.jsx  # Microphone capture + chunked audio streaming
│
├── docs/                         # Architecture & optimization documentation
│   ├── OPTIMIZATION_STRATEGY.md
│   └── PHASE1_REALTIME_ARCHITECTURE.md
│
└── reports/                      # Performance benchmarks & analysis reports
    ├── performance_charts.png
    ├── performance_report.md
    └── stress_pattern_reliability_report.md
```

## Architecture

The system uses a **three-expert fusion** approach:

| Expert | Input | Model | Latency |
|--------|-------|-------|---------|
| Facial | Webcam frames → MediaPipe landmarks | Gradient Boosting | ~8 ms |
| Voice | Microphone chunks → MFCC + pitch | Gradient Boosting | ~140 ms |
| Physiological | EEG + GSR signals | Random Forest | ~5 ms |

Results are **fused via a weighted confidence engine** and streamed in real-time to the frontend via Server-Sent Events (SSE).

### Key Design Decisions

- **Personal Baseline Calibration**: A 3-phase wizard (silence → voice → face) builds a personal reference frame, so stress is measured _relative to the user's own calm state_.
- **OS Thread Pool**: CPU-heavy `librosa` audio processing runs in `eventlet.tpool` to avoid GIL blocking.
- **Web Worker**: Face POST requests run in a background browser thread to keep the webcam feed smooth at 30fps.

## Running Tests

```bash
cd backend
python run_and_verify_all.py   # Full 23-test automated suite
```

## Dependencies

### Backend
```bash
cd backend
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
npm start
```