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
├── dataset_extracted/            # User-extracted StressID baseline indicator CSVs
│   ├── face_indicators_stressid.csv
│   └── voice_indicators_stressid.csv
│
├── docs/                         # Architecture & optimization documentation
│   ├── OPTIMIZATION_STRATEGY.md
│   └── PHASE1_REALTIME_ARCHITECTURE.md
│
└── reports/                      # Performance benchmarks & analysis reports
    ├── performance_charts.png
    ├── performance_report.md
    ├── project_report.md             # Comprehensive project report (June 2026)
    └── stress_pattern_reliability_report.md
```

## Architecture

The system uses a **three-expert fusion** approach:

| Expert | Input | Model | Latency |
|--------|-------|-------|---------|
| Facial | Webcam frames → MediaPipe landmarks | Voting Ensemble (GB + RF + SVM) | ~8 ms |
| Voice | Microphone chunks → 12 Acoustic Biomarkers | Gradient Boosting | ~15 ms |
| Physiological | EEG + GSR signals | Random Forest | ~5 ms |

Results are **fused via a weighted confidence engine** and streamed in real-time to the frontend via Server-Sent Events (SSE).

### Key Design Decisions

- **Interactive Biomarker Guide**: The UI includes a built-in parameter explorer for both Face and Voice. Users can dynamically learn how landmarks (like masseter clench, jaw width) and audio features (like Jitter RAP, Shimmer) are calculated, how they map to physiological stress, and instructions on how to test them.
- **Personal Baseline Calibration**: A 3-phase wizard (silence → voice → face) builds a personal reference frame, so stress is measured _relative to the user's own calm state_.
- **Soft-Voting Ensemble Classifier**: The facial expert employs an ensemble of Gradient Boosting, Random Forest, and Support Vector Machine (SVC) trained on augmented face landmark geometries balanced via SMOTE (accuracy: 65.27% under real-world noise simulation).
- **Fast Autocorrelation Pitch Extractor**: Voice feature extraction uses autocorrelation with parabolic peak interpolation. This yields an execution time of **<15ms** (from 4.6 seconds using librosa's pyin) and resolves raw jitter flatline discretization bugs.
- **15-Second Decay Buffer**: The fusion engine holds the voice stress score in its buffer for 15 seconds after speaking ends, ensuring conversation-style scoring continuity while the user is silent.
- **OS Thread Pool**: CPU-heavy audio processing runs in `eventlet.tpool` to avoid GIL blocking.
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