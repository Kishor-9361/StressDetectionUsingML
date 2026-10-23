# 🚀 Multimodal Stress Detection — Self-Contained Deployment Package

This directory (`deployment/`) contains the complete, self-contained, lightweight bundle required to deploy and run the entire Multimodal Stress Detection system (Frontend, Backend API, and Production Models) without the 22+ GB research datasets or raw media archives.

Total package size: **~85 MB** (compared to the 22.8 GB research repository).

---

## 📁 Package Architecture

```
deployment/
├── backend/                        # Flask + Socket.IO + MediaPipe API Service
│   ├── app.py                      # Main entrypoint with REST, Socket.IO & static UI routing
│   ├── requirements.txt            # CPU-optimized Python dependencies
│   ├── Procfile                    # Render / Heroku process declaration
│   ├── render.yaml                 # One-click Render Infrastructure-as-Code specification
│   ├── face_landmarker.task        # MediaPipe 468-point face mesh model
│   ├── model.py, realtime_core.py  # Feature extractors and real-time buffer pipeline
│   ├── core/                       # Feature contracts and runtime locks
│   ├── explainability/             # SHAP local attribution generator
│   ├── monitoring/                 # Drift tracking and replay verification
│   └── runtime/                    # Unified Phase 7 Inference Runtime
│       ├── runtime_engine.py       # Dual-champion model router
│       ├── conv_moe_mf.py          # ConvMoE-MF PyTorch architecture
│       ├── ssvb_casa_ais.py        # SSVB-CASA-AIS PyTorch architecture
│       └── models/                 # Deployed deep learning weights (.pt)
│           ├── ssvb_casa_ais_production_cnn_baseline_grl.pt (Champion)
│           ├── ssvb_casa_ais_production.pt (ConvMoE-MF)
│           └── ssvb_casa_ais_production_ssvb.pt
├── configs/                        # System contracts (YAML schemas)
│   ├── feature_contract.yaml
│   ├── api_contract.yaml
│   └── performance_contract.yaml
├── models/                         # Model Registry & Weights
│   ├── backend_selected/           # Production Random Forest (Secondary/CPU)
│   │   ├── model_weights.pkl       # 50-tree scikit-learn estimator
│   │   ├── combined_production_rf.pkl # Wrapped dictionary with 360-feature schema
│   │   └── model_card.md           # Model performance card
│   ├── production_deep/            # All Phase 3 production deep checkpoints & configs
│   │   ├── ssvb_casa_ais_production_cnn_baseline_grl.pt
│   │   ├── ssvb_casa_ais_production_cnn_baseline_grl.json
│   │   └── ssvb_casa_ais_production_conv_moe_mf.pt
│   ├── research_champion/          # Modality experts & dynamic router
│   ├── fallback_models/            # Lightweight fallback estimators
│   └── registry.json               # SHA256 version-locked registry
├── frontend/                       # React 19 Biometric Dashboard
│   ├── build/                      # Pre-compiled, production-ready static assets
│   ├── src/                        # Full React JSX & component source code
│   ├── public/                     # Public icons, audio, and HTML templates
│   ├── package.json                # Frontend package specification
│   └── vercel.json                 # Vercel deployment and routing rules
├── Dockerfile                      # Production container recipe (Python 3.10-slim)
├── docker-compose.yml              # Single-command local container orchestration
├── start.bat                       # Windows 1-click launcher
├── start.sh                        # Linux / macOS 1-line launcher
└── README.md                       # This deployment guide
```

---

## ⚡ Quick Start Options

### Option 1: Direct Local Execution (Python 3.10+)

The backend comes pre-configured to automatically serve both the **API endpoints** and the **built React frontend** directly from `http://localhost:5000`:

1. Open a terminal in `deployment/backend`:
   ```bash
   cd deployment/backend
   pip install -r requirements.txt
   python app.py
   ```
   *(Or on Windows, double-click `deployment/start.bat`)*

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

---

### Option 2: Docker / Container Deployment

Deploy the entire stack in an isolated, production container:

```bash
cd deployment
docker compose up --build
```

Access the service at `http://localhost:5000`.

---

### Option 3: Two-Tier Cloud Deployment (Render + Vercel)

This matches the existing production architecture:

#### A. Backend on Render (or Railway / Fly.io / AWS ECS)
1. Point your service to the `deployment/backend` directory.
2. Build command: `pip install -r requirements.txt`
3. Start command: `gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app:app`
4. Set Environment Variables:
   - `PYTHON_VERSION`: `3.10.12`
   - `FRONTEND_URL`: Your Vercel frontend URL (or `*`)
   - `GEMINI_API_KEY`: (Optional) Your Gemini API Key for chatbot responses.

#### B. Frontend on Vercel
1. Set the root directory in Vercel to `deployment/frontend`.
2. Framework Preset: `Create React App`
3. Build command: `npm run build`
4. Output directory: `build`
5. Set Environment Variable:
   - `REACT_APP_BACKEND_URL`: Your Render backend URL (e.g., `https://your-backend.onrender.com`).

---

## 🧠 Active Production Models in this Bundle

1. **Primary Champion (`CNNBaseline+GRL`)**:
   - Location: `deployment/backend/runtime/models/ssvb_casa_ais_production_cnn_baseline_grl.pt`
   - Architecture: 3× 1D-CNN + Gradient Reversal Layer ($\alpha = 0.02$)
   - Features: 69 temporal multi-modal features
   - Optimal Decision Threshold: `p = 0.34`
   - Out-of-sample Cross-Validation: **71.62% Accuracy, 86.22% Recall, 0.7321 F1-Score, 0.7509 ROC-AUC**

2. **Secondary / CPU Fallback (`Random Forest Master Ensemble`)**:
   - Location: `deployment/models/backend_selected/model_weights.pkl`
   - Architecture: 50 Decision Trees across 360 windowed statistics
   - Latency: **< 5 ms on CPU**
   - Out-of-sample Cross-Validation: **74.30% Accuracy, 0.6325 F1-Score**

---

## 🔍 Health & Verification API

Once started, you can verify that all models are active via:
- Health check: `GET http://localhost:5000/api/health`
- Active model info: `GET http://localhost:5000/api/model/version`
- Realtime inference: `POST http://localhost:5000/api/predict/realtime`
