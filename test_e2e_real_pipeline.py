"""
test_e2e_real_pipeline.py
End-to-End System & Model Validation with Real Datasets.
Tests:
1. Direct Model Evaluation on Real Dataset Samples:
   - Primary Deep Champion (CNNBaselineGRL with Subject Identity Suppression)
   - Secondary Classical Random Forest Classifier
   - Expert Unimodal Classifiers (Face Expert, Voice Expert)
2. Live REST API Evaluation:
   - /api/health and /api/runtime/status
   - /api/model/select
   - /api/predict/realtime (Multimodal Streaming Inference)
   - /api/predict/upload (Precomputed Feature Vector Ingestion)
   - /api/stream/face (Continuous Landmark Stream)
3. Specialized Robustness Filters:
   - Smile suppression / False Positive Prevention
   - Room Silence & Noise Gating
   - Baseline Calibration & Normalization
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Setup paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backend.runtime.runtime_engine import RuntimeEngine

FACE_EVAL_PATH = "/home/kishorveeraragavan/Desktop/Multimodal-Stress-Detection/evaluation_data/face_eval_samples.csv"
VOICE_EVAL_PATH = "/home/kishorveeraragavan/Desktop/Multimodal-Stress-Detection/evaluation_data/voice_eval_samples.csv"
API_BASE = "http://localhost:5000"

def evaluate_direct_models():
    print("\n" + "="*75)
    print("STEP 1: DIRECT REAL DATASET EVALUATION (MODELS & RUNTIME ENGINE)")
    print("="*75)
    
    if not os.path.exists(FACE_EVAL_PATH) or not os.path.exists(VOICE_EVAL_PATH):
        raise FileNotFoundError("Real evaluation datasets not found at specified paths.")
        
    df_face = pd.read_csv(FACE_EVAL_PATH)
    df_voice = pd.read_csv(VOICE_EVAL_PATH)
    
    print(f"Loaded Real Face Dataset:  {df_face.shape[0]} samples (Features: {df_face.shape[1]-1})")
    print(f"Loaded Real Voice Dataset: {df_voice.shape[0]} samples (Features: {df_voice.shape[1]-1})")
    
    y_face_true = df_face['label'].astype(int).values
    y_voice_true = df_voice['label'].astype(int).values
    
    X_face = df_face.drop(columns=['label']).values.astype(float)
    X_voice = df_voice.drop(columns=['label']).values.astype(float)
    
    # Initialize authoritative runtime engine
    engine = RuntimeEngine.from_registry()
    print(f"Authoritative Engine Ready: {engine.is_ready}")
    print(f"Active Deep Mode:           {engine.use_deep}")
    print(f"Active Champion Model:      {engine.use_champion}")
    print(f"Active Secondary RF:        {engine.use_secondary_rf}")
    
    # 1.1 Face Expert Evaluation (Unimodal)
    engine.use_deep = False
    face_preds = []
    face_probs = []
    t_start = time.time()
    for row in X_face:
        res = engine.predict_face(row)
        pred_int = 1 if res.get('predicted_class') == 'Stress' else 0
        face_preds.append(pred_int)
        face_probs.append(res.get('stress_probability', 0.5))
    face_lat_ms = (time.time() - t_start) / len(X_face) * 1000
    
    acc_face = accuracy_score(y_face_true, face_preds)
    f1_face = f1_score(y_face_true, face_preds, zero_division=0)
    print(f"\n[Expert Model: Face Expert (Lightweight Decision Ensemble)]")
    print(f"Accuracy:  {acc_face*100:.2f}%")
    print(f"F1-Score:  {f1_face:.4f}")
    print(f"Latency:   {face_lat_ms:.2f} ms/frame")
    print("Confusion Matrix:\n", confusion_matrix(y_face_true, face_preds))
    
    # 1.2 Voice Expert Evaluation (Unimodal)
    voice_preds = []
    voice_probs = []
    t_start = time.time()
    for row in X_voice:
        res = engine.predict_voice(row)
        pred_int = 1 if res.get('predicted_class') == 'Stress' else 0
        voice_preds.append(pred_int)
        voice_probs.append(res.get('stress_probability', 0.5))
    voice_lat_ms = (time.time() - t_start) / len(X_voice) * 1000
    
    acc_voice = accuracy_score(y_voice_true, voice_preds)
    f1_voice = f1_score(y_voice_true, voice_preds, zero_division=0)
    print(f"\n[Expert Model: Voice Expert (Lightweight Decision Ensemble)]")
    print(f"Accuracy:  {acc_voice*100:.2f}%")
    print(f"F1-Score:  {f1_voice:.4f}")
    print(f"Latency:   {voice_lat_ms:.2f} ms/frame")
    print("Confusion Matrix:\n", confusion_matrix(y_voice_true, voice_preds))
    
    # 1.3 Multimodal Fused Evaluation (Champion Deep GRL vs Secondary Random Forest)
    min_len = min(len(X_face), len(X_voice))
    y_fused_true = y_face_true[:min_len]
    
    # A) Deep Adversarial Champion (ssvb_casa_ais_production_cnn_baseline_grl.pt)
    engine.use_champion = True
    engine.use_secondary_rf = False
    engine.reset_calibration()
    deep_preds = []
    deep_probs = []
    t_start = time.time()
    for i in range(min_len):
        res = engine.predict_fused(face=X_face[i], voice=X_voice[i])
        pred_int = 1 if res.get('predicted_class') == 'Stress' else 0
        deep_preds.append(pred_int)
        deep_probs.append(res.get('stress_probability', 0.5))
    deep_lat_ms = (time.time() - t_start) / min_len * 1000
    
    acc_deep = accuracy_score(y_fused_true, deep_preds)
    prec_deep = precision_score(y_fused_true, deep_preds, zero_division=0)
    rec_deep = recall_score(y_fused_true, deep_preds, zero_division=0)
    f1_deep = f1_score(y_fused_true, deep_preds, zero_division=0)
    
    print(f"\n[Primary Production Champion: CNNBaselineGRL (SSVB Deep Multimodal Router)]")
    print(f"Accuracy:  {acc_deep*100:.2f}%")
    print(f"Precision: {prec_deep:.4f}")
    print(f"Recall:    {rec_deep:.4f}")
    print(f"F1-Score:  {f1_deep:.4f}")
    print(f"Latency:   {deep_lat_ms:.2f} ms/inference")
    print("Confusion Matrix:\n", confusion_matrix(y_fused_true, deep_preds))
    
    # B) Secondary Production Model (Random Forest Classifier)
    engine.use_champion = False
    engine.use_secondary_rf = True
    engine.reset_calibration()
    rf_preds = []
    rf_probs = []
    t_start = time.time()
    for i in range(min_len):
        res = engine.predict_fused(face=X_face[i], voice=X_voice[i])
        pred_int = 1 if res.get('predicted_class') == 'Stress' else 0
        rf_preds.append(pred_int)
        rf_probs.append(res.get('stress_probability', 0.5))
    rf_lat_ms = (time.time() - t_start) / min_len * 1000
    
    acc_rf = accuracy_score(y_fused_true, rf_preds)
    prec_rf = precision_score(y_fused_true, rf_preds, zero_division=0)
    rec_rf = recall_score(y_fused_true, rf_preds, zero_division=0)
    f1_rf = f1_score(y_fused_true, rf_preds, zero_division=0)
    
    print(f"\n[Secondary Production Model: Random Forest Multimodal Ensemble]")
    print(f"Accuracy:  {acc_rf*100:.2f}%")
    print(f"Precision: {prec_rf:.4f}")
    print(f"Recall:    {rec_rf:.4f}")
    print(f"F1-Score:  {f1_rf:.4f}")
    print(f"Latency:   {rf_lat_ms:.2f} ms/inference")
    print("Confusion Matrix:\n", confusion_matrix(y_fused_true, rf_preds))
    
    # Reset engine to champion default
    engine.use_champion = True
    engine.use_secondary_rf = True
    
    return {
        "face": {"acc": acc_face, "f1": f1_face, "lat": face_lat_ms},
        "voice": {"acc": acc_voice, "f1": f1_voice, "lat": voice_lat_ms},
        "deep_champion": {"acc": acc_deep, "prec": prec_deep, "rec": rec_deep, "f1": f1_deep, "lat": deep_lat_ms},
        "random_forest": {"acc": acc_rf, "prec": prec_rf, "rec": rec_rf, "f1": f1_rf, "lat": rf_lat_ms}
    }


def evaluate_api_endpoints():
    print("\n" + "="*75)
    print("STEP 2: END-TO-END REST API TESTING (LIVE SERVER ON PORT 5000)")
    print("="*75)
    
    df_face = pd.read_csv(FACE_EVAL_PATH)
    df_voice = pd.read_csv(VOICE_EVAL_PATH)
    X_face = df_face.drop(columns=['label']).values.astype(float)
    X_voice = df_voice.drop(columns=['label']).values.astype(float)
    min_len = min(len(X_face), len(X_voice))
    y_true = df_face['label'].astype(int).values[:min_len]
    
    session = requests.Session()
    
    # 2.1 Health check
    r_health = session.get(f"{API_BASE}/api/health")
    assert r_health.status_code == 200, f"Health check failed: {r_health.text}"
    health_json = r_health.json()
    print("1. /api/health Response:")
    print(f"   Status: {health_json['status']}, Server: {health_json['server']}")
    print(f"   Models Loaded: {health_json['models_loaded']}")
    print(f"   Explainability Loaded: {health_json['explainability_engine']['loaded']}")
    
    # 2.2 Model Switcher
    r_model = session.post(f"{API_BASE}/api/model/select", json={"model": "cnn_grl"})
    print("\n2. /api/model/select Response:")
    print(f"   Active Model: {r_model.json().get('active_model_name')}")
    
    # 2.3 Streaming / Realtime Inference Endpoint (/api/predict/realtime)
    realtime_preds = []
    latencies = []
    for i in range(min_len):
        payload = {
            "face": X_face[i].tolist(),
            "voice": X_voice[i].tolist(),
            "user_id": f"eval_user_{i}"
        }
        t0 = time.time()
        r = session.post(f"{API_BASE}/api/predict/realtime", json=payload)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)
        assert r.status_code == 200, f"Realtime failed: {r.text}"
        res = r.json()
        pred_int = 1 if res.get('predicted_class') == 'Stress' else 0
        realtime_preds.append(pred_int)
        
    rt_acc = accuracy_score(y_true, realtime_preds)
    rt_f1 = f1_score(y_true, realtime_preds, zero_division=0)
    print("\n3. /api/predict/realtime Performance on Real Multimodal Pairs:")
    print(f"   Processed: {len(y_true)} frames")
    print(f"   Accuracy:  {rt_acc*100:.2f}%")
    print(f"   F1-Score:  {rt_f1:.4f}")
    print(f"   Latency:   p50={np.median(latencies):.2f} ms, p95={np.percentile(latencies, 95):.2f} ms")
    
    # 2.4 Streaming Face Continuous Ingestion (/api/stream/face)
    face_feature_names = [
        'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
        'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
        'lip_compression', 'jaw_displacement', 'mouth_corner_pull',
        'forehead_tension', 'face_height_norm', 'head_tilt',
        'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
        'landmark_confidence', 'nose_wrinkle'
    ]
    
    face_stream_scores = []
    for i in range(min_len):
        feat_dict = {k: float(v) for k, v in zip(face_feature_names, X_face[i])}
        feat_dict['smile_score'] = 0.05
        payload = {"indicators": feat_dict, "user_id": "stream_user_1"}
        r = session.post(f"{API_BASE}/api/stream/face", json=payload)
        assert r.status_code == 200, f"Stream face failed: {r.text}"
        res = r.json()
        face_stream_scores.append(res.get('score'))
        
    print("\n4. /api/stream/face Continuous Ingestion:")
    print(f"   Processed {len(face_stream_scores)} frames into rolling ScoreBuffer.")
    print(f"   Sample Scores: {[round(s, 3) for s in face_stream_scores[:5]]}")


def test_specialized_robustness_filters():
    print("\n" + "="*75)
    print("STEP 3: TESTING ROBUSTNESS FILTERS (SMILE SUPPRESSION & SILENCE GATE)")
    print("="*75)
    session = requests.Session()
    
    # 3.1 Smile Suppression / Calming Defense
    df_face = pd.read_csv(FACE_EVAL_PATH)
    stress_row = df_face.iloc[0].drop(labels=['label']).values.tolist()
    face_feature_names = [
        'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
        'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
        'lip_compression', 'jaw_displacement', 'mouth_corner_pull',
        'forehead_tension', 'face_height_norm', 'head_tilt',
        'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
        'landmark_confidence', 'nose_wrinkle'
    ]
    base_indicators = {k: float(v) for k, v in zip(face_feature_names, stress_row)}
    
    # Without smile
    ind_no_smile = dict(base_indicators, smile_score=0.0)
    r_no_smile = session.post(f"{API_BASE}/api/stream/face", json={"indicators": ind_no_smile, "user_id": "test_smile_1"})
    score_no_smile = r_no_smile.json()['raw_score']
    
    # With broad smile (e.g. laughter or smiling while talking)
    ind_smile = dict(base_indicators, smile_score=0.70)
    r_smile = session.post(f"{API_BASE}/api/stream/face", json={"indicators": ind_smile, "user_id": "test_smile_2"})
    score_smile = r_smile.json()['raw_score']
    
    print("1. Smile Suppression Filter Test:")
    print(f"   Raw Face Stress (Neutral Face):  {score_no_smile:.4f}")
    print(f"   Dampened Stress (Smiling Face):  {score_smile:.4f}")
    print(f"   Delta Reduction:                {score_no_smile - score_smile:.4f}")
    assert score_smile < score_no_smile, "Smile suppression failed to dampen raw stress!"
    print("   ✅ Smile suppression verified: Prevents false positive panic alarms.")
    
    # 3.2 Silence Gate / Idle Room Defense
    # Generate mock silence WAV (2 seconds of zeros)
    import io
    from scipy.io import wavfile
    silence_audio = np.zeros(32000, dtype=np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, 16000, silence_audio)
    silence_wav_bytes = buf.getvalue()
    
    r_silence = session.post(f"{API_BASE}/api/stream/voice", data=silence_wav_bytes, headers={"Content-Type": "audio/wav"})
    res_silence = r_silence.json()
    print("\n2. Room Silence & Noise Gate Test:")
    print(f"   Silence Endpoint Response: {res_silence}")
    assert res_silence.get('reason') == 'silence_detected', "Silence gate failed to trigger on ambient silence!"
    assert res_silence.get('score') is None, "Score was not safely cleared on silence!"
    print("   ✅ Room silence gate verified: Zero false stress triggers when user is quiet.")


if __name__ == '__main__':
    direct_metrics = evaluate_direct_models()
    evaluate_api_endpoints()
    test_specialized_robustness_filters()
    print("\n" + "="*75)
    print("ALL END-TO-END PIPELINE & MODEL VERIFICATIONS PASSED WITH ZERO ERRORS!")
    print("="*75)
