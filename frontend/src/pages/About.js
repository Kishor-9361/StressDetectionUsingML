import React from "react";
import "../theme.css";

export default function About() {
  return (
    <div className="container py-5">
      <div className="text-center mb-5">
        <h2 className="neon-text">About The Platform</h2>
        <p className="lead">
          Revolutionizing workplace wellness through intelligent stress detection technology
        </p>
      </div>

      <div className="row mb-5">
        <div className="col-md-6">
          <div className="neon-card fade-in-up">
            <h3>Project Mission</h3>
            <p>
              To develop a robust, real-time stress detection system using multimodal analysis.
              By combining facial expression, audio features, and physiological signals,
              we aim to improve stress classification accuracy beyond unimodal systems.
            </p>
          </div>
        </div>
        <div className="col-md-6">
          <div className="neon-card slide-in-right">
            <h3>Technical Vision</h3>
            <p>
              We implement decision-level fusion techniques to integrate heterogeneous data sources.
              Our modular architecture allows for flexible deployment and easy integration of
              new sensing modalities (e.g., thermal imaging, heart rate variability).
            </p>
          </div>
        </div>
      </div>

      <div className="row mb-5">
        <div className="col-12">
          <div className="neon-card">
            <h3 className="text-center mb-4">How It Works</h3>
            <div className="row">
              <div className="col-md-6">
                <h4>Data Modalities</h4>
                <ul className="list-unstyled">
                  <li>📸 <strong>Facial:</strong> Haar Cascades, CNN Feature Extraction</li>
                  <li>🎤 <strong>Audio:</strong> MFCC, Chroma, Spectral Contrast</li>
                  <li>⚡ <strong>Physio:</strong> EEG (Alpha/Beta waves), GSR Analysis</li>
                </ul>
              </div>
              <div className="col-md-6">
                <h4>Processing Pipeline</h4>
                <ul className="list-unstyled">
                  <li>1. Data Acquisition (Webcam/Mic/Sensors)</li>
                  <li>2. Preprocessing & Feature Extraction</li>
                  <li>3. Unimodal Classification (RF/SVM)</li>
                  <li>4. Decision Fusion (Weighted Average)</li>
                  <li>5. Final Stress Level Prediction</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        <div className="col-md-4">
          <div className="neon-card text-center">
            <h4>Data Privacy</h4>
            <p>No cloud storage. All processing occurs locally in the Flask backend session.</p>
          </div>
        </div>
        <div className="col-md-4">
          <div className="neon-card text-center">
            <h4>Model Architecture</h4>
            <p>Ensemble of optimized classifiers trained on the StressID benchmark dataset.</p>
          </div>
        </div>
        <div className="col-md-4">
          <div className="neon-card text-center">
            <h4>Output format</h4>
            <p>JSON response with probability scores, confidence levels, and modality breakdowns.</p>
          </div>
        </div>
      </div>
    </div>
  );
}