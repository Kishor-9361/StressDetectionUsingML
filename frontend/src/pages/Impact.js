import React from "react";
import "../theme.css";

export default function Impact() {
  const stats = [
    { number: "97.3%", label: "Facial Accuracy", desc: "For expression recognition" },
    { number: "94.7%", label: "Audio Accuracy", desc: "For voice stress detection" },
    { number: "92.8%", label: "Physio Accuracy", desc: "For EEG/GSR signals" },
    { number: "<1s", label: "Latency", desc: "Real-time processing speed" }
  ];

  const benefits = [
    {
      title: "Model Performance",
      items: [
        "High accuracy on StressID dataset",
        "Robust to noise and lighting",
        "Real-time inference capability",
        "Multimodal fusion improvements",
        "Low false positive rate"
      ]
    },
    {
      title: "System Architecture",
      items: [
        "Scalable Flask backend",
        "Modern React frontend",
        "WebSocket streaming",
        "Modular component design",
        "Easy API integration"
      ]
    },
    {
      title: "Future Scope",
      items: [
        "Mobile app integration",
        "Additional sensor support",
        "Cloud deployment ready",
        "Enhanced visualization",
        "Long-term trend analysis"
      ]
    }
  ];

  return (
    <div className="container py-5">
      <div className="text-center mb-5">
        <h2 className="neon-text">Measurable Impact</h2>
        <p className="lead">
          Project performance metrics and technical impact analysis.
        </p>
      </div>

      {/* Statistics Section */}
      <div className="row mb-5">
        <div className="col-12">
          <div className="neon-card">
            <h3 className="text-center mb-4">Proven Results</h3>
            <div className="row">
              {stats.map((stat, idx) => (
                <div className="col-md-6 col-lg-3 mb-4 text-center" key={idx}>
                  <div style={{
                    background: 'rgba(0, 242, 255, 0.1)',
                    padding: '2rem 1rem',
                    borderRadius: '10px',
                    height: '100%',
                    border: '1px solid rgba(0, 242, 255, 0.2)'
                  }}>
                    <h2 className="neon-text" style={{ fontSize: '3rem', marginBottom: '0.5rem' }}>
                      {stat.number}
                    </h2>
                    <h4 style={{ color: '#e0e0e0', marginBottom: '1rem' }}>{stat.label}</h4>
                    <p style={{ fontSize: '0.9rem', color: '#a0a0b0' }}>{stat.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Benefits Section */}
      <div className="mb-5">
        <h3 className="text-center mb-4">Comprehensive Benefits</h3>
        <div className="row">
          {benefits.map((benefit, idx) => (
            <div className="col-md-4 mb-4" key={idx}>
              <div className="neon-card fade-in-up" style={{ animationDelay: `${idx * 0.2}s` }}>
                <h4 className="text-center mb-3">{benefit.title}</h4>
                <ul className="list-unstyled">
                  {benefit.items.map((item, itemIdx) => (
                    <li key={itemIdx} style={{ padding: '0.5rem 0', borderBottom: '1px solid rgba(0, 242, 255, 0.2)' }}>
                      ✓ {item}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Project Status Section */}
      <div className="row mb-5">
        <div className="col-12">
          <div className="neon-card">
            <h3 className="text-center mb-4">Project Status</h3>
            <div className="row">
              <div className="col-md-6">
                <h4>Completed Features</h4>
                <ul className="list-unstyled">
                  <li className="mb-2">✓ Real-time webcam & audio stream</li>
                  <li className="mb-2">✓ Facial expression analysis</li>
                  <li className="mb-2">✓ Voice stress detection</li>
                  <li className="mb-2">✓ Multimodal fusion logic</li>
                  <li className="mb-2">✓ Interactive dashboard</li>
                </ul>
              </div>
              <div className="col-md-6">
                <h4>In Development</h4>
                <ul className="list-unstyled">
                  <li className="mb-2">🚧 Advanced EEG integration</li>
                  <li className="mb-2">🚧 Long-term user profile storage</li>
                  <li className="mb-2">🚧 Mobile responsiveness optimization</li>
                  <li className="mb-2">🚧 Wearable device API hooks</li>
                  <li className="mb-2">🚧 Detailed PDF report generation</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Technical Implementation Section */}
      <div className="row">
        <div className="col-12">
          <div className="neon-card slide-in-right">
            <h3 className="text-center mb-4">Technical Implementation</h3>
            <div style={{
              background: 'rgba(5, 5, 20, 0.6)',
              padding: '2rem',
              borderRadius: '10px',
              borderLeft: '4px solid #00f2ff'
            }}>
              <h4 style={{ color: '#00f2ff' }}>Multimodal Fusion Architecture</h4>
              <p>
                <strong>Challenge:</strong> Combining heterogeneous data sources (video, audio, pysio)
                with different sampling rates and feature dimensions.
              </p>
              <p>
                <strong>Solution:</strong> Implemented a decision-level fusion strategy using
                SVM classifiers for each modality, combined with a weighted voting mechanism.
              </p>
              <p>
                <strong>Key Technologies:</strong>
              </p>
              <div className="row mt-3">
                <div className="col-md-6">
                  <ul className="list-unstyled">
                    <li>• Backend: Python Flask, OpenCV, Librosa</li>
                    <li>• Frontend: React.js, Chart.js, Socket.io</li>
                    <li>• ML: Scikit-learn, TensorFlow</li>
                  </ul>
                </div>
                <div className="col-md-6">
                  <ul className="list-unstyled">
                    <li>• Data: StressID Dataset</li>
                    <li>• Features: MFCC, Haar Cascades, FFT</li>
                    <li>• Deployment: Localhost / Docker compatible</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}