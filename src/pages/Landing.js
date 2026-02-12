import React from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import "../theme.css";

export default function Landing() {
  return (
    <>
      {/* Hero Section */}
      <section className="hero">
        <motion.div
          className="container text-center"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          <h1 className="neon-text">Intelligent Stress Detection</h1>
          <p className="lead">
            Real-time multimodal stress detection using Facial, Voice, and Physiological analysis.
          </p>
          <div className="btn-group">
            <Link to="/dashboard" className="btn btn-neon me-3">Start Detection</Link>
            <Link to="/features" className="btn btn-outline-neon">Learn More</Link>
          </div>
        </motion.div>
      </section>

      {/* Key Features Preview */}
      <section className="container py-5">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="neon-text text-center mb-5">Why Choose StressConnect?</h2>
          <div className="row">
            <div className="col-md-4 mb-4">
              <div className="neon-card text-center fade-in-up">
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🎯</div>
                <h4>94% Accuracy</h4>
                <p>
                  Model trained on StressID dataset achieving high validation accuracy.
                </p>
              </div>
            </div>
            <div className="col-md-4 mb-4">
              <div className="neon-card text-center fade-in-up" style={{ animationDelay: '0.2s' }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚡</div>
                <h4>Real-Time Analysis</h4>
                <p>
                  Live processing of webcam and microphone inputs for instant results.
                </p>
              </div>
            </div>
            <div className="col-md-4 mb-4">
              <div className="neon-card text-center fade-in-up" style={{ animationDelay: '0.4s' }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🔒</div>
                <h4>Privacy First</h4>
                <p>
                  Data processed locally or securely. No personal data storage.
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Technology Overview */}
      <section style={{ background: 'rgba(0, 242, 255, 0.03)', padding: '4rem 0' }}>
        <div className="container">
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="neon-text text-center mb-5">Multi-Modal Detection Technology</h2>
            <div className="row align-items-center">
              <div className="col-md-6">
                <div className="neon-card">
                  <h3>Four Integrated Detection Methods</h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div className="mb-3" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <span style={{ fontSize: '2rem' }}>📸</span>
                      <div>
                        <strong>Facial Analysis:</strong> CNN-based expression detection
                      </div>
                    </div>
                    <div className="mb-3" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <span style={{ fontSize: '2rem' }}>🎤</span>
                      <div>
                        <strong>Voice Analysis:</strong> MFCC & Spectral features
                      </div>
                    </div>
                    <div className="mb-3" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <span style={{ fontSize: '2rem' }}>⚡</span>
                      <div>
                        <strong>Physiological:</strong> EEG & GSR signal processing
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="col-md-6 text-center">
                <div style={{
                  fontSize: '8rem',
                  background: 'linear-gradient(45deg, #00f2ff, #0099cc)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  filter: 'drop-shadow(0 4px 8px rgba(0, 242, 255, 0.3))'
                }}>
                  🧬
                </div>
                <h4 style={{ color: '#00f2ff', marginTop: '1rem' }}>
                  Comprehensive Biometric Analysis
                </h4>
                <p>
                  Decision-level fusion of multiple modalities for robust stress detection.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="container py-5">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="neon-text text-center mb-5">Transform Your Workplace Wellness</h2>
          <div className="row">
            <div className="col-md-6">
              <div className="neon-card slide-in-right">
                <h3>For Individuals</h3>
                <ul className="list-unstyled">
                  <li className="mb-2">✓ Early stress detection</li>
                  <li className="mb-2">✓ Real-time monitoring</li>
                  <li className="mb-2">✓ Multimodal accuracy</li>
                  <li className="mb-2">✓ Non-invasive analysis</li>
                  <li className="mb-2">✓ Instant feedback</li>
                </ul>
              </div>
            </div>
            <div className="col-md-6">
              <div className="neon-card slide-in-right" style={{ animationDelay: '0.3s' }}>
                <h3>Technical Specs</h3>
                <ul className="list-unstyled">
                  <li className="mb-2">✓ React.js Frontend</li>
                  <li className="mb-2">✓ Flask Python Backend</li>
                  <li className="mb-2">✓ TensorFlow/Keras Models</li>
                  <li className="mb-2">✓ Socket.IO Real-time Stream</li>
                  <li className="mb-2">✓ OpenCV & Librosa Processing</li>
                </ul>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Call to Action */}
      <section style={{
        background: 'linear-gradient(135deg, #0a0a12 0%, #111120 50%, #050510 100%)',
        padding: '4rem 0',
        borderTop: '1px solid rgba(0, 242, 255, 0.1)'
      }}>
        <div className="container text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="neon-text mb-4">Ready to Transform Your Workplace?</h2>
            <p className="lead mb-4">
              Experience the power of multimodal AI stress detection.
            </p>
            <div className="btn-group">
              <Link to="/dashboard" className="btn btn-neon me-3">Try It Now</Link>
              <Link to="/about" className="btn btn-outline-neon">Learn More</Link>
            </div>
            <div style={{
              marginTop: '2rem',
              padding: '1rem',
              background: 'rgba(0, 242, 255, 0.1)',
              borderRadius: '10px',
              display: 'inline-block',
              border: '1px solid rgba(0, 242, 255, 0.2)'
            }}>
              <small style={{ color: '#00f2ff' }}>
                🎁 <strong>Free Trial:</strong> Start with a 30-day complimentary assessment for your team
              </small>
            </div>
          </motion.div>
        </div>
      </section>
    </>
  );
}