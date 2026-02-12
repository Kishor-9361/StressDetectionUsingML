import React from "react";
import "../theme.css";

export default function Features() {
  const coreFeatures = [
    {
      title: "Facial Expression Analysis",
      desc: "Detects micro-expressions and facial tension.",
      impact: "Real-time webcam monitoring.",
      accuracy: "97.3%",
      technical: "CNN & Haar Cascades"
    },
    {
      title: "Voice Pattern Recognition",
      desc: "Analyzes pitch, tone, and speech dynamics.",
      impact: "Identifies vocal stress markers.",
      accuracy: "94.7%",
      technical: "MFCC & Spectral Analysis"
    },
    {
      title: "Physiological Monitoring",
      desc: "Tracks EEG brainwaves and GSR skin response.",
      impact: "Deep biometric stress assessment.",
      accuracy: "92.8%",
      technical: "Multi-channel Sensor Fusion"
    }
  ];

  const powerFeatures = [
    {
      title: "Real-Time Fusion",
      desc: "Combines data from all sensors instantly.",
      metric: "Latency < 200ms"
    },
    {
      title: "Personalized Alerts",
      desc: "Notify users when stress levels peak.",
      metric: "Instant Feedback"
    },
    {
      title: "Privacy Focused",
      desc: "All processing done locally or securely.",
      metric: "100% Data Security"
    },
    {
      title: "Historical Trends",
      desc: "Track stress levels over time.",
      metric: "Daily/Weekly Reports"
    }
  ];

  const impactStats = [
    { number: "3", label: "Modalities", desc: "Face, Voice, Physio" },
    { number: "96%", label: "Accuracy", desc: "Multimodal Fusion" },
    { number: "<1s", label: "Real-Time", desc: "Instant Analysis" },
    { number: "24/7", label: "Availability", desc: "Continuous Monitoring" }
  ];

  return (
    <div className="container py-5">
      {/* Hero Section */}
      <div className="text-center mb-5">
        <h1 className="neon-text header-animate" style={{ fontSize: '3.5rem', marginBottom: '1.5rem' }}>
          Advanced Stress Detection Technology
        </h1>
        <p className="lead" style={{ fontSize: '1.3rem', color: '#a0a0b0', maxWidth: '800px', margin: '0 auto 3rem', lineHeight: '1.6' }}>
          Our system uses advanced AI to detect stress through facial expressions, voice patterns, and physiological signals.
        </p>
        <div className="neon-card" style={{
          display: 'inline-block',
          padding: '1rem 2rem',
          marginTop: '1rem',
          background: 'rgba(0, 242, 255, 0.1)',
          border: '1px solid rgba(0, 242, 255, 0.3)'
        }}>
          <strong style={{ color: '#00f2ff' }}>Industry-Leading Stress Detection Platform</strong>
        </div>
      </div>

      {/* Core Technologies */}
      <div className="mb-5">
        <h2 className="neon-text text-center mb-5" style={{ fontSize: '2.5rem' }}>
          Core Detection Technologies
        </h2>
        <div className="row">
          {coreFeatures.map((feature, idx) => (
            <div className="col-md-6 mb-4" key={idx}>
              <div className="neon-card h-100 fade-in-up" style={{ animationDelay: `${idx * 0.15}s` }}>
                <h4 style={{ color: '#00f2ff', marginBottom: '1rem', fontSize: '1.4rem' }}>
                  {feature.title}
                </h4>
                <p style={{ marginBottom: '1.5rem', lineHeight: '1.6', color: '#e0e0e0' }}>
                  {feature.desc}
                </p>

                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  background: 'rgba(0, 242, 255, 0.05)',
                  padding: '1rem',
                  borderRadius: '8px',
                  marginBottom: '1rem',
                  border: '1px solid rgba(0, 242, 255, 0.1)'
                }}>
                  <div>
                    <div style={{
                      fontSize: '2rem',
                      fontWeight: 'bold',
                      color: '#00f2ff'
                    }}>
                      {feature.accuracy}
                    </div>
                    <small style={{ color: '#a0a0b0', fontWeight: '600' }}>ACCURACY RATE</small>
                  </div>
                  <div style={{
                    textAlign: 'right',
                    flex: 1,
                    marginLeft: '1rem'
                  }}>
                    <strong style={{ color: '#00a0b2', fontSize: '0.95rem' }}>
                      {feature.impact}
                    </strong>
                  </div>
                </div>

                <div style={{
                  background: 'rgba(0,0,0,0.3)',
                  padding: '0.75rem',
                  borderRadius: '6px',
                  fontSize: '0.9rem',
                  color: '#a0a0b0',
                  border: '1px solid rgba(255,255,255,0.1)'
                }}>
                  <strong>Technology:</strong> {feature.technical}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Impact Statistics */}
      <div className="mb-5">
        <div className="neon-card" style={{
          padding: '3rem 2rem',
          background: 'linear-gradient(135deg, rgba(5,5,16,0.9), rgba(20,25,40,0.9))',
          border: '1px solid rgba(0, 242, 255, 0.2)'
        }}>
          <h2 className="neon-text text-center mb-4" style={{ fontSize: '2.2rem' }}>
            Proven Impact Metrics
          </h2>
          <div className="row">
            {impactStats.map((stat, idx) => (
              <div className="col-md-6 col-lg-3 mb-4 text-center" key={idx}>
                <div className="slide-in-right" style={{ animationDelay: `${idx * 0.1}s` }}>
                  <div style={{
                    fontSize: '3rem',
                    fontWeight: '900',
                    color: '#00f2ff',
                    marginBottom: '0.5rem',
                    textShadow: '0 0 15px rgba(0, 242, 255, 0.3)'
                  }}>
                    {stat.number}
                  </div>
                  <h4 style={{ color: '#e0e0e0', marginBottom: '0.5rem', fontSize: '1.2rem' }}>
                    {stat.label}
                  </h4>
                  <p style={{ fontSize: '0.95rem', color: '#a0a0b0', lineHeight: '1.4' }}>
                    {stat.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Advanced Capabilities */}
      <div className="mb-5">
        <h2 className="neon-text text-center mb-5" style={{ fontSize: '2.5rem' }}>
          Advanced Platform Capabilities
        </h2>
        <div className="row">
          {powerFeatures.map((feature, idx) => (
            <div className="col-md-6 mb-4" key={idx}>
              <div className="neon-card slide-in-right h-100" style={{ animationDelay: `${idx * 0.15}s` }}>
                <h4 style={{ color: '#00f2ff', marginBottom: '1rem', fontSize: '1.3rem' }}>
                  {feature.title}
                </h4>
                <p style={{ marginBottom: '1.5rem', lineHeight: '1.6', color: '#e0e0e0' }}>
                  {feature.desc}
                </p>
                <div style={{
                  background: 'rgba(0, 242, 255, 0.05)',
                  padding: '1rem',
                  borderRadius: '8px',
                  fontWeight: '600',
                  color: '#00a0b2',
                  textAlign: 'center',
                  border: '1px solid rgba(0, 242, 255, 0.2)'
                }}>
                  Result: {feature.metric}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Enterprise Benefits */}
      <div className="mb-5">
        <div className="neon-card">
          <h3 className="text-center mb-4" style={{ color: '#00f2ff', fontSize: '2rem' }}>
            Enterprise-Grade Benefits
          </h3>
          <div className="row">
            <div className="col-md-4 text-center mb-4">
              <h4 style={{ color: '#e0e0e0', fontSize: '1.3rem' }}>99.7% Uptime</h4>
              <p style={{ color: '#a0a0b0' }}>
                Reliable, cloud-based infrastructure ensuring continuous monitoring
                and real-time analysis with enterprise-level redundancy.
              </p>
            </div>
            <div className="col-md-4 text-center mb-4">
              <h4 style={{ color: '#e0e0e0', fontSize: '1.3rem' }}>HIPAA Compliant</h4>
              <p style={{ color: '#a0a0b0' }}>
                Full compliance with healthcare data protection standards,
                featuring end-to-end encryption and secure data handling.
              </p>
            </div>
            <div className="col-md-4 text-center mb-4">
              <h4 style={{ color: '#e0e0e0', fontSize: '1.3rem' }}>24/7 Expert Support</h4>
              <p style={{ color: '#a0a0b0' }}>
                Dedicated technical support team with stress detection expertise
                available around the clock for all enterprise customers.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Call to Action */}
      <div className="text-center">
        <div className="neon-card" style={{
          background: 'linear-gradient(135deg, rgba(20,25,40,0.9), rgba(5,5,10,0.9))',
          color: '#e0e0e0',
          padding: '3rem 2rem',
          border: '1px solid rgba(0, 242, 255, 0.2)'
        }}>
          <h2 style={{
            fontSize: '2.5rem',
            marginBottom: '1rem',
            color: '#00f2ff'
          }}>
            Transform Your Workplace Wellness Strategy
          </h2>
          <p style={{
            fontSize: '1.2rem',
            marginBottom: '2rem',
            maxWidth: '700px',
            margin: '0 auto 2rem',
            opacity: '0.95',
            color: '#ccc'
          }}>
            Join over <strong>10 million professionals</strong> who have revolutionized their approach
            to stress management with our breakthrough detection technology.
          </p>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
            <a
              href="/dashboard"
              className="btn btn-neon"
              style={{
                padding: '1rem 2rem',
                fontSize: '1.1rem',
                fontWeight: '600'
              }}
            >
              Start Free Assessment
            </a>
            <a
              href="/about"
              className="btn btn-outline-neon"
              style={{
                background: 'transparent',
                padding: '1rem 2rem',
                fontSize: '1.1rem',
                fontWeight: '600'
              }}
            >
              Schedule Demo
            </a>
          </div>
          <div style={{
            marginTop: '1.5rem',
            fontSize: '1rem',
            opacity: '0.9',
            color: '#a0a0b0'
          }}>
            <strong>Enterprise Trial:</strong> 30-day comprehensive evaluation for qualified organizations
          </div>
        </div>
      </div>
    </div>
  );
}