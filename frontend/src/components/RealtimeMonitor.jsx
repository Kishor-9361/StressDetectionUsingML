import React, { useEffect, useState, useRef } from 'react';
import FaceStream from './FaceStream';
import WaveformRecorder from './WaveformRecorder';
import CalibrationWizard from './CalibrationWizard';

const LEVEL_COLOR = { 
  Low: '#00f2ff',       // Cyan glow
  Moderate: '#FF9800',  // Orange
  High: '#F44336'       // Red
};

function IndicatorBar({ label, value, scale = 1.0, format }) {
  const percent = Math.min(100, Math.max(0, (value / scale) * 100));
  const barColor = percent > 75 ? '#F44336' : percent > 40 ? '#FF9800' : '#00f2ff';
  
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-color)' }}>
        <span>{label}</span>
        <span style={{ fontWeight: 'bold', fontFamily: 'monospace', color: barColor }}>{format(value)}</span>
      </div>
      <div style={{ width: '100%', height: 6, background: 'rgba(255, 255, 255, 0.05)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ 
          width: `${percent}%`, 
          height: '100%', 
          background: barColor, 
          borderRadius: 3, 
          transition: 'width 0.3s ease-out, background-color 0.3s ease' 
        }} />
      </div>
    </div>
  );
}

export default function RealtimeMonitor() {
  const [active, setActive] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [faceScore, setFaceScore] = useState(null);
  const [voiceScore, setVoiceScore] = useState(null);
  const [faceIndicators, setFaceIndicators] = useState(null);
  const [voiceIndicators, setVoiceIndicators] = useState(null);
  const esRef = useRef(null);

  // Calibration states
  const [calibrationPhase, setCalibrationPhase] = useState('idle');
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const silenceRmsRef = useRef([]);

  // Smooth UI display values
  const [smoothFusedScore, setSmoothFusedScore] = useState(0);
  const [smoothFaceScore, setSmoothFaceScore] = useState(null);
  const [smoothVoiceScore, setSmoothVoiceScore] = useState(null);

  // Easing/smoothing loops for high-responsiveness display updates
  useEffect(() => {
    if (!active) {
      setSmoothFusedScore(0);
      return;
    }
    const target = result && result.fused_score !== undefined ? result.fused_score * 100 : 0;
    const interval = setInterval(() => {
      setSmoothFusedScore(prev => {
        const diff = target - prev;
        if (Math.abs(diff) < 0.5) return target;
        return prev + diff * 0.15; // Smooth exponential step
      });
    }, 50);
    return () => clearInterval(interval);
  }, [result, active]);

  useEffect(() => {
    if (!active || faceScore === null) {
      setSmoothFaceScore(null);
      return;
    }
    const target = faceScore;
    const interval = setInterval(() => {
      setSmoothFaceScore(prev => {
        if (prev === null) return target;
        const diff = target - prev;
        if (Math.abs(diff) < 0.01) return target;
        return prev + diff * 0.15;
      });
    }, 50);
    return () => clearInterval(interval);
  }, [faceScore, active]);

  useEffect(() => {
    if (!active || voiceScore === null) {
      setSmoothVoiceScore(null);
      return;
    }
    const target = voiceScore;
    const interval = setInterval(() => {
      setSmoothVoiceScore(prev => {
        if (prev === null) return target;
        const diff = target - prev;
        if (Math.abs(diff) < 0.01) return target;
        return prev + diff * 0.15;
      });
    }, 50);
    return () => clearInterval(interval);
  }, [voiceScore, active]);

  const connectSSE = () => {
    const es = new EventSource('http://127.0.0.1:5000/api/stream/fused');
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.status === 'active') {
          setResult(data);
          setHistory(h => [
            ...h.slice(-29), 
            {
              t: new Date().toLocaleTimeString(),
              score: Math.round(data.fused_score * 100),
            }
          ]);
        } else if (data.status === 'waiting') {
          setResult(data);
        }
      } catch (err) {
        console.error("SSE parse error: ", err);
      }
    };
    es.onerror = () => {
      console.error("SSE Connection failed.");
      es.close();
    };
    esRef.current = es;
  };

  const startMonitoring = () => {
    setActive(true);
    setResult(null);
    setHistory([]);
    setFaceScore(null);
    setVoiceScore(null);
    setFaceIndicators(null);
    setVoiceIndicators(null);
    setSmoothFusedScore(0);
    setSmoothFaceScore(null);
    setSmoothVoiceScore(null);

    if (!isCalibrated) {
      setCalibrating(true);
      return; // Guided CalibrationWizard will trigger connectSSE on Complete
    }

    connectSSE();
  };

  const handleCalibrationComplete = (calibration) => {
    setIsCalibrated(true);
    setCalibrating(false);
    setCalibrationPhase('idle');
    console.log("[Calibration] Guided baseline loaded successfully: ", calibration);
    
    // Start active monitoring session
    connectSSE();
  };

  const resetCalibration = () => {
    setIsCalibrated(false);
    stopMonitoring();
  };

  const stopMonitoring = () => {
    setActive(false);
    setCalibrating(false);
    setCalibrationPhase('idle');
    setFaceScore(null);
    setVoiceScore(null);
    setFaceIndicators(null);
    setVoiceIndicators(null);
    setSmoothFusedScore(0);
    setSmoothFaceScore(null);
    setSmoothVoiceScore(null);
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
  };

  const handleVoiceChunk = async (blob) => {
    try {
      const response = await fetch('http://127.0.0.1:5000/api/stream/voice?user_id=default', {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: blob,
      });
      const data = await response.json();
      
      if (data && data.indicators !== undefined) {
        setVoiceIndicators(data.indicators);
        
        if (calibrationPhase === 'silence') {
          if (data.indicators.voice_intensity !== undefined) {
            silenceRmsRef.current.push(data.indicators.voice_intensity);
            console.log("[Calibration] Added silence RMS sample:", data.indicators.voice_intensity);
          }
        } else if (calibrationPhase === 'voice') {
          await fetch('http://127.0.0.1:5000/api/calibrate/voice_sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'default', indicators: data.indicators }),
          });
          console.log("[Calibration] Posted voice sample indicators.");
        }
      }
      
      // Update vocal stress level only in normal monitoring mode
      if (calibrationPhase === 'idle' && !calibrating && data && data.score !== undefined) {
        setVoiceScore(data.score);
      }
    } catch (err) {
      console.error("Failed to POST voice chunk: ", err);
    }
  };

  useEffect(() => {
    return () => {
      if (esRef.current) esRef.current.close();
    };
  }, []);

  const displayLevel = smoothFusedScore > 70 ? 'High' : smoothFusedScore > 40 ? 'Moderate' : 'Low';
  const levelColor = active && !calibrating ? LEVEL_COLOR[displayLevel] : 'var(--text-muted)';
  const stressPercent = Math.round(smoothFusedScore);

  return (
    <div className="neon-card fade-in-up" style={{ padding: 24, marginTop: 16 }}>
      <h3 className="text-center mb-4 neon-text" style={{ fontSize: '1.8rem' }}>🧠 Real-Time Multimodal Monitoring</h3>
      
      {/* Controls & Overall Fused Assessment */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16, marginBottom: 24, borderBottom: '1px solid rgba(0, 242, 255, 0.1)', paddingBottom: 20 }}>
        <div>
          {!active ? (
            <button className="btn btn-neon" onClick={startMonitoring} style={{ fontSize: '1.05rem', padding: '12px 28px' }}>
              ▶ Start Session
            </button>
          ) : (
            <div style={{ display: 'flex', gap: 12 }}>
              <button className="btn" onClick={stopMonitoring} style={{ background: '#D32F2F', color: '#fff', fontSize: '1.05rem', padding: '12px 28px', boxShadow: '0 4px 15px rgba(211, 47, 47, 0.3)' }}>
                ⏹ Stop Session
              </button>
              {isCalibrated && (
                <button className="btn" onClick={resetCalibration} style={{ background: 'transparent', border: '1px solid rgba(0,242,255,0.4)', color: '#00f2ff', fontSize: '1.05rem', padding: '12px 24px' }}>
                  🔄 Recalibrate
                </button>
              )}
            </div>
          )}
        </div>

        {active && !calibrating && result && result.status === 'active' && (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 16,
            background: 'rgba(20, 25, 45, 0.8)',
            padding: '10px 24px',
            borderRadius: 12,
            border: `1.5px solid ${levelColor}`,
            boxShadow: `0 0 15px ${levelColor}22`
          }}>
            <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Fused Multimodal Stress Level:</span>
            <span style={{ 
              fontWeight: 800, 
              color: levelColor, 
              fontSize: '1.4rem', 
              textShadow: `0 0 10px ${levelColor}44` 
            }}>
              {displayLevel} ({stressPercent}%)
            </span>
          </div>
        )}
      </div>

      {/* Guided Calibration Wizard Container */}
      {active && calibrating && (
        <div style={{ marginBottom: 28 }} className="fade-in-up">
          <CalibrationWizard 
            userId="default"
            silenceRmsRef={silenceRmsRef}
            onPhaseChange={(phase) => setCalibrationPhase(phase)}
            onComplete={handleCalibrationComplete}
          />
        </div>
      )}

      {/* Main Grid: Face (Modality 1) & Voice (Modality 2) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 24, marginBottom: 24 }}>
        
        {/* Face Card */}
        <div style={{ 
          background: 'rgba(10, 15, 30, 0.4)',
          borderRadius: 16,
          padding: 20,
          border: '1px solid rgba(0, 242, 255, 0.1)',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
          display: 'flex',
          flexDirection: 'column',
          gap: 16
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(0, 242, 255, 0.1)', paddingBottom: 10 }}>
            <h4 style={{ margin: 0, fontSize: '1.1rem', color: 'var(--text-color)' }}>📸 Facial Modality</h4>
            {active && !calibrating && (
              <span style={{ 
                fontWeight: 700, 
                color: smoothFaceScore === null ? 'var(--text-muted)' : smoothFaceScore > 0.7 ? '#F44336' : smoothFaceScore > 0.4 ? '#FF9800' : '#00f2ff',
                fontSize: '0.9rem',
                padding: '4px 10px',
                borderRadius: 6,
                background: 'rgba(20, 25, 45, 0.8)',
                border: `1px solid ${smoothFaceScore === null ? 'rgba(255,255,255,0.1)' : smoothFaceScore > 0.7 ? '#F4433633' : smoothFaceScore > 0.4 ? '#FF980033' : '#00f2ff33'}`
              }}>
                {smoothFaceScore === null ? 'No Face Detected' : `Stress: ${Math.round(smoothFaceScore * 100)}%`}
              </span>
            )}
            {active && calibrating && (
              <span style={{ fontWeight: 600, color: '#FF9800', fontSize: '0.85rem' }}>
                {calibrationPhase === 'face' ? 'RECORDING BASELINE...' : 'CAMERA PREVIEW'}
              </span>
            )}
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <FaceStream 
              active={active} 
              calibrationMode={calibrationPhase === 'face'}
              userId="default"
              onResult={(data) => {
                if (!calibrating && data && data.score !== undefined) {
                  setFaceScore(data.score);
                }
              }} 
              onIndicatorsUpdate={(indicators) => {
                setFaceIndicators(indicators);
                if (indicators === null) {
                  setFaceScore(null);
                }
              }}
            />
          </div>

          <div style={{ background: 'rgba(5, 5, 16, 0.4)', borderRadius: 10, padding: 14, border: '1px solid rgba(0, 242, 255, 0.08)', flexGrow: 1 }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 12 }}>Live Tracked Facial Patterns</div>
            {active && faceIndicators ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <IndicatorBar label="Jaw Displacement (Mouth Open)" value={faceIndicators.jaw_displacement} scale={2.0} format={(v) => `${(v).toFixed(2)}`} />
                <IndicatorBar label="Jaw Width (Masseter Clench)" value={faceIndicators.jaw_angle_width} scale={2.5} format={(v) => `${Math.round(v * 100)}%`} />
                <IndicatorBar label="Blink Velocity (Stress Blink)" value={faceIndicators.blink_velocity} scale={5.0} format={(v) => `${(v).toFixed(2)} /s`} />
                <IndicatorBar label="Brow Tension (Contraction)" value={faceIndicators.forehead_tension} scale={0.5} format={(v) => `${Math.round(v * 100)}%`} />
                <IndicatorBar label="Lip Compression" value={faceIndicators.lip_compression} scale={0.5} format={(v) => `${Math.round(v * 100)}%`} />
                <IndicatorBar label="Head Tilt / Movement" value={faceIndicators.head_tilt} scale={30.0} format={(v) => `${(v).toFixed(1)}°`} />
              </div>
            ) : (
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', textAlign: 'center', padding: '20px 0' }}>
                {active ? (faceIndicators === null ? "No face detected in video frame" : "Extracting landmarks...") : "Start session to enable camera tracking"}
              </div>
            )}
          </div>
        </div>

        {/* Voice Card */}
        <div style={{ 
          background: 'rgba(10, 15, 30, 0.4)',
          borderRadius: 16,
          padding: 20,
          border: '1px solid rgba(0, 242, 255, 0.1)',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
          display: 'flex',
          flexDirection: 'column',
          gap: 16
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(0, 242, 255, 0.1)', paddingBottom: 10 }}>
            <h4 style={{ margin: 0, fontSize: '1.1rem', color: 'var(--text-color)' }}>🎤 Vocal Modality</h4>
            {active && !calibrating && (
              <span style={{ 
                fontWeight: 700, 
                color: smoothVoiceScore === null ? 'var(--text-muted)' : smoothVoiceScore > 0.7 ? '#F44336' : smoothVoiceScore > 0.4 ? '#FF9800' : '#00f2ff',
                fontSize: '0.9rem',
                padding: '4px 10px',
                borderRadius: 6,
                background: 'rgba(20, 25, 45, 0.8)',
                border: `1px solid ${smoothVoiceScore === null ? 'rgba(255,255,255,0.1)' : smoothVoiceScore > 0.7 ? '#F4433633' : smoothVoiceScore > 0.4 ? '#FF980033' : '#00f2ff33'}`
              }}>
                {smoothVoiceScore === null ? 'Silent / Ambient' : `Stress: ${Math.round(smoothVoiceScore * 100)}%`}
              </span>
            )}
            {active && calibrating && (
              <span style={{ fontWeight: 600, color: '#FF9800', fontSize: '0.85rem' }}>
                {calibrationPhase === 'silence' ? 'RECORDING SILENCE...' : calibrationPhase === 'voice' ? 'RECORDING VOICE...' : 'MICROPHONE PREVIEW'}
              </span>
            )}
          </div>
          
          <WaveformRecorder
            continuous={active}
            chunkIntervalMs={1000}
            onChunk={handleVoiceChunk}
            voiceScore={smoothVoiceScore}
          />

          <div style={{ background: 'rgba(5, 5, 16, 0.4)', borderRadius: 10, padding: 14, border: '1px solid rgba(0, 242, 255, 0.08)', flexGrow: 1 }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 12 }}>Acoustic Stress Biomarkers</div>
            {active && voiceIndicators ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <IndicatorBar label="Pitch (F0 Mean)" value={voiceIndicators.f0_mean} scale={300.0} format={(v) => `${Math.round(v)} Hz`} />
                <IndicatorBar label="Jitter (Micro-instability)" value={voiceIndicators.jitter_percent} scale={5.0} format={(v) => `${(v).toFixed(2)}%`} />
                <IndicatorBar label="Shimmer (Amplitude Var)" value={voiceIndicators.shimmer_db} scale={3.0} format={(v) => `${(v).toFixed(2)} dB`} />
                <IndicatorBar label="Speaking Rate (ZCR)" value={voiceIndicators.speaking_rate_proxy} scale={0.2} format={(v) => `${(v * 100).toFixed(0)}%`} />
                <IndicatorBar label="Voice Intensity (RMS)" value={voiceIndicators.voice_intensity} scale={0.1} format={(v) => `${(v * 1000).toFixed(0)}`} />
              </div>
            ) : (
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', textAlign: 'center', padding: '20px 0' }}>
                {active ? "Waiting for voice buffer..." : "Start session to enable microphone"}
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Dynamic Reliability and Weights panel */}
      {active && !calibrating && result && result.status === 'active' && result.weights && (
        <div style={{ background: 'rgba(10, 12, 22, 0.5)', padding: 16, borderRadius: 12, border: '1px solid rgba(0, 242, 255, 0.08)', marginBottom: 24 }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 10 }}>Dynamic Sensor Reliability Weights</div>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            {Object.entries(result.weights).map(([mode, weight]) => (
              <div key={mode} style={{ flex: '1 1 150px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.8rem', padding: '6px 12px', background: 'rgba(20, 25, 45, 0.4)', borderRadius: 8, border: '1px solid rgba(0, 242, 255, 0.05)' }}>
                <span style={{ textTransform: 'capitalize', color: 'var(--text-color)' }}>{mode} sensor weight:</span>
                <span style={{ fontWeight: 700, color: '#00f2ff' }}>{Math.round(weight * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Mini trend chart — last 30 readings */}
      {active && !calibrating && history.length > 1 && (
        <div style={{ background: 'rgba(5, 5, 16, 0.6)', padding: 16, borderRadius: 12, border: '1px solid rgba(0, 242, 255, 0.1)' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 8, fontWeight: 500 }}>Live Session Stress Trend Timeline</div>
          <svg width="100%" height={80} style={{ display: 'block', overflow: 'visible' }}>
            <g>
              {history.map((h, i) => {
                const x = (i / (history.length - 1)) * 100;
                const y = 75 - (h.score / 100) * 60;
                return i === 0 ? null : (
                  <line key={i}
                    x1={`${((i - 1) / (history.length - 1)) * 100}%`}
                    y1={75 - (history[i - 1].score / 100) * 60}
                    x2={`${x}%`} 
                    y2={y}
                    stroke={h.score > 70 ? LEVEL_COLOR.High : h.score > 40 ? LEVEL_COLOR.Moderate : LEVEL_COLOR.Low}
                    strokeWidth={3.5}
                    strokeLinecap="round"
                  />
                );
              })}
              {/* Highlight current point */}
              {history.length > 0 && (
                <circle
                  cx="100%"
                  cy={75 - (history[history.length - 1].score / 100) * 60}
                  r={5}
                  fill={history[history.length - 1].score > 70 ? LEVEL_COLOR.High : history[history.length - 1].score > 40 ? LEVEL_COLOR.Moderate : LEVEL_COLOR.Low}
                  style={{ filter: 'drop-shadow(0px 0px 4px rgba(0,242,255,0.6))' }}
                />
              )}
            </g>
          </svg>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 8 }}>
            <span>Session Start</span>
            <span>Now</span>
          </div>
        </div>
      )}

      {active && !calibrating && (!result || result.status === 'waiting') && (
        <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--text-muted)' }} className="fade-in-up">
          <div className="spinner-border text-info mb-2" role="status" style={{ width: '1.5rem', height: '1.5rem' }}></div>
          <p style={{ fontStyle: 'italic', margin: 0, fontSize: '0.85rem' }}>
            Waiting for face and voice feed telemetry...
          </p>
        </div>
      )}

      {!active && (
        <p className="text-center" style={{ color: 'var(--text-muted)', margin: 0, fontStyle: 'italic', fontSize: '0.9rem' }}>
          Session inactive. Click "Start Session" to begin real-time calibration and diagnostics.
        </p>
      )}
    </div>
  );
}
