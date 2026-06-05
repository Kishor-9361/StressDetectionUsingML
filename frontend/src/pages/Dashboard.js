// import React, { useState, useEffect } from "react";
// import "../theme.css";

// export default function Dashboard() {
//   const [faceResult, setFaceResult] = useState(null);
//   const [voiceResult, setVoiceResult] = useState(null);
//   const [eegResult, setEegResult] = useState(null);
//   const [gsrResult, setGsrResult] = useState(null);
//   const [isRecording, setIsRecording] = useState(false);
//   const [overallStress, setOverallStress] = useState(null);
//   const [voiceFile, setVoiceFile] = useState(null);
//   const [voicePreviewUrl, setVoicePreviewUrl] = useState(null);

//   const handleFaceUpload = async (e) => {
//     const file = e.target.files[0];
//     if (file) {
//       if (!file.type.startsWith('image/')) {
//         setFaceResult('⚠️ Please upload a valid image file (JPG, JPEG, PNG).');
//         return;
//       }
//       setFaceResult("Analyzing facial expressions...");
//       try {
//         const formData = new FormData();
//         formData.append('file', file);
//         const response = await fetch('http://localhost:5000/api/face/upload', {
//           method: 'POST',
//           body: formData,
//         });
//         const data = await response.json();
//         if (data.status === 'success') {
//           setFaceResult(`✅ Analysis Complete - Facial Stress: ${data.percentage}% (${data.stress_level})`);
//           calculateOverallStress();
//         } else {
//           setFaceResult(`⚠️ Error: ${data.message}`);
//         }
//       } catch (error) {
//         setFaceResult(`⚠️ Error: Network error - ${error.message}. Is the server running?`);
//       }
//     }
//   };

//   const handleVoiceRecord = async () => {
//     setIsRecording(true);
//     setVoiceResult("🎤 Recording voice sample...");
//     try {
//       const response = await fetch('http://localhost:5000/api/voice/record', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ duration: 3 }),
//       });
//       const data = await response.json();
//       setIsRecording(false);
//       if (data.status === 'success') {
//         setVoiceResult(`✅ Analysis Complete - Vocal Stress: ${data.percentage}% (${data.stress_level})`);
//         calculateOverallStress();
//       } else {
//         setVoiceResult(`⚠️ Error: ${data.message}`);
//       }
//     } catch (error) {
//       setIsRecording(false);
//       setVoiceResult(`⚠️ Error: Network error - ${error.message}. Is the server running?`);
//     }
//   };

//   const handleVoiceUpload = (e) => {
//     const file = e.target.files[0];
//     if (file) {
//       if (!file.type.startsWith('audio/')) {
//         setVoiceResult('⚠️ Please upload a valid audio file (WAV or MP3).');
//         return;
//       }
//       setVoiceFile(file);
//       setVoiceResult('✅ Audio file selected — ready to analyze');
//       const url = URL.createObjectURL(file);
//       setVoicePreviewUrl(url);
//     }
//   };

//   const handleVoiceAnalyze = async () => {
//     if (!voiceFile) {
//       setVoiceResult('⚠️ No audio file selected. Please upload a file first.');
//       return;
//     }
//     setVoiceResult('Analyzing uploaded audio sample...');
//     try {
//       const formData = new FormData();
//       formData.append('file', voiceFile);
//       const response = await fetch('http://localhost:5000/api/voice/upload', {
//         method: 'POST',
//         body: formData,
//       });
//       const data = await response.json();
//       if (data.status === 'success') {
//         setVoiceResult(`✅ Analysis Complete - Vocal Stress: ${data.percentage}% (${data.stress_level})`);
//         calculateOverallStress();
//       } else {
//         setVoiceResult(`⚠️ Error: ${data.message}`);
//       }
//     } catch (error) {
//       setVoiceResult(`⚠️ Error: Network error - ${error.message}. Is the server running?`);
//     }
//   };

//   useEffect(() => {
//     const current = voicePreviewUrl;
//     return () => {
//       if (current) {
//         URL.revokeObjectURL(current);
//       }
//     };
//   }, [voicePreviewUrl]);

//   const handleEEGUpload = (e) => {
//     const file = e.target.files[0];
//     if (file) {
//       setEegResult("Processing EEG data...");
//       setTimeout(() => {
//         setEegResult("✅ Analysis Complete - Neural Stress: 55% (Moderate-High)");
//         calculateOverallStress();
//       }, 2500);
//     }
//   };

//   const handleGSRUpload = (e) => {
//     const file = e.target.files[0];
//     if (file) {
//       setGsrResult("Analyzing GSR patterns...");
//       setTimeout(() => {
//         setGsrResult("✅ Analysis Complete - Physiological Stress: 34% (Low)");
//         calculateOverallStress();
//       }, 1500);
//     }
//   };

//   const calculateOverallStress = () => {
//     setTimeout(() => {
//       setOverallStress({
//         level: 42,
//         status: "Moderate Stress",
//         recommendation: "Consider taking short breaks and practicing deep breathing exercises."
//       });
//     }, 500);
//   };

//   const getStressColor = (level) => {
//     if (level < 30) return "#8d9740";
//     if (level < 60) return "#e4a853";
//     return "#c74545";
//   };

//   return (
//     <div className="container py-5">
//       <div className="text-center mb-5">
//         <h2 className="neon-text">Stress Detection Dashboard</h2>
//         <p className="lead">Real-time multi-modal stress analysis and monitoring</p>
//       </div>

//       {overallStress && (
//         <div className="row mb-5">
//           <div className="col-12">
//             <div className="neon-card text-center" style={{
//               background: `linear-gradient(135deg, rgba(178, 187, 95, 0.1), rgba(178, 187, 95, 0.05))`
//             }}>
//               <h3>Overall Stress Assessment</h3>
//               <div style={{
//                 fontSize: '4rem',
//                 fontWeight: 'bold',
//                 color: getStressColor(overallStress.level),
//                 textShadow: `0 0 20px ${getStressColor(overallStress.level)}40`
//               }}>
//                 {overallStress.level}%
//               </div>
//               <h4 style={{color: getStressColor(overallStress.level)}}>
//                 {overallStress.status}
//               </h4>
//               <p style={{
//                 background: 'rgba(178, 187, 95, 0.1)',
//                 padding: '1rem',
//                 borderRadius: '8px',
//                 marginTop: '1rem'
//               }}>
//                 <strong>Recommendation:</strong> {overallStress.recommendation}
//               </p>
//             </div>
//           </div>
//         </div>
//       )}

//       <div className="row">
//         <div className="col-md-6 mb-4">
//           <div className="neon-card">
//             <div className="text-center mb-3">
//               <span style={{fontSize: '3rem'}}>📸</span>
//             </div>
//             <h4>Facial Stress Detection</h4>
//             <p>Upload a photo or use your webcam to analyze facial expressions for stress indicators.</p>
//             <div className="mb-3">
//               <input 
//                 type="file" 
//                 onChange={handleFaceUpload} 
//                 className="form-control mb-2"
//                 accept="image/*"
//               />
//               <button className="btn btn-neon w-100" onClick={() => alert('Webcam feature coming soon!')}>
//                 📹 Open Webcam
//               </button>
//             </div>
//             {faceResult && (
//               <div style={{
//                 background: 'rgba(178, 187, 95, 0.1)',
//                 padding: '1rem',
//                 borderRadius: '8px',
//                 marginTop: '1rem'
//               }}>
//                 <strong>Result:</strong> {faceResult}
//               </div>
//             )}
//           </div>
//         </div>

//         <div className="col-md-6 mb-4">
//           <div className="neon-card">
//             <div className="text-center mb-3">
//               <span style={{fontSize: '3rem'}}>🎤</span>
//             </div>
//             <h4>Voice Stress Detection</h4>
//             <p>Record a voice sample or upload an audio file to analyze vocal patterns for stress markers.</p>
//             <div className="mb-3">
//               <button 
//                 className={`btn ${isRecording ? 'btn-outline-neon' : 'btn-neon'} w-100`}
//                 onClick={handleVoiceRecord}
//                 disabled={isRecording}
//               >
//                 {isRecording ? '🔴 Recording...' : '🎙️ Start Recording'}
//               </button>
//             </div>
//             <div className="mb-3">
//               <label className="form-label">Or upload an audio file</label>
//               <input
//                 type="file"
//                 accept="audio/*"
//                 onChange={handleVoiceUpload}
//                 className="form-control mb-2"
//               />
//               {voicePreviewUrl && (
//                 <div style={{marginBottom: '0.75rem'}}>
//                   <audio controls src={voicePreviewUrl} style={{width: '100%'}} />
//                 </div>
//               )}
//               <button className="btn btn-outline-neon w-100" onClick={handleVoiceAnalyze}>
//                 Analyze Uploaded Audio
//               </button>
//             </div>
//             {voiceResult && (
//               <div style={{
//                 background: 'rgba(178, 187, 95, 0.1)',
//                 padding: '1rem',
//                 borderRadius: '8px',
//                 marginTop: '1rem'
//               }}>
//                 <strong>Result:</strong> {voiceResult}
//               </div>
//             )}
//           </div>
//         </div>

//         <div className="col-md-6 mb-4">
//           <div className="neon-card">
//             <div className="text-center mb-3">
//               <span style={{fontSize: '3rem'}}>🧠</span>
//             </div>
//             <h4>EEG Data Analysis</h4>
//             <p>Upload EEG data files to analyze brainwave patterns and neurological stress indicators.</p>
//             <div className="mb-3">
//               <input 
//                 type="file" 
//                 onChange={handleEEGUpload}
//                 className="form-control"
//                 accept=".edf,.csv,.txt"
//               />
//               <small style={{color: '#556022'}}>
//                 Supported formats: EDF, CSV, TXT
//               </small>
//             </div>
//             {eegResult && (
//               <div style={{
//                 background: 'rgba(178, 187, 95, 0.1)',
//                 padding: '1rem',
//                 borderRadius: '8px',
//                 marginTop: '1rem'
//               }}>
//                 <strong>Result:</strong> {eegResult}
//               </div>
//             )}
//           </div>
//         </div>

//         <div className="col-md-6 mb-4">
//           <div className="neon-card">
//             <div className="text-center mb-3">
//               <span style={{fontSize: '3rem'}}>⚡</span>
//             </div>
//             <h4>GSR Data Analysis</h4>
//             <p>Monitor Galvanic Skin Response data to evaluate physiological stress responses.</p>
//             <div className="mb-3">
//               <input 
//                 type="file" 
//                 onChange={handleGSRUpload}
//                 className="form-control"
//                 accept=".csv,.txt,.json"
//               />
//               <small style={{color: '#556022'}}>
//                 Supported formats: CSV, TXT, JSON
//               </small>
//             </div>
//             {gsrResult && (
//               <div style={{
//                 background: 'rgba(178, 187, 95, 0.1)',
//                 padding: '1rem',
//                 borderRadius: '8px',
//                 marginTop: '1rem'
//               }}>
//                 <strong>Result:</strong> {gsrResult}
//               </div>
//             )}
//           </div>
//         </div>
//       </div>

//       <div className="row mt-5">
//         <div className="col-12">
//           <div className="neon-card">
//             <h3 className="text-center mb-4">How to Use the Dashboard</h3>
//             <div className="row">
//               <div className="col-md-6">
//                 <h4>Getting Started</h4>
//                 <ul className="list-unstyled">
//                   <li>1. Choose one or more detection methods</li>
//                   <li>2. Upload your data or use real-time capture</li>
//                   <li>3. Wait for analysis to complete</li>
//                   <li>4. Review your stress assessment results</li>
//                   <li>5. Follow personalized recommendations</li>
//                 </ul>
//               </div>
//               <div className="col-md-6">
//                 <h4>Best Practices</h4>
//                 <ul className="list-unstyled">
//                   <li>• Use multiple detection methods for accuracy</li>
//                   <li>• Take measurements in consistent environments</li>
//                   <li>• Regular monitoring provides better insights</li>
//                   <li>• Follow up on high stress readings</li>
//                   <li>• Share results with wellness coordinators</li>
//                 </ul>
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }


import React, { useState, useRef, useEffect } from "react";
import RealtimeMonitor from "../components/RealtimeMonitor";
import "../theme.css";

export default function Dashboard() {
  const [mode, setMode] = useState('upload'); // 'upload' or 'realtime'
  // File states
  const [faceImage, setFaceImage] = useState(null);
  const [facePreview, setFacePreview] = useState(null);
  const [voiceFile, setVoiceFile] = useState(null);
  const [voicePreviewUrl, setVoicePreviewUrl] = useState(null);
  const [eegData, setEegData] = useState("");
  const [gsrData, setGsrData] = useState("");

  // UI states
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [webcamActive, setWebcamActive] = useState(false);

  // Refs for webcam
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (voicePreviewUrl) URL.revokeObjectURL(voicePreviewUrl);
      if (facePreview) URL.revokeObjectURL(facePreview);
      stopWebcam();
    };
  }, [voicePreviewUrl, facePreview]);

  const handleFaceUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please upload a valid image file (JPG, JPEG, PNG)');
        return;
      }
      setFaceImage(file);
      setFacePreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleVoiceUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('audio/')) {
        setError('Please upload a valid audio file (WAV, MP3)');
        return;
      }
      setVoiceFile(file);
      setVoicePreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setWebcamActive(true);
      }
    } catch (err) {
      setError('Could not access webcam: ' + err.message);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setWebcamActive(false);
  };

  const captureWebcam = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      canvas.toBlob((blob) => {
        const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });
        setFaceImage(file);
        setFacePreview(URL.createObjectURL(file));
        stopWebcam();
      }, 'image/jpeg');
    }
  };

  const analyzeMultimodal = async () => {
    if (!faceImage && !voiceFile && !eegData && !gsrData) {
      setError('Please provide at least one input (image, audio, EEG, or GSR data)');
      return;
    }

    setAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();

      if (faceImage) formData.append('face_image', faceImage);
      if (voiceFile) formData.append('voice_audio', voiceFile);
      if (eegData) formData.append('eeg_data', eegData);
      if (gsrData) formData.append('gsr_data', gsrData);

      const response = await fetch('http://127.0.0.1:5000/api/multimodal/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.status === 'success') {
        setResult(data);
      } else {
        setError(data.message || 'Analysis failed');
      }
    } catch (err) {
      setError('Network error: ' + err.message + '. Is the server running on port 5000?');
    } finally {
      setAnalyzing(false);
    }
  };

  const clearAll = () => {
    setFaceImage(null);
    setFacePreview(null);
    setVoiceFile(null);
    setVoicePreviewUrl(null);
    setEegData("");
    setGsrData("");
    setResult(null);
    setError(null);
    stopWebcam();
  };

  const getStressColor = (percentage) => {
    if (percentage < 30) return "#8d9740";
    if (percentage < 60) return "#e4a853";
    return "#c74545";
  };

  const getRecommendation = (level) => {
    switch (level) {
      case "Low":
        return "You're doing well! Maintain your current stress management practices.";
      case "Moderate":
        return "Consider taking short breaks and practicing deep breathing exercises.";
      case "High":
        return "High stress detected. Consider speaking with a wellness professional and taking immediate breaks.";
      default:
        return "Continue monitoring your stress levels regularly.";
    }
  };

  return (
    <div className="container py-5">
      <div className="text-center mb-5">
        <h2 className="neon-text">Multimodal Stress Detection</h2>
        <p className="lead">Intelligent stress analysis using facial, vocal, and physiological indicators</p>

        <div className="btn-group mt-3">
          <button
            className={`btn ${mode === 'upload' ? 'btn-neon' : 'btn-outline-neon'}`}
            onClick={() => setMode('upload')}
          >
            📂 Upload Mode
          </button>
          <button
            className={`btn ${mode === 'realtime' ? 'btn-neon' : 'btn-outline-neon'}`}
            onClick={() => setMode('realtime')}
          >
            🔴 Real-Time Mode
          </button>
        </div>
      </div>

      {mode === 'realtime' ? (
        <RealtimeMonitor />
      ) : (
        <>

          {/* Error Display */}
          {error && (
            <div className="row mb-4">
              <div className="col-12">
                <div style={{
                  background: 'rgba(199, 69, 69, 0.1)',
                  border: '2px solid #c74545',
                  borderRadius: '8px',
                  padding: '1rem',
                  color: '#c74545'
                }}>
                  <strong>⚠️ Error:</strong> {error}
                </div>
              </div>
            </div>
          )}

          {/* Result Display */}
          {result && (
            <div className="row mb-5">
              <div className="col-12">
                <div className="neon-card text-center" style={{
                  background: `linear-gradient(135deg, rgba(178, 187, 95, 0.1), rgba(178, 187, 95, 0.05))`
                }}>
                  <h3>Overall Stress Assessment</h3>
                  <div style={{
                    fontSize: '4rem',
                    fontWeight: 'bold',
                    color: getStressColor(result.percentage),
                    textShadow: `0 0 20px ${getStressColor(result.percentage)}40`
                  }}>
                    {result.percentage.toFixed(1)}%
                  </div>
                  <h4 style={{ color: getStressColor(result.percentage) }}>
                    {result.stress_level} Stress - {result.predicted_class}
                  </h4>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    gap: '1rem',
                    margin: '1.5rem 0',
                    flexWrap: 'wrap'
                  }}>
                    <div style={{
                      background: 'rgba(178, 187, 95, 0.15)',
                      padding: '0.75rem 1.5rem',
                      borderRadius: '8px'
                    }}>
                      <small style={{ color: '#556022' }}>Confidence</small>
                      <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>
                        {(result.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div style={{
                      background: 'rgba(178, 187, 95, 0.15)',
                      padding: '0.75rem 1.5rem',
                      borderRadius: '8px'
                    }}>
                      <small style={{ color: '#556022' }}>Stress Probability</small>
                      <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>
                        {(result.stress_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div style={{
                      background: 'rgba(178, 187, 95, 0.15)',
                      padding: '0.75rem 1.5rem',
                      borderRadius: '8px'
                    }}>
                      <small style={{ color: '#556022' }}>No Stress Probability</small>
                      <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>
                        {(result.no_stress_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* Individual Modality Results */}
                  {result.individual_predictions && (
                    <div className="row mt-4">
                      <div className="col-12">
                        <h5 style={{ marginBottom: '1rem', color: '#556022' }}>Individual Modality Analysis</h5>
                        <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
                          {result.individual_predictions.facial !== null && (
                            <div style={{
                              background: 'rgba(178, 187, 95, 0.2)',
                              padding: '1rem',
                              borderRadius: '8px',
                              minWidth: '150px'
                            }}>
                              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📸</div>
                              <small style={{ color: '#556022' }}>Facial</small>
                              <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
                                {(result.individual_predictions.facial * 100).toFixed(1)}%
                              </div>
                            </div>
                          )}
                          {result.individual_predictions.voice !== null && (
                            <div style={{
                              background: 'rgba(178, 187, 95, 0.2)',
                              padding: '1rem',
                              borderRadius: '8px',
                              minWidth: '150px'
                            }}>
                              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🎤</div>
                              <small style={{ color: '#556022' }}>Voice</small>
                              <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
                                {(result.individual_predictions.voice * 100).toFixed(1)}%
                              </div>
                            </div>
                          )}
                          {result.individual_predictions.physiological !== null && (
                            <div style={{
                              background: 'rgba(178, 187, 95, 0.2)',
                              padding: '1rem',
                              borderRadius: '8px',
                              minWidth: '150px'
                            }}>
                              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>⚡</div>
                              <small style={{ color: '#556022' }}>Physiological</small>
                              <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
                                {(result.individual_predictions.physiological * 100).toFixed(1)}%
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  <p style={{
                    background: 'rgba(178, 187, 95, 0.1)',
                    padding: '1rem',
                    borderRadius: '8px',
                    marginTop: '1.5rem'
                  }}>
                    <strong>Recommendation:</strong> {getRecommendation(result.stress_level)}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Input Section */}
          <div className="row">
            <div className="col-12">
              <div className="neon-card">
                <h3 className="text-center mb-4">Provide Your Data</h3>
                <p className="text-center" style={{ color: '#556022', marginBottom: '2rem' }}>
                  Upload any combination of facial images, voice recordings, or physiological data for comprehensive stress analysis
                </p>

                <div className="row">
                  {/* Facial Input */}
                  <div className="col-md-6 mb-4">
                    <div style={{
                      border: '2px dashed rgba(178, 187, 95, 0.3)',
                      borderRadius: '12px',
                      padding: '1.5rem',
                      background: 'rgba(178, 187, 95, 0.05)'
                    }}>
                      <div className="text-center mb-3">
                        <span style={{ fontSize: '3rem' }}>📸</span>
                        <h4>Facial Analysis</h4>
                        <p style={{ color: '#556022', fontSize: '0.9rem' }}>
                          Upload a photo or use webcam
                        </p>
                      </div>

                      {facePreview && (
                        <div style={{ marginBottom: '1rem', position: 'relative' }}>
                          <img
                            src={facePreview}
                            alt="Preview"
                            style={{
                              width: '100%',
                              borderRadius: '8px',
                              maxHeight: '200px',
                              objectFit: 'cover'
                            }}
                          />
                          <button
                            onClick={() => {
                              setFaceImage(null);
                              setFacePreview(null);
                            }}
                            className="btn btn-danger"
                            style={{
                              position: 'absolute',
                              top: '8px',
                              right: '8px',
                              padding: '0.25rem 0.5rem',
                              fontSize: '0.875rem'
                            }}
                          >
                            Remove
                          </button>
                        </div>
                      )}

                      {webcamActive && (
                        <div style={{ marginBottom: '1rem' }}>
                          <video
                            ref={videoRef}
                            autoPlay
                            style={{ width: '100%', borderRadius: '8px' }}
                          />
                          <button
                            onClick={captureWebcam}
                            className="btn btn-neon w-100 mt-2"
                          >
                            📷 Capture Photo
                          </button>
                          <button
                            onClick={stopWebcam}
                            className="btn btn-outline-neon w-100 mt-2"
                          >
                            Cancel
                          </button>
                        </div>
                      )}

                      {!webcamActive && !facePreview && (
                        <>
                          <input
                            type="file"
                            accept="image/*"
                            onChange={handleFaceUpload}
                            className="form-control mb-2"
                          />
                          <button
                            onClick={startWebcam}
                            className="btn btn-outline-neon w-100"
                          >
                            📹 Use Webcam
                          </button>
                        </>
                      )}
                      <canvas ref={canvasRef} style={{ display: 'none' }} />
                    </div>
                  </div>

                  {/* Voice Input */}
                  <div className="col-md-6 mb-4">
                    <div style={{
                      border: '2px dashed rgba(178, 187, 95, 0.3)',
                      borderRadius: '12px',
                      padding: '1.5rem',
                      background: 'rgba(178, 187, 95, 0.05)'
                    }}>
                      <div className="text-center mb-3">
                        <span style={{ fontSize: '3rem' }}>🎤</span>
                        <h4>Voice Analysis</h4>
                        <p style={{ color: '#556022', fontSize: '0.9rem' }}>
                          Upload an audio recording
                        </p>
                      </div>

                      {voicePreviewUrl && (
                        <div style={{ marginBottom: '1rem' }}>
                          <audio
                            controls
                            src={voicePreviewUrl}
                            style={{ width: '100%', marginBottom: '0.5rem' }}
                          />
                          <button
                            onClick={() => {
                              setVoiceFile(null);
                              setVoicePreviewUrl(null);
                            }}
                            className="btn btn-danger w-100"
                            style={{ fontSize: '0.875rem' }}
                          >
                            Remove Audio
                          </button>
                        </div>
                      )}

                      {!voicePreviewUrl && (
                        <>
                          <input
                            type="file"
                            accept="audio/*"
                            onChange={handleVoiceUpload}
                            className="form-control"
                          />
                          <small style={{ color: '#556022', display: 'block', marginTop: '0.5rem' }}>
                            Supported: WAV, MP3, OGG
                          </small>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Physiological Input */}
                  <div className="col-12 mb-4">
                    <div style={{
                      border: '2px dashed rgba(178, 187, 95, 0.3)',
                      borderRadius: '12px',
                      padding: '1.5rem',
                      background: 'rgba(178, 187, 95, 0.05)'
                    }}>
                      <div className="text-center mb-3">
                        <span style={{ fontSize: '3rem' }}>🧠⚡</span>
                        <h4>Physiological Data</h4>
                        <p style={{ color: '#556022', fontSize: '0.9rem' }}>
                          Enter EEG and GSR data as comma-separated values
                        </p>
                      </div>

                      <div className="row">
                        <div className="col-md-6 mb-3">
                          <label className="form-label">
                            <strong>🧠 EEG Data</strong>
                          </label>
                          <textarea
                            value={eegData}
                            onChange={(e) => setEegData(e.target.value)}
                            placeholder="e.g., 0.5, 0.7, 0.6, 0.8, 0.65, 0.72..."
                            className="form-control"
                            rows="3"
                          />
                          <small style={{ color: '#556022' }}>
                            Enter brainwave measurement values
                          </small>
                        </div>

                        <div className="col-md-6 mb-3">
                          <label className="form-label">
                            <strong>⚡ GSR Data</strong>
                          </label>
                          <textarea
                            value={gsrData}
                            onChange={(e) => setGsrData(e.target.value)}
                            placeholder="e.g., 2.1, 2.3, 2.5, 2.4, 2.6, 2.2..."
                            className="form-control"
                            rows="3"
                          />
                          <small style={{ color: '#556022' }}>
                            Enter skin conductance values
                          </small>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="text-center mt-4">
                  <button
                    onClick={analyzeMultimodal}
                    disabled={analyzing}
                    className="btn btn-neon"
                    style={{
                      padding: '0.75rem 3rem',
                      fontSize: '1.1rem',
                      marginRight: '1rem'
                    }}
                  >
                    {analyzing ? '⏳ Analyzing...' : '🔍 Analyze Stress Level'}
                  </button>
                  <button
                    onClick={clearAll}
                    className="btn btn-outline-neon"
                    style={{
                      padding: '0.75rem 2rem',
                      fontSize: '1.1rem'
                    }}
                  >
                    🗑️ Clear All
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* How to Use Section */}
          <div className="row mt-5">
            <div className="col-12">
              <div className="neon-card">
                <h3 className="text-center mb-4">How to Use the Dashboard</h3>
                <div className="row">
                  <div className="col-md-4">
                    <h4 style={{ color: '#b2bb5f' }}>📸 Facial Data</h4>
                    <ul className="list-unstyled">
                      <li>• Upload a clear photo of your face</li>
                      <li>• Or use webcam for live capture</li>
                      <li>• Ensure good lighting</li>
                      <li>• Look directly at camera</li>
                    </ul>
                  </div>
                  <div className="col-md-4">
                    <h4 style={{ color: '#b2bb5f' }}>🎤 Voice Data</h4>
                    <ul className="list-unstyled">
                      <li>• Upload a voice recording</li>
                      <li>• Speak naturally for 3-5 seconds</li>
                      <li>• Minimize background noise</li>
                      <li>• Use standard audio formats</li>
                    </ul>
                  </div>
                  <div className="col-md-4">
                    <h4 style={{ color: '#b2bb5f' }}>⚡ Physiological Data</h4>
                    <ul className="list-unstyled">
                      <li>• Enter comma-separated values</li>
                      <li>• EEG: Brainwave measurements</li>
                      <li>• GSR: Skin conductance values</li>
                      <li>• Use sensor device outputs</li>
                    </ul>
                  </div>
                </div>
                <div className="alert" style={{
                  background: 'rgba(178, 187, 95, 0.1)',
                  border: '1px solid rgba(178, 187, 95, 0.3)',
                  marginTop: '1.5rem',
                  textAlign: 'center'
                }}>
                  <strong>💡 Pro Tip:</strong> For best results, provide multiple data sources.
                  The system uses advanced multimodal fusion to combine insights from all available inputs.
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}