import React, { useState, useRef, useEffect, useCallback } from "react";
import RealtimeMonitor from "../components/RealtimeMonitor";
import "../theme.css";
import AnalysisPanel from "../components/AnalysisPanel";
import CopilotMessage from "../components/CopilotMessage";
import GamePanel from "../components/GamePanel";
import StressChatbot from "../components/StressChatbot";
import { validateAnalysisInputs, validateAnalysisResponse } from "../utils/validateInputs";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

import { API_BASE } from "../config";

// Standard PCM WAV encoder helpers
function bufferToWav(buffer, sampleRate) {
  const bufferLength = buffer.length;
  const wavBuffer = new ArrayBuffer(44 + bufferLength * 2);
  const view = new DataView(wavBuffer);

  /* RIFF identifier */
  writeString(view, 0, 'RIFF');
  /* file length */
  view.setUint32(4, 36 + bufferLength * 2, true);
  /* RIFF type */
  writeString(view, 8, 'WAVE');
  /* format chunk identifier */
  writeString(view, 12, 'fmt ');
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw) */
  view.setUint16(20, 1, true);
  /* channel count */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, 'data');
  /* data chunk length */
  view.setUint32(40, bufferLength * 2, true);

  floatTo16BitPCM(view, 44, buffer);

  return new Blob([view], { type: 'audio/wav' });
}

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

export default function Dashboard() {
  const [mode, setMode] = useState('upload'); // 'upload' or 'realtime'
  const [serverOnline, setServerOnline] = useState(true);
  
  // Model state (Production Multimodal Expert Fusion)
  const [activeModelKey, setActiveModelKey] = useState('multimodal_expert');
  const [activeModelName, setActiveModelName] = useState('Multimodal Expert Fusion');
  const [modelSwitching, setModelSwitching] = useState(false);

  // File states
  const [faceImage, setFaceImage] = useState(null);
  const [facePreview, setFacePreview] = useState(null);
  const [voiceFile, setVoiceFile] = useState(null);
  const [voicePreviewUrl, setVoicePreviewUrl] = useState(null);
  const [eegData, setEegData] = useState("");
  const [gsrData, setGsrData] = useState("");
  const [eegFile, setEegFile] = useState(null);
  const [gsrFile, setGsrFile] = useState(null);
  
  // Graph preview states
  const [gsrPreviewData, setGsrPreviewData] = useState([]);
  const [gsrPreviewKeys, setGsrPreviewKeys] = useState([]);
  
  // Live capture in upload panel
  const [liveFaceResult, setLiveFaceResult] = useState(null);
  const [liveVoiceResult, setLiveVoiceResult] = useState(null);
  const [isMicRecording, setIsMicRecording] = useState(false);
  
  // Phase state machine
  const [phase, setPhase] = useState('idle'); // 'idle' | 'analyzing' | 'currentResult' | 'game' | 'reanalyzing' | 'comparison' | 'result'
  const [currentResult, setCurrentResult] = useState(null);
  const [previousResult, setPreviousResult] = useState(null);
  const [analysisPayload, setAnalysisPayload] = useState(null);

  // UI states
  const [error, setError] = useState(null);
  const [webcamActive, setWebcamActive] = useState(false);

  // Muse stream states
  const [museDuration, setMuseDuration] = useState(20);
  const [museFilename, setMuseFilename] = useState("uploads/eeg_session.csv");
  const [museCollecting, setMuseCollecting] = useState(false);
  const [musePoints, setMusePoints] = useState([]);
  const [museSessionError, setMuseSessionError] = useState(null);
  const [museElapsed, setMuseElapsed] = useState(0);
  
  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const micStreamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const waveformFrameRef = useRef(null);
  const micCanvasRef = useRef(null);
  const processorRef = useRef(null);

  // Background Server Health & Model check
  const fetchModelStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/model/select`);
      if (res.ok) {
        const data = await res.json();
        if (data.active_model_key) {
          setActiveModelKey(data.active_model_key);
          setActiveModelName(data.active_model_name);
        }
      }
    } catch (_e) {}
  }, []);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/health`);
        setServerOnline(response.ok);
      } catch (err) {
        setServerOnline(false);
      }
    };
    checkHealth();
    fetchModelStatus();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, [fetchModelStatus]);

  // Handler to switch model (High Spec Deep vs Low Spec RF)
  const handleModelChange = async (targetKey) => {
    if (targetKey === activeModelKey || modelSwitching) return;
    setModelSwitching(true);
    try {
      const response = await fetch(`${API_BASE}/api/model/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: targetKey })
      });
      const data = await response.json();
      if (data.status === 'success') {
        setActiveModelKey(data.active_model_key);
        setActiveModelName(data.active_model_name);
      }
    } catch (err) {
      console.error('Failed to change model:', err);
    } finally {
      setModelSwitching(false);
    }
  };

  // Scroll to top on navigation/redirect
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [mode, phase]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (voicePreviewUrl) URL.revokeObjectURL(voicePreviewUrl);
      if (facePreview) URL.revokeObjectURL(facePreview);
      stopWebcam();
      stopMicRecording();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voicePreviewUrl, facePreview]);

  useEffect(() => {
    if (!webcamActive || !videoRef.current || !streamRef.current) {
      return;
    }
    const video = videoRef.current;
    video.srcObject = streamRef.current;
    video.muted = true;
    video.playsInline = true;
    video.play().catch(() => {});
  }, [webcamActive]);

  const parseDelimitedSeries = (text, keyName = "value") => {
    const values = (text || "")
      .split(/[\s,;]+/)
      .map((item) => Number(item.trim()))
      .filter((item) => Number.isFinite(item));

    return values.slice(0, 300).map((value, index) => ({ index, [keyName]: value }));
  };

  const parseSignalCsvForPreview = (file, preferredHeaders = []) =>
    new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const text = String(reader.result || "");
          const rows = text
            .split(/\r?\n/)
            .map((row) => row.trim())
            .filter(Boolean)
            .slice(0, 1200);

          if (rows.length === 0) {
            resolve({ data: [], keys: [] });
            return;
          }

          const firstRow = rows[0].split(",").map((cell) => cell.trim());
          const hasHeader = firstRow.some((cell) => Number.isNaN(Number(cell)));
          const headers = hasHeader ? firstRow : firstRow.map((_, index) => `col_${index}`);
          const bodyRows = hasHeader ? rows.slice(1) : rows;

          const normalizedPreferred = preferredHeaders.map((item) => item.toLowerCase());
          const selectedIndexes = headers
            .map((header, index) => ({ header, index }))
            .filter(({ header }) => {
              const normalized = header.toLowerCase().replace(/\s+/g, "");
              if (normalized.includes("timestamp") || normalized === "time") return false;
              return (
                normalizedPreferred.length === 0 ||
                normalizedPreferred.includes(normalized) ||
                normalizedPreferred.includes(header.toLowerCase())
              );
            })
            .map((entry) => entry.index);

          const fallBackIndexes =
            selectedIndexes.length > 0
              ? selectedIndexes
              : headers
                  .map((header, index) => ({ header, index }))
                  .filter(({ header }) => !header.toLowerCase().includes("timestamp"))
                  .map((entry) => entry.index)
                  .slice(0, 4);

          const safeIndexes = fallBackIndexes.slice(0, 5);
          const safeKeys = safeIndexes.map((idx) => headers[idx].replace(/\s+/g, "") || `col_${idx}`);

          const points = [];
          for (let i = 0; i < bodyRows.length && points.length < 280; i += 1) {
            const cells = bodyRows[i].split(",").map((cell) => cell.trim());
            const point = { index: points.length };
            let hasAny = false;

            safeIndexes.forEach((sourceIdx, kIdx) => {
              const value = Number(cells[sourceIdx]);
              if (Number.isFinite(value)) {
                point[safeKeys[kIdx]] = value;
                hasAny = true;
              }
            });

            if (hasAny) points.push(point);
          }

          resolve({ data: points, keys: safeKeys });
        } catch (_err) {
          resolve({ data: [], keys: [] });
        }
      };

      reader.onerror = () => resolve({ data: [], keys: [] });
      reader.readAsText(file);
    });

  const stopMicRecording = (shouldAnalyze = false) => {
    if (waveformFrameRef.current) {
      cancelAnimationFrame(waveformFrameRef.current);
      waveformFrameRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop());
      micStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setIsMicRecording(false);

    if (shouldAnalyze) {
      const samples = audioChunksRef.current;
      if (samples && samples.length > 0) {
        const sampleRate = 16000;
        const wavBlob = bufferToWav(samples, sampleRate);
        const file = new File([wavBlob], "live-recording.wav", { type: "audio/wav" });
        setVoiceFile(file);
        const url = URL.createObjectURL(wavBlob);
        setVoicePreviewUrl(url);
        analyzeVoiceFile(file);
      }
    }
    audioChunksRef.current = [];
  };

  const analyzeVoiceFile = async (fileToAnalyze) => {
    try {
      const formData = new FormData();
      formData.append("file", fileToAnalyze);
      const response = await fetch(`${API_BASE}/api/voice/upload`, {
        method: "POST",
        body: formData,
      });
      const text = await response.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch (jsonErr) {
        throw new Error(`Server returned status ${response.status}: ${text.slice(0, 100) || response.statusText || "Empty response"}`);
      }
      if (response.ok && data.status === "success") {
        setLiveVoiceResult(data);
      } else {
        setError(data.message || `Live voice analysis failed (Status ${response.status}).`);
      }
    } catch (err) {
      setError(`Live voice analysis failed: ${err.message}`);
    }
  };

  const startMicRecording = async () => {
    try {
      setError(null);
      setLiveVoiceResult(null);
      audioChunksRef.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      micStreamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioCtx({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      source.connect(processor);
      processor.connect(audioContext.destination);

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioChunksRef.current.push(...inputData);
      };

      const updateWaveform = () => {
        if (!analyserRef.current || !micCanvasRef.current) return;
        const canvas = micCanvasRef.current;
        const ctx = canvas.getContext("2d");
        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
          if (!analyserRef.current || !micCanvasRef.current) return;
          waveformFrameRef.current = requestAnimationFrame(draw);

          analyserRef.current.getByteTimeDomainData(dataArray);

          const styles = getComputedStyle(canvas);
          const bgColor = styles.getPropertyValue('--chat-bg').trim() || '#070e1b';
          const themePrimary = styles.getPropertyValue('--primary-color').trim() || '#0066ff';

          ctx.fillStyle = bgColor;
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          ctx.lineWidth = 2.5;
          ctx.strokeStyle = themePrimary;
          ctx.beginPath();

          const sliceWidth = canvas.width / bufferLength;
          let x = 0;

          for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * canvas.height) / 2;

            if (i === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
            x += sliceWidth;
          }

          ctx.lineTo(canvas.width, canvas.height / 2);
          ctx.stroke();
        };

        draw();
      };
      updateWaveform();

      setIsMicRecording(true);
    } catch (err) {
      setError(`Could not start microphone recording: ${err.message}`);
      stopMicRecording(false);
    }
  };

  const analyzeLiveWebcam = async () => {
    if (!videoRef.current || !canvasRef.current) {
      setError("Webcam is not active. Please start webcam first.");
      return;
    }
    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);

      const base64Image = canvas.toDataURL("image/jpeg", 0.9);
      const response = await fetch(`${API_BASE}/api/webcam/capture`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Image }),
      });

      const data = await response.json();
      if (data.status === "success") {
        setLiveFaceResult(data);
      } else {
        setError(data.message || "Live webcam analysis failed.");
      }
    } catch (err) {
      setError(`Live webcam analysis failed: ${err.message}`);
    }
  };

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

  const handleVoiceUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setVoiceFile(file);
      setVoicePreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleGsrTextChange = (text) => {
    setGsrData(text);
    const parsed = parseDelimitedSeries(text, "gsr");
    setGsrPreviewData(parsed);
    setGsrPreviewKeys(parsed.length ? ["gsr"] : []);
  };

  const handleGsrFileUpload = async (file) => {
    setGsrFile(file);
    if (!file) {
      setGsrPreviewData([]);
      setGsrPreviewKeys([]);
      return;
    }
    const preview = await parseSignalCsvForPreview(file, ["gsr", "eda", "conductance", "skin"]);
    setGsrPreviewData(preview.data);
    setGsrPreviewKeys(preview.keys);
  };

  const startWebcam = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      });
      streamRef.current = stream;
      setWebcamActive(true);
    } catch (err) {
      setError(`Cannot access webcam: ${err.message}. Please check permissions.`);
      setWebcamActive(false);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
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
    setError(null);
    const formData = new FormData();
    if (faceImage) formData.append('face_image', faceImage);
    if (voiceFile) formData.append('voice_audio', voiceFile);
    if (eegData) formData.append('eeg_data', eegData);
    if (gsrData) formData.append('gsr_data', gsrData);
    if (eegFile) formData.append('eeg_file', eegFile);
    if (gsrFile) formData.append('gsr_file', gsrFile);

    const validationErrors = validateAnalysisInputs({
      faceFile: faceImage,
      voiceFile,
      eegData,
      gsrData,
      eegFile,
      gsrFile
    });

    if (validationErrors.length > 0) {
      setError(validationErrors.join(' '));
      return;
    }

    setPhase('analyzing');
    setAnalysisPayload(formData);
    setPreviousResult(null);
    setCurrentResult(null);

    try {
      const response = await fetch(`${API_BASE}/api/multimodal/analyze`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      if (data.status === 'error' || data.error) {
        setError(data.message || data.error || 'Analysis failed');
        setPhase('idle');
        return;
      }
      
      setCurrentResult(data);
      
      // Auto-trigger game recommendation for high/extreme stress
      if (data.stress_level === 'Extreme' || data.stress_level === 'High') {
        setPhase('result');
        setTimeout(() => {
          if (window.confirm("High stress detected. Would you like to enter PRISM's relaxation module to reduce your stress?")) {
            setPhase('game');
          }
        }, 1200);
      } else {
        setPhase('result');
      }
    } catch (err) {
      setError(err.message || 'Analysis failed');
      setPhase('idle');
    }
  };

  const handleRequestGame = useCallback(() => {
    setPhase('game');
  }, []);

  const handleGameComplete = useCallback(async () => {
    if (!analysisPayload) {
      setPhase('idle');
      return;
    }
    setPhase('reanalyzing');
    setPreviousResult(currentResult);

    try {
      const response = await fetch(`${API_BASE}/api/multimodal/analyze`, {
        method: 'POST',
        body: analysisPayload,
      });
      const data = await response.json();
      const respErrs = validateAnalysisResponse(data);
      if (respErrs.length > 0) {
        throw new Error(respErrs.join(' '));
      }
      setCurrentResult(data);
      setPhase('comparison');
    } catch (err) {
      console.error('Re-analysis failed:', err);
      setError('Re-analysis failed. Please try again.');
      setPhase('result');
    }
  }, [analysisPayload, currentResult]);

  const pollMuseStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/muse/status?limit=280`);
      const data = await response.json();
      if (data.status !== 'success') return;

      setMuseCollecting(Boolean(data.collecting));
      setMuseElapsed(Number(data.elapsed_seconds || 0));
      setMusePoints(Array.isArray(data.points) ? data.points : []);

      if (data.error) {
        setMuseSessionError(data.error);
      }
      if (data.prediction && data.prediction.status === 'success') {
        setCurrentResult(data.prediction);
      }
    } catch (err) {
      setMuseSessionError('Could not poll Muse status: ' + err.message);
    }
  };

  const startMuseCapture = async () => {
    setMuseSessionError(null);
    try {
      const response = await fetch(`${API_BASE}/api/muse/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          duration: Number(museDuration),
          filename: museFilename,
        }),
      });
      const data = await response.json();
      if (data.status === 'success') {
        setMuseCollecting(true);
        setMuseElapsed(0);
        setMusePoints([]);
      } else {
        setMuseSessionError(data.message || 'Could not start Muse recording.');
      }
    } catch (err) {
      setMuseSessionError('Could not start Muse recording: ' + err.message);
    }
  };

  const stopMuseCapture = async () => {
    try {
      await fetch(`${API_BASE}/api/muse/stop`, { method: 'POST' });
      setMuseCollecting(false);
      await pollMuseStatus();
    } catch (err) {
      setMuseSessionError('Could not stop Muse recording: ' + err.message);
    }
  };

  useEffect(() => {
    let timer = null;
    if (museCollecting) {
      timer = setInterval(() => {
        pollMuseStatus();
      }, 1000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [museCollecting]);

  const clearAll = () => {
    setFaceImage(null);
    setFacePreview(null);
    setVoiceFile(null);
    setVoicePreviewUrl(null);
    setEegData("");
    setGsrData("");
    setEegFile(null);
    setGsrFile(null);
    setCurrentResult(null);
    setPhase('idle');
    setError(null);
    setMusePoints([]);
    setMuseSessionError(null);
    setMuseElapsed(0);
    setGsrPreviewData([]);
    setGsrPreviewKeys([]);
    setLiveFaceResult(null);
    setLiveVoiceResult(null);
    stopWebcam();
    stopMicRecording(false);
  };

  return (
    <div className="App" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Header Shell */}
      <header className="prism-header">
        <div className="prism-header-inner">
          <div className="prism-header-left">
            <span className="material-symbols-outlined text-primary fs-3">psychology_alt</span>
            <div>
              <div className="prism-brand">
                <span className="text-gradient">PRISM</span>
                <span className="prism-badge">AI Intelligence</span>
              </div>
              <small className="text-muted d-none d-sm-block" style={{ fontSize: '0.72rem', letterSpacing: '0.04em' }}>
                Multimodal Biometric Stress Detection & Resilience System
              </small>
            </div>
          </div>

          <div className="prism-header-right">
            {/* System Status Pill */}
            <div className={`status-pill ${serverOnline ? 'online' : 'offline'}`}>
              <span className={`status-dot ${serverOnline ? 'online' : 'offline'}`} />
              <span>{serverOnline ? 'BACKEND ONLINE' : 'BACKEND OFFLINE'}</span>
            </div>

            {/* Active Model Indicator */}
            <div className="status-pill" style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }} title="Authoritative multimodal inference model">
              <span className="material-symbols-outlined fs-6" style={{ color: '#0054d6' }}>neurology</span>
              <span>Multimodal Expert Fusion</span>
            </div>


            {/* Backend Controls (Restart / Shutdown) */}
            <div className="d-flex gap-2">
              <button 
                title="Restart Backend Server"
                className="btn btn-secondary btn-pill p-2 d-flex align-items-center justify-content-center"
                style={{ width: '36px', height: '36px' }}
                onClick={() => {
                  if (window.confirm("Are you sure you want to restart the PRISM backend server?")) {
                    fetch(`${API_BASE}/api/restart/backend`, { method: 'POST' })
                      .then(() => alert("PRISM backend is restarting. Please wait a few seconds before analyzing."))
                      .catch(e => console.error(e));
                  }
                }}
              >
                <span className="material-symbols-outlined text-warning fs-5">refresh</span>
              </button>
              <button 
                title="Shutdown Backend Server"
                className="btn btn-secondary btn-pill p-2 d-flex align-items-center justify-content-center"
                style={{ width: '36px', height: '36px' }}
                onClick={() => {
                  if (window.confirm("Are you sure you want to shut down the PRISM backend server?")) {
                    fetch(`${API_BASE}/api/shutdown/backend`, { method: 'POST' })
                      .then(() => alert("PRISM backend is shutting down."))
                      .catch(e => console.error(e));
                  }
                }}
              >
                <span className="material-symbols-outlined text-danger fs-5">power_settings_new</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="prism-container flex-grow-1">
        {/* Navigation Tabs Bar */}
        <div className="prism-nav-bar">
          <div className="prism-tabs">
            <button
              className={`prism-tab-btn ${mode === 'upload' && phase !== 'game' ? 'active' : ''}`}
              onClick={() => {
                setMode('upload');
                setPhase(currentResult ? 'result' : 'idle');
              }}
            >
              <span className="material-symbols-outlined fs-5">folder_open</span>
              <span>Upload & Livestreams</span>
            </button>
            <button
              className={`prism-tab-btn ${mode === 'realtime' && phase !== 'game' ? 'active' : ''}`}
              onClick={() => {
                setMode('realtime');
                setPhase('idle');
              }}
            >
              <span className="material-symbols-outlined fs-5">sensors</span>
              <span>Real-Time Streaming</span>
            </button>
            <button
              className={`prism-tab-btn ${phase === 'game' ? 'active' : ''}`}
              onClick={() => setPhase('game')}
            >
              <span className="material-symbols-outlined fs-5">sports_esports</span>
              <span>Relaxation Game</span>
            </button>
          </div>

          <div className="text-muted d-flex align-items-center gap-2" style={{ fontSize: '0.8rem' }}>
            <span className="material-symbols-outlined text-primary fs-6">verified</span>
            <span>Active Model: <strong>{activeModelName}</strong></span>
          </div>
        </div>

        {/* Real-Time Streaming View */}
        {mode === 'realtime' && phase !== 'game' ? (
          <div className="animate-fade-in-up">
            <RealtimeMonitor />
          </div>
        ) : (
          <>
            {/* Error Display */}
            {error && (
              <div className="row mb-4 animate-fade-in">
                <div className="col-12">
                  <div style={{
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.35)',
                    borderRadius: '16px',
                    padding: '1rem 1.25rem',
                    color: 'var(--danger-color)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12
                  }}>
                    <span className="material-symbols-outlined fs-4">error</span>
                    <div>
                      <strong>System Notice:</strong> {error}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Loading Spinner */}
            {(phase === 'analyzing' || phase === 'reanalyzing') && (
              <div className="text-center py-5 animate-fade-in">
                <div className="spinner-border text-primary mb-3" role="status" style={{ width: '3.5rem', height: '3.5rem' }}></div>
                <h4 className="fw-bold mb-1">
                  {phase === 'reanalyzing' ? 'Re-evaluating Biometrics...' : 'Synthesizing Multimodal Stream...'}
                </h4>
                <p className="text-muted" style={{ fontSize: '0.9rem' }}>
                  Extracting face geometry, acoustic formant strain, and physiological tone through {activeModelName}
                </p>
              </div>
            )}

            {/* Results View */}
            {(phase === 'currentResult' || phase === 'comparison' || phase === 'result') && currentResult && (
              <div className="animate-fade-in-up">
                <div className="d-flex justify-content-between align-items-center mb-4 flex-wrap gap-2">
                  <button 
                    onClick={clearAll}
                    className="btn btn-secondary d-flex align-items-center gap-2"
                  >
                    <span className="material-symbols-outlined fs-5">arrow_back</span>
                    <span>Start New Analysis</span>
                  </button>
                  
                  {phase === 'comparison' && (
                    <span className="badge bg-primary-subtle text-primary border border-primary-subtle px-3 py-2 rounded-pill font-label-caps">
                      ✨ Comparison Baseline Active
                    </span>
                  )}
                </div>

                <div className="mb-4">
                  <AnalysisPanel
                    result={currentResult}
                    previousResult={phase === 'comparison' ? previousResult : null}
                    onRequestGame={handleRequestGame}
                  />
                </div>

                <div className="row">
                  <div className="col-12 mb-4">
                    <CopilotMessage 
                      stressLevel={currentResult?.stress_level} 
                      explainability={currentResult?.explainability} 
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Relaxation Game Module */}
            {phase === 'game' && (
              <div className="animate-fade-in-up">
                <GamePanel
                  stressLevel={currentResult?.stress_level}
                  onGameComplete={handleGameComplete}
                  onDismiss={() => setPhase(currentResult ? 'result' : 'idle')}
                />
              </div>
            )}

            {/* Upload & Multi-Stream Data Intake */}
            {phase === 'idle' && (
              <div className="animate-fade-in-up">
                <div className="text-center mb-5">
                  <h2 className="font-display-hero text-gradient mb-2">Multimodal Biometric Intake</h2>
                  <p className="text-muted mx-auto" style={{ maxWidth: '680px', fontSize: '0.95rem' }}>
                    Capture high-precision visual geometry, vocal acoustic strain, and physiological bio-signals for unified clinical stress assessment.
                  </p>
                </div>

                <div className="prism-grid-2col">
                  {/* Facial Modality Card */}
                  <div className="prism-card">
                    <div>
                      <div className="prism-card-header">
                        <div className="prism-icon-box" style={{ background: 'rgba(0, 102, 255, 0.12)', color: 'var(--primary-color)' }}>
                          <span className="material-symbols-outlined fs-3">face</span>
                        </div>
                        <div>
                          <h4 className="mb-0 fw-bold">Facial Biometrics</h4>
                          <small className="text-muted">468-point mesh tension & ocular dynamics</small>
                        </div>
                      </div>

                      {facePreview && (
                        <div className="position-relative mb-3 rounded-3 overflow-hidden border border-primary-subtle">
                          <img 
                            src={facePreview} 
                            alt="Facial Preview" 
                            style={{ width: '100%', maxHeight: '240px', objectFit: 'cover' }}
                          />
                          <button
                            onClick={() => {
                              setFaceImage(null);
                              setFacePreview(null);
                            }}
                            className="btn btn-danger position-absolute top-0 end-0 m-2 btn-sm rounded-pill"
                          >
                            Remove
                          </button>
                        </div>
                      )}

                      {webcamActive && (
                        <div className="mb-3">
                          <div className="position-relative rounded-3 overflow-hidden border border-primary-subtle bg-dark mb-2">
                            <video 
                              ref={videoRef} 
                              autoPlay 
                              muted
                              playsInline
                              onLoadedMetadata={() => {
                                if (videoRef.current) {
                                  videoRef.current.play().catch(() => {});
                                }
                              }}
                              style={{ width: '100%', minHeight: '260px', objectFit: 'cover' }}
                            />
                          </div>
                          <div className="d-grid gap-2">
                            <button onClick={captureWebcam} className="btn btn-primary d-flex align-items-center justify-content-center gap-2">
                              <span className="material-symbols-outlined fs-5">photo_camera</span>
                              <span>Capture Photo</span>
                            </button>
                            <button 
                              onClick={analyzeLiveWebcam} 
                              disabled={!serverOnline}
                              className="btn btn-outline-primary d-flex align-items-center justify-content-center gap-2"
                            >
                              <span className="material-symbols-outlined fs-5">bolt</span>
                              <span>{serverOnline ? 'Analyze Live Frame' : 'Server Offline'}</span>
                            </button>
                            <button onClick={stopWebcam} className="btn btn-secondary">
                              Cancel
                            </button>
                          </div>

                          {liveFaceResult && (
                            <div className="card-inner mt-3">
                              <span className="label-caps">Instant Live Frame Output</span>
                              <div className="fw-bold fs-5 text-primary">
                                {liveFaceResult.stress_level || liveFaceResult.predicted_class} ({Number(liveFaceResult.percentage || 0).toFixed(1)}%)
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {!webcamActive && !facePreview && (
                        <div className="d-grid gap-2 mt-3">
                          <label className="btn btn-outline-primary d-flex align-items-center justify-content-center gap-2 py-3 cursor-pointer">
                            <span className="material-symbols-outlined fs-5">upload_file</span>
                            <span>Select Image File</span>
                            <input
                              type="file"
                              accept="image/*"
                              onChange={handleFaceUpload}
                              style={{ display: 'none' }}
                            />
                          </label>
                          <button onClick={startWebcam} className="btn btn-secondary d-flex align-items-center justify-content-center gap-2 py-3">
                            <span className="material-symbols-outlined fs-5">videocam</span>
                            <span>Launch Webcam</span>
                          </button>
                        </div>
                      )}
                      <canvas ref={canvasRef} style={{ display: 'none' }} />
                    </div>
                  </div>

                  {/* Vocal Strain Card */}
                  <div className="prism-card">
                    <div>
                      <div className="prism-card-header">
                        <div className="prism-icon-box" style={{ background: 'rgba(104, 52, 210, 0.12)', color: 'var(--tertiary-color)' }}>
                          <span className="material-symbols-outlined fs-3">mic</span>
                        </div>
                        <div>
                          <h4 className="mb-0 fw-bold">Vocal Strain</h4>
                          <small className="text-muted">Pitch jitter, shimmer, formant & HNR acoustics</small>
                        </div>
                      </div>

                      {voicePreviewUrl && (
                        <div className="mb-3 card-inner">
                          <audio controls src={voicePreviewUrl} style={{ width: '100%' }} />
                          <button
                            onClick={() => {
                              setVoiceFile(null);
                              setVoicePreviewUrl(null);
                            }}
                            className="btn btn-danger btn-sm w-100 mt-2 rounded-pill"
                          >
                            Remove Audio
                          </button>
                        </div>
                      )}

                      {!voicePreviewUrl && (
                        <div>
                          <div className="d-flex gap-2 mb-3">
                            <button
                              type="button"
                              className="btn btn-primary flex-grow-1 d-flex align-items-center justify-content-center gap-2"
                              onClick={startMicRecording}
                              disabled={isMicRecording}
                            >
                              <span className="material-symbols-outlined fs-5">mic</span>
                              <span>{isMicRecording ? 'Recording Active...' : 'Start Microphone'}</span>
                            </button>
                            <button
                              type="button"
                              className="btn btn-outline-primary d-flex align-items-center justify-content-center gap-2"
                              onClick={() => stopMicRecording(true)}
                              disabled={!isMicRecording}
                            >
                              <span className="material-symbols-outlined fs-5">stop_circle</span>
                              <span>Stop & Analyze</span>
                            </button>
                          </div>

                          <div className="card-inner mb-3 p-2">
                            <canvas
                              ref={micCanvasRef}
                              width={320}
                              height={90}
                              style={{ width: '100%', height: '90px', display: 'block', borderRadius: 8 }}
                            />
                          </div>

                          <label className="btn btn-secondary d-flex align-items-center justify-content-center gap-2 w-100 py-3 cursor-pointer">
                            <span className="material-symbols-outlined fs-5">audio_file</span>
                            <span>Upload Audio File (WAV, MP3, OGG)</span>
                            <input
                              type="file"
                              accept="audio/*"
                              onChange={handleVoiceUpload}
                              style={{ display: 'none' }}
                            />
                          </label>

                          {liveVoiceResult && (
                            <div className="card-inner mt-3">
                              <span className="label-caps">Instant Vocal Strain Output</span>
                              <div className="fw-bold fs-5 text-tertiary">
                                {liveVoiceResult.stress_level || liveVoiceResult.predicted_class} ({Number(liveVoiceResult.percentage || 0).toFixed(1)}%)
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Physiological Tone Card */}
                  <div className="prism-card prism-grid-full">
                    <div>
                      <div className="prism-card-header">
                        <div className="prism-icon-box" style={{ background: 'rgba(0, 227, 253, 0.12)', color: 'var(--secondary-color)' }}>
                          <span className="material-symbols-outlined fs-3">ecg_heart</span>
                        </div>
                        <div>
                          <h4 className="mb-0 fw-bold">Physiological Signals</h4>
                          <small className="text-muted">Electrodermal Activity (GSR/EDA) & Autonomic Tone</small>
                        </div>
                      </div>

                      <div className="row g-3">
                        <div className="col-lg-6">
                          <label className="form-label fw-bold small">
                            GSR / EDA Values (Comma-Separated Stream)
                          </label>
                          <textarea
                            value={gsrData}
                            onChange={(e) => handleGsrTextChange(e.target.value)}
                            placeholder="e.g. 2.1, 2.3, 2.5, 2.4, 2.8, 2.9, 2.2..."
                            className="form-control"
                            rows="4"
                          />
                          <div className="mt-2">
                            <label className="btn btn-secondary btn-sm d-inline-flex align-items-center gap-2 cursor-pointer">
                              <span className="material-symbols-outlined fs-6">upload</span>
                              <span>Upload Sensor CSV / TXT</span>
                              <input
                                type="file"
                                accept=".csv,.txt"
                                onChange={(e) => handleGsrFileUpload(e.target.files[0] || null)}
                                style={{ display: 'none' }}
                              />
                            </label>
                            {gsrFile && <span className="ms-2 small text-muted">{gsrFile.name}</span>}
                          </div>
                        </div>

                        <div className="col-lg-6">
                          {gsrPreviewData.length > 0 ? (
                            <div className="card-inner h-100">
                              <span className="label-caps mb-2 d-block">Signal Preview Stream</span>
                              <div style={{ width: '100%', height: 160 }}>
                                <ResponsiveContainer>
                                  <LineChart data={gsrPreviewData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 102, 255, 0.1)" />
                                    <XAxis dataKey="index" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                                    <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                                    <Tooltip />
                                    {gsrPreviewKeys.map((key) => (
                                      <Line
                                        key={key}
                                        type="monotone"
                                        dataKey={key}
                                        dot={false}
                                        strokeWidth={2}
                                        stroke="var(--secondary-color)"
                                      />
                                    ))}
                                  </LineChart>
                                </ResponsiveContainer>
                              </div>
                            </div>
                          ) : (
                            <div className="card-inner h-100 d-flex align-items-center justify-content-center text-center p-4">
                              <div className="text-muted">
                                <span className="material-symbols-outlined fs-1 mb-2 opacity-50">show_chart</span>
                                <div className="small">Sensor trace preview will render here upon input</div>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Main Action Bar */}
                <div className="prism-actions-row">
                  <button
                    onClick={analyzeMultimodal}
                    disabled={(phase === 'analyzing' || phase === 'reanalyzing') || !serverOnline}
                    className="btn btn-primary btn-pill px-5 py-3 fs-5 d-flex align-items-center gap-2"
                  >
                    <span className="material-symbols-outlined fs-4">psychology</span>
                    <span>Analyze Stress</span>
                  </button>
                  <button
                    onClick={clearAll}
                    className="btn btn-secondary btn-pill px-4 py-3 fs-5"
                  >
                    Clear All
                  </button>
                </div>

                {/* How To Use Guide Card */}
                <div className="glass-card p-4 mt-5">
                  <h4 className="fw-bold mb-3 d-flex align-items-center gap-2">
                    <span className="material-symbols-outlined text-primary">info</span>
                    <span>How PRISM Multimodal Fusion Works</span>
                  </h4>
                  <div className="row g-4 text-muted small">
                    <div className="col-md-4">
                      <strong className="text-body d-block mb-1">📸 1. Facial Landmark Tracking</strong>
                      MediaPipe extracts 468 landmark coordinates at 30 FPS to quantify eyebrow contraction, eyelid aspect ratio velocity, and masseter jaw clenching.
                    </div>
                    <div className="col-md-4">
                      <strong className="text-body d-block mb-1">🎙️ 2. Acoustic Voice Strain</strong>
                      Speech signals undergo spectral flux analysis to track fundamental pitch (F0), cycle jitter, amplitude shimmer, and harmonics-to-noise ratio.
                    </div>
                    <div className="col-md-4">
                      <strong className="text-body d-block mb-1">⚡ 3. Unified Decision Router</strong>
                      Modalities are normalized through subject baselines and fused using your selected inference champion ({activeModelName}).
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </main>

      {/* Floating AI Copilot Assistant */}
      <StressChatbot
        stressLevel={currentResult?.stress_level || 'Moderate'}
        stressPercentage={currentResult ? currentResult.fused_score * 100 : null}
      />
    </div>
  );
}