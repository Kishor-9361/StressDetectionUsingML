
import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import { useNavigate } from 'react-router-dom';

const StreamAnalyzer = () => {
    // Socket
    const socketRef = useRef(null);
    const navigate = useNavigate();
    const [status, setStatus] = useState('Disconnected');

    // Media Refs
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const overlayRef = useRef(null); // New: For bounding boxes
    const audioCanvasRef = useRef(null); // New: For audio graph
    const audioContextRef = useRef(null);
    const scriptProcessorRef = useRef(null);
    const audioStreamRef = useRef(null);

    // State
    const [isVideoActive, setIsVideoActive] = useState(false);
    const [isAudioActive, setIsAudioActive] = useState(false);
    const [sensitivity, setSensitivity] = useState(0.5);
    const [logs, setLogs] = useState([]); // New: Logging

    // Results
    const [stressResult, setStressResult] = useState(null);
    const [lastUpdate, setLastUpdate] = useState(null);

    // Config
    const VIDEO_FPS = 5;
    const AUDIO_BUFFER_SIZE = 4096;

    const addLog = (msg) => {
        setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev].slice(0, 10));
    };

    useEffect(() => {
        // Connect to Socket.IO
        socketRef.current = io('http://localhost:5000');

        socketRef.current.on('connect', () => {
            setStatus('Connected');
            addLog("Connected to server");
        });

        socketRef.current.on('disconnect', () => {
            setStatus('Disconnected');
            addLog("Disconnected from server");
        });

        socketRef.current.on('status', (data) => {
            addLog(`Server: ${data.msg}`);
        });

        socketRef.current.on('stress_update', (data) => {
            setLastUpdate(new Date());
            setStressResult(prev => ({
                ...prev,
                [data.type]: data.result
            }));

            // Log significant events
            if (data.result.predicted_class === 'Stress') {
                // only log stress to avoid spam
                // addLog(`${data.type.toUpperCase()} Stress Detected: ${data.result.stress_level}`);
            }
        });

        return () => {
            stopVideo();
            stopAudio();
            if (socketRef.current) socketRef.current.disconnect();
        };
    }, []);

    // ---------------------------
    // Draw Bounding Box (Video)
    // ---------------------------
    useEffect(() => {
        if (!stressResult?.video?.face_box || !overlayRef.current) {
            // Clear canvas if no box
            if (overlayRef.current) {
                const ctx = overlayRef.current.getContext('2d');
                ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
            }
            return;
        }

        const ctx = overlayRef.current.getContext('2d');
        const [x, y, w, h] = stressResult.video.face_box;
        const isStress = stressResult.video.predicted_class === 'Stress';
        const color = isStress ? '#f44336' : '#4caf50';

        // Clear and Draw
        ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // Label
        ctx.fillStyle = color;
        ctx.font = '16px Arial';
        ctx.fillText(isStress ? "Stress" : "Normal", x, y - 5);

    }, [stressResult?.video]);

    // ---------------------------
    // Video Handling
    // ---------------------------
    const startVideo = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                setIsVideoActive(true);
                addLog("Camera started");

                // Start sending frames
                const interval = setInterval(() => {
                    sendVideoFrame();
                }, 1000 / VIDEO_FPS);

                videoRef.current.frameInterval = interval;
            }
        } catch (err) {
            console.error("Error accessing webcam:", err);
            alert("Could not access webcam");
        }
    };

    const stopVideo = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            videoRef.current.srcObject = null;
            clearInterval(videoRef.current.frameInterval);
        }
        setIsVideoActive(false);
        addLog("Camera stopped");
        // Clear overlay
        if (overlayRef.current) {
            const ctx = overlayRef.current.getContext('2d');
            ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
        }
    };

    const sendVideoFrame = () => {
        if (videoRef.current && canvasRef.current && socketRef.current) {
            const context = canvasRef.current.getContext('2d');

            // Sync dimensions
            if (canvasRef.current.width !== videoRef.current.videoWidth) {
                canvasRef.current.width = videoRef.current.videoWidth;
                canvasRef.current.height = videoRef.current.videoHeight;
                if (overlayRef.current) {
                    overlayRef.current.width = videoRef.current.videoWidth;
                    overlayRef.current.height = videoRef.current.videoHeight;
                }
            }

            context.drawImage(videoRef.current, 0, 0);

            const dataUrl = canvasRef.current.toDataURL('image/jpeg', 0.7);
            socketRef.current.emit('stream_video', {
                image: dataUrl,
                sensitivity: sensitivity
            });
        }
    };

    // ---------------------------
    // Audio Handling
    // ---------------------------
    const startAudio = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioStreamRef.current = stream;
            setIsAudioActive(true);
            addLog("Microphone started");

            if (!audioContextRef.current) {
                audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
            }
            const audioContext = audioContextRef.current;

            const processor = audioContext.createScriptProcessor(AUDIO_BUFFER_SIZE, 1, 1);
            const source = audioContext.createMediaStreamSource(stream);

            source.connect(processor);
            processor.connect(audioContext.destination);

            scriptProcessorRef.current = { processor, source };

            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                const audioArray = Array.from(inputData); // For sending

                // Draw Waveform
                drawWaveform(inputData);

                // Send to backend
                if (socketRef.current) {
                    const maxAmp = audioArray.reduce((acc, val) => Math.max(acc, Math.abs(val)), 0);
                    if (maxAmp > 0.01) {
                        socketRef.current.emit('stream_audio', {
                            audio: audioArray,
                            sr: audioContext.sampleRate,
                            sensitivity: sensitivity
                        });
                    }
                }
            };

        } catch (err) {
            console.error("Error accessing microphone:", err);
            alert("Could not access microphone");
        }
    };

    const drawWaveform = (data) => {
        if (!audioCanvasRef.current) return;
        const canvas = audioCanvasRef.current;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.fillStyle = '#222';
        ctx.fillRect(0, 0, width, height);

        ctx.lineWidth = 2;
        ctx.strokeStyle = '#4caf50';
        ctx.beginPath();

        const sliceWidth = width * 1.0 / data.length;
        let x = 0;

        for (let i = 0; i < data.length; i++) {
            const v = data[i] * 200.0; // Scale amplitude
            const y = height / 2 + v; // Center

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);

            x += sliceWidth;
        }

        ctx.stroke();
    };

    const stopAudio = () => {
        if (scriptProcessorRef.current) {
            const { processor, source } = scriptProcessorRef.current;
            if (source) source.disconnect();
            if (processor) processor.disconnect();
            scriptProcessorRef.current = null;
        }
        if (audioStreamRef.current) {
            audioStreamRef.current.getTracks().forEach(track => track.stop());
            audioStreamRef.current = null;
        }
        setIsAudioActive(false);
        addLog("Microphone stopped");
    };

    // ---------------------------
    // Render
    // ---------------------------

    const getResultDisplay = (type) => {
        if (!stressResult || !stressResult[type]) return "Waiting for data...";
        const data = stressResult[type];
        if (data.error) return `Error: ${data.error}`;
        if (data.status === 'buffering') return "Buffering audio...";
        return `${data.predicted_class} (${data.percentage.toFixed(1)}%) - ${data.stress_level}`;
    };

    const getStressColor = (type) => {
        if (!stressResult || !stressResult[type] || stressResult[type].error || stressResult[type].status === 'buffering') return '#666';
        return stressResult[type].predicted_class === 'Stress' ? '#f44336' : '#4caf50';
    };

    return (
        <div className="neon-card mt-4">
            <h3 className="text-center mb-4">🔴 Live Real-Time Analysis</h3>

            <div className="text-center mb-3">
                <span className={`badge ${status === 'Connected' ? 'bg-success' : 'bg-danger'} me-2`}>
                    Server: {status}
                </span>


            </div>

            <div className="row">
                {/* Video Column */}
                <div className="col-md-6 mb-4">
                    <div style={{
                        border: '2px solid rgba(0, 242, 255, 0.3)',
                        borderRadius: '12px',
                        padding: '1rem',
                        background: 'rgba(5, 5, 16, 0.8)'
                    }}>
                        <div className="d-flex justify-content-between align-items-center mb-3">
                            <h5 className="text-white mb-0">🎥 Video Stream</h5>
                            <span className="badge" style={{
                                background: 'rgba(0, 242, 255, 0.1)',
                                color: '#00f2ff',
                                border: '1px solid rgba(0, 242, 255, 0.3)'
                            }}>
                                Model Acc: 97.3%
                            </span>
                        </div>
                        <div style={{ position: 'relative', minHeight: '240px', background: '#222', borderRadius: '8px', overflow: 'hidden' }}>
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                muted
                                style={{ width: '100%', height: '100%', objectFit: 'cover', display: isVideoActive ? 'block' : 'none' }}
                            />
                            {/* Overlay Canvas for Bounding Box */}
                            <canvas
                                ref={overlayRef}
                                style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: '100%',
                                    pointerEvents: 'none',
                                    display: isVideoActive ? 'block' : 'none'
                                }}
                            />

                            {!isVideoActive && (
                                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: '#666' }}>
                                    Camera Off
                                </div>
                            )}
                            <canvas ref={canvasRef} style={{ display: 'none' }} />
                        </div>

                        <div className="mt-3 d-grid gap-2">
                            {!isVideoActive ? (
                                <button className="btn btn-neon" onClick={startVideo}>Start Camera</button>
                            ) : (
                                <>
                                    <button className="btn btn-outline-light me-3" onClick={() => navigate(0)} style={{ margin: '0 10px' }}>
                                        ← Reset
                                    </button>
                                    <button
                                        className="btn btn-danger"
                                        onClick={stopVideo}
                                        style={{ margin: '0 10px' }}
                                    >
                                        Stop Camera
                                    </button>
                                </>
                            )}
                        </div>

                        <div className="mt-3 text-center">
                            <h5 style={{ color: getStressColor('video') }}>
                                {getResultDisplay('video')}
                            </h5>
                        </div>
                    </div>
                </div>

                {/* Audio Column */}
                <div className="col-md-6 mb-4">
                    <div style={{
                        border: '2px solid rgba(0, 242, 255, 0.3)',
                        borderRadius: '12px',
                        padding: '1rem',
                        background: 'rgba(5, 5, 16, 0.8)'
                    }}>
                        <div className="d-flex justify-content-between align-items-center mb-3">
                            <h5 className="text-white mb-0">🎙️ Audio Stream</h5>
                            <span className="badge" style={{
                                background: 'rgba(0, 242, 255, 0.1)',
                                color: '#00f2ff',
                                border: '1px solid rgba(0, 242, 255, 0.3)'
                            }}>
                                Model Acc: 94.7%
                            </span>
                        </div>

                        {/* Audio Waveform Canvas */}
                        <div style={{
                            height: '100px',
                            background: '#222',
                            borderRadius: '8px',
                            marginBottom: '1rem',
                            border: isAudioActive ? '2px solid #4caf50' : 'none',
                            overflow: 'hidden'
                        }}>
                            <canvas
                                ref={audioCanvasRef}
                                width={400}
                                height={100}
                                style={{ width: '100%', height: '100%' }}
                            />
                        </div>

                        <div className="d-grid gap-2">
                            {!isAudioActive ? (
                                <button className="btn btn-neon" onClick={startAudio}>Start Microphone</button>
                            ) : (
                                <button className="btn btn-outline-danger" onClick={stopAudio}>Stop Microphone</button>
                            )}
                        </div>

                        <div className="mt-3 text-center">
                            <h5 style={{ color: getStressColor('audio') }}>
                                {getResultDisplay('audio')}
                            </h5>
                        </div>
                    </div>
                </div>
            </div>

            {/* Logs Section */}
            <div className="card bg-dark text-white mt-3" style={{ fontSize: '0.85rem' }}>
                <div className="card-header border-bottom border-secondary">
                    📜 System Logs & Timing
                </div>
                <div className="card-body p-2" style={{ maxHeight: '150px', overflowY: 'auto', fontFamily: 'monospace' }}>
                    {logs.length === 0 ? <span className="text-muted">No logs yet...</span> :
                        logs.map((log, i) => <div key={i}>{log}</div>)
                    }
                </div>
            </div>

            <div className="text-center text-muted mt-2" style={{ fontSize: '0.8rem' }}>
                Live updates every ~1s for audio and ~200ms for video. Requires backend running.
            </div>
        </div>
    );
};

export default StreamAnalyzer;
