// Web Worker — runs in its own OS thread
// Receives face indicator objects from main thread
// Sends them to the Flask backend without blocking MediaPipe or the UI

let pendingPost = false;  // rate limiter — one POST in flight at a time

self.onmessage = async function(e) {
    const { indicators, timestamp, calibrationMode, userId } = e.data;
    
    // Drop this frame if a POST is already in flight
    // This prevents queue buildup when server is slow
    if (pendingPost) return;
    
    pendingPost = true;
    try {
        const isDevPort = self.location && self.location.origin && self.location.origin.includes(':3000');
        const origin = isDevPort ? 'http://127.0.0.1:5000' : ((self.location && self.location.origin && self.location.origin !== 'null') ? self.location.origin : 'http://127.0.0.1:5000');
        const endpoint = calibrationMode ? '/api/calibrate/face_sample' : '/api/stream/face';
        const url = `${origin}${endpoint}`;
            
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ indicators, timestamp, user_id: userId || 'default' }),
        });
        const data = await response.json();
        // Send result back to main thread
        self.postMessage({ type: 'face_result', data });
    } catch (err) {
        self.postMessage({ type: 'error', error: err.message });
    } finally {
        pendingPost = false;
    }
};
