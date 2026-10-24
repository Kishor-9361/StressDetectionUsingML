import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app import app, socketio

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Multimodal Stress Detection server on 0.0.0.0:{port}...")
    socketio.run(app, debug=False, host='0.0.0.0', port=port, use_reloader=False)
