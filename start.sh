#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  Multimodal Stress Detection - Production Launcher"
echo "========================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/backend"

export PORT=5000
export FRONTEND_URL="*"

echo "[1/2] Verifying Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.10+."
    exit 1
fi

echo "[2/2] Starting server on http://localhost:5000..."
echo "Web UI & API available at: http://localhost:5000"
echo ""

if command -v gunicorn &> /dev/null; then
    exec gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app
else
    exec python3 app.py
fi
