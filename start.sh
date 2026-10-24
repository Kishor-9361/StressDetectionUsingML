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

if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

echo "[1/2] Verifying Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.10+."
    exit 1
fi

if command -v fuser &> /dev/null; then
    fuser -k 5000/tcp 2>/dev/null || true
elif command -v lsof &> /dev/null; then
    lsof -ti tcp:5000 | xargs -r kill 2>/dev/null || true
fi

echo "[2/2] Starting server on http://localhost:5000..."
echo "Web UI & API available at: http://localhost:5000"
echo ""

exec python3 app.py
