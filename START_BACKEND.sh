#!/bin/bash
# Script to start the backend server for local development

cd "$(dirname "$0")/backend"

# Activate virtual environment
source venv/bin/activate

# Ensure dependencies are installed (idempotent)
pip install -q -r requirements.txt

# Start the backend server (use python3 for macOS compatibility)
echo "Starting backend server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

python3 main.py
