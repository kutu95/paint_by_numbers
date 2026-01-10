#!/bin/bash
# Script to start the backend server for local development

cd "$(dirname "$0")/backend"

# Activate virtual environment
source venv/bin/activate

# Start the backend server
echo "Starting backend server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

python main.py
