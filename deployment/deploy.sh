#!/bin/bash
# Quick deployment script for Ubuntu server
# Usage: ./deploy.sh user@192.168.0.146

set -e

SERVER="$1"
APP_DIR="/opt/layerpainter"

if [ -z "$SERVER" ]; then
    echo "Usage: $0 user@192.168.0.146"
    exit 1
fi

echo "Deploying LayerPainter to $SERVER..."

# Transfer files (excluding node_modules, venv, .next)
echo "Transferring files..."
rsync -avz --exclude 'node_modules' \
           --exclude 'venv' \
           --exclude '.next' \
           --exclude '__pycache__' \
           --exclude '*.pyc' \
           --exclude '.git' \
           "$(dirname "$0")/../" \
           "$SERVER:$APP_DIR/"

# Setup backend
echo "Setting up backend..."
ssh "$SERVER" << 'ENDSSH'
cd /opt/layerpainter/backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt
ENDSSH

# Setup frontend
echo "Setting up frontend..."
ssh "$SERVER" << 'ENDSSH'
cd /opt/layerpainter/frontend

echo "Installing dependencies..."
npm install

echo "Building production bundle..."
export NEXT_PUBLIC_API_BASE_URL=http://192.168.0.146:8000
npm run build
ENDSSH

# Setup systemd services
echo "Setting up systemd services..."
ssh "$SERVER" << 'ENDSSH'
sudo cp /opt/layerpainter/deployment/backend.service /etc/systemd/system/
sudo cp /opt/layerpainter/deployment/frontend.service /etc/systemd/system/

# Set permissions (adjust user if needed)
sudo chown -R www-data:www-data /opt/layerpainter || sudo chown -R $USER:$USER /opt/layerpainter

sudo systemctl daemon-reload
sudo systemctl enable backend.service
sudo systemctl enable frontend.service
sudo systemctl restart backend.service
sudo systemctl restart frontend.service
ENDSSH

echo "Deployment complete!"
echo "Backend: http://192.168.0.146:8000"
echo "Frontend: http://192.168.0.146:3000"
echo ""
echo "Check status with:"
echo "  ssh $SERVER 'sudo systemctl status backend.service'"
echo "  ssh $SERVER 'sudo systemctl status frontend.service'"
