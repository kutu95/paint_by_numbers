# Quick Deployment Guide

Deploy LayerPainter to Ubuntu server at `192.168.0.146`.

## Quick Steps

### 1. Transfer Files to Server

```bash
# From your local machine, transfer files
rsync -avz --exclude 'node_modules' --exclude 'venv' --exclude '.next' \
  "/Users/bowskill/Documents/Paint by Numbers/" \
  user@192.168.0.146:/opt/layerpainter/
```

### 2. SSH into Server and Setup

```bash
ssh user@192.168.0.146
```

### 3. Setup Backend

```bash
cd /opt/layerpainter/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Setup Frontend

```bash
cd /opt/layerpainter/frontend
npm install
export NEXT_PUBLIC_API_BASE_URL=http://192.168.0.146:8000
npm run build
```

### 5. Setup systemd Services

```bash
# Copy service files
sudo cp /opt/layerpainter/deployment/backend.service /etc/systemd/system/
sudo cp /opt/layerpainter/deployment/frontend.service /etc/systemd/system/

# Set permissions (use www-data or your user)
sudo chown -R www-data:www-data /opt/layerpainter

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable backend.service frontend.service
sudo systemctl start backend.service frontend.service

# Check status
sudo systemctl status backend.service
sudo systemctl status frontend.service
```

### 6. Open Firewall Ports (if needed)

```bash
sudo ufw allow 8000/tcp  # Backend
sudo ufw allow 3000/tcp  # Frontend
```

### 7. Access the App

- **Frontend**: http://192.168.0.146:3000
- **Backend API Docs**: http://192.168.0.146:8000/docs

## Troubleshooting

Check logs:
```bash
sudo journalctl -u backend.service -f
sudo journalctl -u frontend.service -f
```

Restart services:
```bash
sudo systemctl restart backend.service
sudo systemctl restart frontend.service
```

For detailed instructions, see [deployment/DEPLOYMENT.md](deployment/DEPLOYMENT.md)
