# Deployment Guide for Ubuntu Server

This guide will help you deploy LayerPainter to an Ubuntu server at IP `192.168.0.146`.

## Prerequisites

On your Ubuntu server, install:
- Python 3.8+ and pip
- Node.js 18+ and npm
- systemd (usually pre-installed on Ubuntu)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv

# Install Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installations
python3 --version
node --version
npm --version
```

## Step 1: Transfer Files to Server

Transfer the entire project to your server. You can use `scp` or `rsync`:

```bash
# From your local machine
rsync -avz --exclude 'node_modules' --exclude 'venv' --exclude '.next' \
  "/Users/bowskill/Documents/Paint by Numbers/" \
  user@192.168.0.146:/opt/layerpainter/
```

Or manually copy files via SFTP/SCP to `/opt/layerpainter/` on the server.

## Step 2: Setup Backend

SSH into your server and run:

```bash
cd /opt/layerpainter/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test run (optional, to verify it works)
python main.py
# Press Ctrl+C after verifying it starts
```

## Step 3: Setup Frontend

In a new terminal on the server:

```bash
cd /opt/layerpainter/frontend

# Install dependencies
npm install

# Set environment variable for API base URL
export NEXT_PUBLIC_API_BASE_URL=http://192.168.0.146:8000

# Build the production bundle
npm run build

# Test run (optional)
npm start
# Press Ctrl+C after verifying it works
```

## Step 4: Configure systemd Services

### Copy service files

```bash
# Copy service files to systemd directory
sudo cp /opt/layerpainter/deployment/backend.service /etc/systemd/system/
sudo cp /opt/layerpainter/deployment/frontend.service /etc/systemd/system/
```

### Edit service files (if needed)

You may need to adjust paths, user, or IP address in the service files:

```bash
sudo nano /etc/systemd/system/backend.service
sudo nano /etc/systemd/system/frontend.service
```

Key things to check:
- `WorkingDirectory` should match where you deployed the app
- `CORS_ORIGINS` should include your server IP and any other clients
- `NEXT_PUBLIC_API_BASE_URL` should match your server IP
- `User` can be changed to your preferred user (must have permissions to the app directory)

### Set proper permissions

```bash
# Make sure www-data (or your user) owns the files
sudo chown -R www-data:www-data /opt/layerpainter

# Or if using a different user:
sudo chown -R $USER:$USER /opt/layerpainter
```

### Reload systemd and start services

```bash
# Reload systemd to recognize new services
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable backend.service
sudo systemctl enable frontend.service

# Start the services
sudo systemctl start backend.service
sudo systemctl start frontend.service

# Check status
sudo systemctl status backend.service
sudo systemctl status frontend.service
```

## Step 5: Configure Firewall (if enabled)

If you have `ufw` firewall enabled:

```bash
# Allow backend API (port 8000)
sudo ufw allow 8000/tcp

# Allow frontend (port 3000)
sudo ufw allow 3000/tcp

# Check status
sudo ufw status
```

## Step 6: Verify Deployment

1. **Check backend**: Open `http://192.168.0.146:8000/docs` in your browser (FastAPI auto-generated docs)
2. **Check frontend**: Open `http://192.168.0.146:3000` in your browser
3. **Check logs**:
   ```bash
   sudo journalctl -u backend.service -f
   sudo journalctl -u frontend.service -f
   ```

## Useful Commands

### View logs
```bash
# Backend logs
sudo journalctl -u backend.service -n 50
sudo journalctl -u backend.service -f

# Frontend logs
sudo journalctl -u frontend.service -n 50
sudo journalctl -u frontend.service -f
```

### Restart services
```bash
sudo systemctl restart backend.service
sudo systemctl restart frontend.service
```

### Stop services
```bash
sudo systemctl stop backend.service
sudo systemctl stop frontend.service
```

### Check service status
```bash
sudo systemctl status backend.service
sudo systemctl status frontend.service
```

## Troubleshooting

### Backend not starting
- Check logs: `sudo journalctl -u backend.service -n 50`
- Verify virtual environment is activated in service file
- Check Python path is correct
- Verify dependencies are installed: `source venv/bin/activate && pip list`

### Frontend not starting
- Check logs: `sudo journalctl -u frontend.service -n 50`
- Verify Node.js version: `node --version` (should be 18+)
- Rebuild: `cd /opt/layerpainter/frontend && npm run build`
- Check environment variable is set correctly

### CORS errors in browser
- Verify `CORS_ORIGINS` in backend service includes your client IP
- Restart backend after changing CORS settings: `sudo systemctl restart backend.service`

### Port already in use
- Check what's using the port: `sudo lsof -i :8000` or `sudo lsof -i :3000`
- Kill the process or change ports in service files

### Permission denied errors
- Check file ownership: `ls -la /opt/layerpainter`
- Fix ownership: `sudo chown -R www-data:www-data /opt/layerpainter`
- Or change user in service files to match your user

## Optional: Using Nginx as Reverse Proxy

For production, you may want to use Nginx as a reverse proxy on port 80:

1. Install Nginx: `sudo apt install nginx`
2. Create config file `/etc/nginx/sites-available/layerpainter`:

```nginx
server {
    listen 80;
    server_name 192.168.0.146;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

3. Enable site: `sudo ln -s /etc/nginx/sites-available/layerpainter /etc/nginx/sites-enabled/`
4. Test config: `sudo nginx -t`
5. Reload Nginx: `sudo systemctl reload nginx`

Then update `NEXT_PUBLIC_API_BASE_URL` to use port 80 instead of 8000, and update CORS accordingly.
