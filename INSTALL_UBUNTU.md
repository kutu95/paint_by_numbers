# Ubuntu Server Installation Guide

Complete installation guide for LayerPainter on Ubuntu server at `192.168.0.146`.

## Prerequisites Check

First, SSH into your server:
```bash
ssh user@192.168.0.146
```

Check if required software is installed:
```bash
# Check Python (should be 3.8+)
python3 --version

# Check Node.js (should be 18+)
node --version

# Check npm
npm --version
```

## Step 1: Install Prerequisites (if needed)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv

# Install Node.js 18+ (if not installed)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installations
python3 --version
node --version
npm --version
```

## Step 2: Clone the Repository

```bash
# Create application directory
sudo mkdir -p /opt/layerpainter
sudo chown $USER:$USER /opt/layerpainter

# Clone from GitHub
cd /opt/layerpainter
git clone https://github.com/kutu95/paint_by_numbers.git .

# Verify files were cloned
ls -la
```

## Step 3: Setup Backend

```bash
cd /opt/layerpainter/backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Test that it works (optional - press Ctrl+C after checking)
python main.py
# You should see: "INFO:     Uvicorn running on http://0.0.0.0:8000"
# Press Ctrl+C to stop
deactivate
```

## Step 4: Setup Frontend

```bash
cd /opt/layerpainter/frontend

# Install Node.js dependencies
npm install

# Set environment variable for API URL
export NEXT_PUBLIC_API_BASE_URL=http://192.168.0.146:8000

# Build the production bundle
npm run build

# Verify build succeeded (should see "Creating an optimized production build")
ls -la .next
```

## Step 5: Configure Systemd Services

```bash
# Copy service files to systemd directory
sudo cp /opt/layerpainter/deployment/backend.service /etc/systemd/system/
sudo cp /opt/layerpainter/deployment/frontend.service /etc/systemd/system/

# Edit the service files to match your setup (if needed)
# Check the IP address and paths in the files:
sudo nano /etc/systemd/system/backend.service
sudo nano /etc/systemd/system/frontend.service
```

**Important settings to verify in the service files:**

In `backend.service`, verify:
- `WorkingDirectory=/opt/layerpainter/backend` (correct path)
- `CORS_ORIGINS=http://192.168.0.146:3000,http://localhost:3000` (your server IP)
- `User=www-data` (or change to your username)

In `frontend.service`, verify:
- `WorkingDirectory=/opt/layerpainter/frontend` (correct path)
- `NEXT_PUBLIC_API_BASE_URL=http://192.168.0.146:8000` (your server IP)
- `User=www-data` (or change to your username)

## Step 6: Set Permissions

```bash
# Option 1: Use www-data user (recommended for production)
sudo chown -R www-data:www-data /opt/layerpainter

# Option 2: Use your user account (easier for testing)
# Replace 'yourusername' with your actual username
sudo chown -R yourusername:yourusername /opt/layerpainter
# Then update the service files to use your username instead of www-data
```

**If using your own user**, update the service files:
```bash
sudo sed -i 's/User=www-data/User=yourusername/g' /etc/systemd/system/backend.service
sudo sed -i 's/User=www-data/User=yourusername/g' /etc/systemd/system/frontend.service
```

## Step 7: Enable and Start Services

```bash
# Reload systemd to recognize new services
sudo systemctl daemon-reload

# Enable services to start automatically on boot
sudo systemctl enable backend.service
sudo systemctl enable frontend.service

# Start the services
sudo systemctl start backend.service
sudo systemctl start frontend.service

# Check if they're running
sudo systemctl status backend.service
sudo systemctl status frontend.service
```

## Step 8: Configure Firewall (if enabled)

If you have `ufw` firewall enabled:

```bash
# Check firewall status
sudo ufw status

# If firewall is active, allow the ports
sudo ufw allow 8000/tcp comment 'LayerPainter Backend'
sudo ufw allow 3000/tcp comment 'LayerPainter Frontend'

# Verify rules were added
sudo ufw status numbered
```

## Step 9: Verify Installation

### Check Backend
```bash
# Test backend API
curl http://localhost:8000/docs
# Or open in browser: http://192.168.0.146:8000/docs
```

### Check Frontend
```bash
# Test frontend
curl http://localhost:3000
# Or open in browser: http://192.168.0.146:3000
```

### Check Logs
```bash
# View backend logs
sudo journalctl -u backend.service -n 50 --no-pager

# View frontend logs
sudo journalctl -u frontend.service -n 50 --no-pager

# Follow logs in real-time
sudo journalctl -u backend.service -f
sudo journalctl -u frontend.service -f
```

## Troubleshooting

### Service won't start

**Check service status:**
```bash
sudo systemctl status backend.service
sudo systemctl status frontend.service
```

**Check logs:**
```bash
sudo journalctl -u backend.service -n 100
sudo journalctl -u frontend.service -n 100
```

### Backend errors

**Common issues:**
- Virtual environment not found → Check path in service file
- Port 8000 already in use → `sudo lsof -i :8000` to find what's using it
- Permission denied → Check file ownership: `ls -la /opt/layerpainter`

**Fix permissions:**
```bash
sudo chown -R www-data:www-data /opt/layerpainter
# Or use your username
sudo chown -R $USER:$USER /opt/layerpainter
```

### Frontend errors

**Common issues:**
- Build failed → Rebuild: `cd /opt/layerpainter/frontend && npm run build`
- Port 3000 already in use → `sudo lsof -i :3000`
- Environment variable not set → Check service file has `NEXT_PUBLIC_API_BASE_URL`

**Rebuild frontend:**
```bash
cd /opt/layerpainter/frontend
export NEXT_PUBLIC_API_BASE_URL=http://192.168.0.146:8000
npm run build
sudo systemctl restart frontend.service
```

### CORS errors in browser

**Update backend CORS:**
```bash
# Edit backend service
sudo nano /etc/systemd/system/backend.service

# Update CORS_ORIGINS environment variable to include your client IP
# Example: Environment="CORS_ORIGINS=http://192.168.0.146:3000,http://192.168.0.1:3000"

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart backend.service
```

## Useful Commands

### View logs
```bash
# Last 50 lines
sudo journalctl -u backend.service -n 50
sudo journalctl -u frontend.service -n 50

# Follow logs in real-time
sudo journalctl -u backend.service -f
sudo journalctl -u frontend.service -f

# Logs since today
sudo journalctl -u backend.service --since today
```

### Restart services
```bash
sudo systemctl restart backend.service
sudo systemctl restart frontend.service

# Or restart both at once
sudo systemctl restart backend.service frontend.service
```

### Stop services
```bash
sudo systemctl stop backend.service
sudo systemctl stop frontend.service
```

### Disable auto-start on boot
```bash
sudo systemctl disable backend.service
sudo systemctl disable frontend.service
```

## Accessing the Application

Once everything is running:
- **Frontend (Web UI)**: http://192.168.0.146:3000
- **Backend API**: http://192.168.0.146:8000
- **API Documentation**: http://192.168.0.146:8000/docs

## Updating the Application

When you push new changes to GitHub:

```bash
cd /opt/layerpainter

# Pull latest changes
git pull origin main

# Restart services to pick up changes
sudo systemctl restart backend.service

# If frontend changed, rebuild
cd frontend
npm install  # If dependencies changed
npm run build
sudo systemctl restart frontend.service
```

## Next Steps

After installation:
1. Test uploading an image at http://192.168.0.146:3000
2. Set up paint library (click "Manage Paints")
3. Configure any additional settings
4. Optionally set up Nginx reverse proxy (see deployment/DEPLOYMENT.md)
