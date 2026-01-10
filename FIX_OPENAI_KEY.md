# Fix OpenAI API Key Configuration

## Problem
The systemd service file shows: `Invalid environment assignment, ignoring:`

This means the environment variable line is incorrectly formatted in `/etc/systemd/system/backend.service`.

## Solution

Edit the service file on your Ubuntu server:

```bash
sudo nano /etc/systemd/system/backend.service
```

**Make sure the `[Service]` section looks exactly like this** (with your actual API key):

```ini
[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/layerpainter/backend
Environment="PATH=/opt/layerpainter/backend/venv/bin"
Environment="CORS_ORIGINS=https://layerpainter.margies.app,http://192.168.0.146:3003,http://localhost:3003,http://192.168.0.146:3000,http://localhost:3000"
Environment="OPENAI_API_KEY=sk-proj-YOUR_API_KEY_HERE"
ExecStart=/opt/layerpainter/backend/venv/bin/python main.py
Restart=always
RestartSec=10
```

**Important points:**
- Each `Environment=` line must be on its own line
- The key and value must be inside double quotes: `Environment="OPENAI_API_KEY=..."`
- There should be NO spaces around the `=` sign
- The entire API key should be on one line (no line breaks)

After editing, save (Ctrl+X, then Y, then Enter) and then:

```bash
# Reload systemd to read the updated file
sudo systemctl daemon-reload

# Restart the service
sudo systemctl restart backend.service

# Check that it started correctly (should NOT see "Invalid environment assignment")
sudo systemctl status backend.service

# Verify the key is loaded
sudo systemctl show backend.service | grep OPENAI_API_KEY
```

The last command should show: `OPENAI_API_KEY=sk-proj-...`

## Alternative: Use EnvironmentFile (More Secure)

If you prefer, you can use a separate `.env` file:

1. Create the environment file:
```bash
sudo nano /opt/layerpainter/backend/.env
```

Add:
```
OPENAI_API_KEY=sk-proj-YOUR_API_KEY_HERE
```

2. Secure it:
```bash
sudo chmod 600 /opt/layerpainter/backend/.env
sudo chown www-data:www-data /opt/layerpainter/backend/.env
```

3. Update the service file to use it:
```bash
sudo nano /etc/systemd/system/backend.service
```

Change:
```ini
EnvironmentFile=/opt/layerpainter/backend/.env
Environment="PATH=/opt/layerpainter/backend/venv/bin"
Environment="CORS_ORIGINS=..."
```

Note: When using `EnvironmentFile`, you should NOT also have `Environment="OPENAI_API_KEY=..."` in the service file. The `.env` file will load it.

4. Reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart backend.service
```
