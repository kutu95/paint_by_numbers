# Setting Up OpenAI API Key

## ⚠️ SECURITY WARNING
**NEVER commit your API key to git or store it in code files!**

## Method 1: Set in systemd service file (Recommended)

On your Ubuntu server, edit the backend service file:

```bash
sudo nano /etc/systemd/system/backend.service
```

Add this line in the `[Service]` section (replace the key with your actual key):

```ini
[Service]
Environment="OPENAI_API_KEY=sk-proj-YOUR_API_KEY_HERE"
```

**Note:** Replace `YOUR_API_KEY_HERE` with your actual OpenAI API key (the one starting with `sk-proj-...`).

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart backend.service
sudo systemctl status backend.service
```

## Method 2: Use environment file (More secure)

Create a secure environment file:

```bash
sudo nano /opt/layerpainter/backend/.env
```

Add:
```
OPENAI_API_KEY=sk-proj-YOUR_API_KEY_HERE
```

**Note:** Replace `YOUR_API_KEY_HERE` with your actual OpenAI API key (the one starting with `sk-proj-...`).

Secure the file:
```bash
sudo chmod 600 /opt/layerpainter/backend/.env
sudo chown www-data:www-data /opt/layerpainter/backend/.env
```

Then update `/etc/systemd/system/backend.service` to load it:
```ini
[Service]
EnvironmentFile=/opt/layerpainter/backend/.env
```

Reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart backend.service
```

## Verify it's working

Check the backend logs:
```bash
sudo journalctl -u backend.service -n 50 --no-pager
```

You should NOT see "OpenAI API key not configured" errors.

Test by generating recipes in the app - they should now come from ChatGPT.

## If you need to update the key

1. Edit the service file or .env file with the new key
2. Run `sudo systemctl daemon-reload`
3. Run `sudo systemctl restart backend.service`
4. Verify with logs
