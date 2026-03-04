# Setting Up OpenAI API Key

Recipe generation uses the OpenAI API. If you see "OpenAI API key not configured", set the key as below.

**Why did my key disappear?** A deploy or upgrade that copies `deployment/backend.service` over `/etc/systemd/system/backend.service` replaces the whole file. If the key was only in that file, it gets overwritten. Use the `.env` method below so your key lives in a file that is not overwritten by deploy.

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

## Method 2: Use environment file (Recommended – survives deploys)

The repo’s `backend.service` already loads `EnvironmentFile=-/opt/layerpainter/backend/.env` if it exists. Create that file with your key so it is not overwritten when you deploy:

```bash
sudo nano /opt/layerpainter/backend/.env
```

Add (replace with your actual key):

```
OPENAI_API_KEY=sk-proj-YOUR_API_KEY_HERE
```

Secure the file:

```bash
sudo chmod 600 /opt/layerpainter/backend/.env
sudo chown www-data:www-data /opt/layerpainter/backend/.env
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
