# Upgrading the App from GitHub on the Server

Use this procedure when you have pushed changes to GitHub and want to update the running app on your Ubuntu server. The server must have the app installed via **git clone** (e.g. as in [INSTALL_UBUNTU.md](../INSTALL_UBUNTU.md)) so that `git pull` works.

## Prerequisites

- SSH access to the server
- App installed at `/opt/layerpainter` with a git repository (cloned from GitHub)
- Backend and frontend running as systemd services (`backend.service`, `frontend.service`)

## Upgrade Steps

### 1. SSH into the server and pull latest code

```bash
ssh john@192.168.0.146

cd /opt/layerpainter
git pull origin main
```

If you see merge conflicts or “would be overwritten” errors, resolve them or stash local changes (e.g. `git stash` then `git pull origin main`). **Note:** `data/paint/library.json` is no longer in the repo (per-server data). The first time you pull after that change, git may remove the file on the server. Before pulling, back it up: `cp data/paint/library.json data/paint/library.json.bak`. After pulling, if the file is gone, restore it: `cp data/paint/library.json.bak data/paint/library.json`. Later pulls will not touch it (it’s ignored).

### 2. Update and restart the backend

```bash
cd /opt/layerpainter/backend

# If backend dependencies changed (requirements.txt)
source venv/bin/activate
pip install -r requirements.txt
deactivate

# Restart so new code is loaded
sudo systemctl restart backend.service
```

### 3. Rebuild and restart the frontend

The frontend build bakes `NEXT_PUBLIC_API_BASE_URL` into the bundle. Use the **same** URL your users hit for the API (e.g. your Cloudflare tunnel URL or server IP).

```bash
cd /opt/layerpainter/frontend

# If frontend dependencies changed (package.json)
npm install

# Build with the correct API URL (change if your setup differs)
export NEXT_PUBLIC_API_BASE_URL=https://layerpainter-api.margies.app
npm run build

# Restart so the new bundle is served
sudo systemctl restart frontend.service
```

If you use a different API URL (e.g. `http://YOUR_SERVER_IP:8000`), set that in `NEXT_PUBLIC_API_BASE_URL` before `npm run build`. See [FIX_FRONTEND_API_URL.md](../FIX_FRONTEND_API_URL.md) for details.

### 4. Verify

```bash
# Service status
sudo systemctl status backend.service
sudo systemctl status frontend.service

# Logs (if something fails)
sudo journalctl -u backend.service -n 50
sudo journalctl -u frontend.service -n 50
```

In the browser, do a **hard refresh** (Ctrl+Shift+R or Cmd+Shift+R) or open the app in a private/incognito window so the new frontend bundle is loaded.

## One-liner (after SSH)

If dependencies rarely change and your API URL is already set in the frontend service or env:

```bash
cd /opt/layerpainter && git pull origin main && sudo systemctl restart backend.service && cd frontend && npm run build && sudo systemctl restart frontend.service
```

Only use this if `NEXT_PUBLIC_API_BASE_URL` is already set correctly in your environment when `npm run build` runs (e.g. in a script or systemd unit).

## If the server was deployed without git (rsync only)

If the server copy was created with `rsync` or `deploy.sh` and there is **no** git repo at `/opt/layerpainter`, you have two options:

1. **Re-run your deploy script** from your local machine (e.g. `./deployment/deploy.sh john@192.168.0.146`) so it transfers the latest code and rebuilds on the server.
2. **Initialize git on the server** once, then use the steps above for future upgrades:
   ```bash
   cd /opt/layerpainter
   git init
   git remote add origin https://github.com/YOUR_ORG/paint_by_numbers.git
   git fetch origin main
   git reset --hard origin/main
   ```
   Then run the backend and frontend update steps (2 and 3) above.

## See also

- [DEPLOYMENT.md](DEPLOYMENT.md) – Initial deployment and service setup
- [INSTALL_UBUNTU.md](../INSTALL_UBUNTU.md) – Full install including “Updating the Application”
- [FIX_FRONTEND_API_URL.md](../FIX_FRONTEND_API_URL.md) – API URL and clean rebuild
