# Updating LayerPainter from GitHub

The running app follows `main` on GitHub. Push from this checkout first. The server then fetches that commit. Do not rsync a working tree over `/opt/layerpainter`.

Project data, `backend/.env`, and `frontend/.env.production` stay on the server. They are not in git. The update does not replace the systemd units.

## 1. Push

From this repository, on `main`:

```bash
git push origin main
```

## 2. Update the server

SSH to the server and run:

```bash
sudo bash /opt/layerpainter/deployment/update-on-server.sh
```

That fetches `origin/main`, checks that commit out, restarts the API, rebuilds the frontend as `www-data` (so the existing production env file is used), and restarts the frontend.

The first time this script is not on the server yet, fetch it and run it:

```bash
curl -fsSL https://raw.githubusercontent.com/kutu95/paint_by_numbers/main/deployment/update-on-server.sh | sudo bash
```

## 3. Check

```bash
sudo systemctl is-active backend.service frontend.service
```

Hard-refresh the browser so it loads the new frontend bundle.
