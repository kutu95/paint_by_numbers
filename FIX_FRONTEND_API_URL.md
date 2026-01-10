# Fix Frontend API URL Configuration

## Problem
Frontend is trying to connect to `http://localhost:8000` instead of `https://layerpainter-api.margies.app`.

## Root Cause
Next.js environment variables starting with `NEXT_PUBLIC_` are **baked into the JavaScript bundle at BUILD time**, not runtime. The service file sets the environment variable, but the frontend was built with the wrong (or missing) value.

## Solution

On your Ubuntu server, you need to **rebuild the frontend** with the correct API URL:

```bash
cd /opt/layerpainter/frontend

# Set the environment variable for the build
export NEXT_PUBLIC_API_BASE_URL=https://layerpainter-api.margies.app

# Rebuild the frontend (this bakes the API URL into the bundle)
npm run build

# Restart the frontend service
sudo systemctl restart frontend.service

# Verify it's running
sudo systemctl status frontend.service
```

## Verify the Fix

After rebuilding:
1. Check the browser console - should no longer see `localhost:8000` errors
2. Check the Network tab - API calls should go to `https://layerpainter-api.margies.app`
3. Test layer generation - should work now

## Important Note

**Every time you rebuild the frontend**, make sure `NEXT_PUBLIC_API_BASE_URL` is set correctly before running `npm run build`.

You can verify what was baked in by checking:
```bash
cd /opt/layerpainter/frontend
grep -r "localhost:8000" .next/static/chunks/ 2>/dev/null | head -5
```

If you see `localhost:8000` in the built files, the rebuild didn't pick up the correct URL.

## Alternative: Update deploy.sh

You should also update `deployment/deploy.sh` to always set this when building:

```bash
# In deploy.sh, before npm run build:
export NEXT_PUBLIC_API_BASE_URL=https://layerpainter-api.margies.app
npm run build
```

This ensures the API URL is always set correctly during deployment.
