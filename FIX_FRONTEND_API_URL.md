# Fix Frontend API URL Configuration

## Problem
Frontend is trying to connect to `http://localhost:8000` instead of `https://layerpainter-api.margies.app`.

## Root Cause
Next.js environment variables starting with `NEXT_PUBLIC_` are **baked into the JavaScript bundle at BUILD time**, not runtime. The service file sets the environment variable, but the frontend was built with the wrong (or missing) value.

## Solution

On your Ubuntu server, do a **clean rebuild** so the correct API URL is baked in:

```bash
cd /opt/layerpainter/frontend

# Option A: Pass the env var inline (recommended – guarantees it’s used)
rm -rf .next
NEXT_PUBLIC_API_BASE_URL=https://layerpainter-api.margies.app npm run build
sudo systemctl restart frontend.service
```

**Option B: Use a file so you don’t have to type the URL each time**

```bash
cd /opt/layerpainter/frontend
cp .env.production.example .env.production
# Edit .env.production and set NEXT_PUBLIC_API_BASE_URL to your API URL
rm -rf .next
npm run build
sudo systemctl restart frontend.service
```

After deploying, **hard refresh the site** (Ctrl+Shift+R or Cmd+Shift+R) or open it in a private/incognito window so the browser doesn’t use an old cached bundle that still points at localhost.

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
