# Layer Numbering Fix - Deployment Instructions

## Issue
Preview page showing layers as 0-15 instead of 1-16.

## Fix Applied
The code has been updated to display `{layerIdx + 1}` instead of `{layerIdx}`.

## Verification
The change is confirmed in git:
- File: `frontend/app/page.tsx`
- Line 868: `<div className="text-lg font-mono">{layerIdx + 1}</div>`
- Commit: `02f3df9` and `d3ec3f2`

## Deployment Steps

### On Your Ubuntu Server:

```bash
# 1. Navigate to project directory
cd /opt/layerpainter

# 2. Pull latest code
git pull origin main

# 3. Verify the change is present
grep -n "layerIdx + 1" frontend/app/page.tsx
# Should show: 868:                      <div className="text-lg font-mono">{layerIdx + 1}</div>

# 4. Navigate to frontend directory
cd frontend

# 5. Clear Next.js build cache (important!)
rm -rf .next

# 6. Rebuild the frontend
npm run build

# 7. Restart the frontend service
sudo systemctl restart frontend.service

# 8. Check service status
sudo systemctl status frontend.service
```

### Clear Browser Cache
After deployment, **hard refresh** your browser:
- **Chrome/Edge**: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- **Firefox**: `Ctrl+F5` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- Or clear browser cache completely

## Expected Result
After deployment and cache clear:
- Preview page: Layers numbered **1-16** (not 0-15)
- Projection page: Layers numbered **1-16** (already correct)

## If Still Not Working

1. **Verify code on server**:
   ```bash
   cd /opt/layerpainter
   git log --oneline -3
   # Should see: 02f3df9 Fix remaining layer numbering
   
   cat frontend/app/page.tsx | grep -n "layerIdx + 1"
   # Should show line 868 with {layerIdx + 1}
   ```

2. **Check if frontend was rebuilt**:
   ```bash
   ls -la /opt/layerpainter/frontend/.next
   # Should exist and be recent (after git pull)
   ```

3. **Clear browser cache completely** or use incognito/private browsing mode

4. **Check browser console** for any JavaScript errors
