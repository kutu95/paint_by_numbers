# Recipe Fix Deployment Checklist

## Changes Made
1. ✅ Removed ALL 2-pigment recipe generation code
2. ✅ `find_best_multi_pigment_recipe` now ONLY accepts 3+ pigments (explicitly rejects 2)
3. ✅ Recipe generation requires minimum 3 colors
4. ✅ All changes committed and pushed to GitHub

## Verify on Server

### Step 1: Pull Latest Code
```bash
cd /opt/layerpainter
git pull origin main
```

### Step 2: Verify Changes
```bash
# Check that the latest commit is there
git log --oneline -3

# Should see:
# 95cc8dc Explicitly reject 2-pigment recipes in find_best_multi_pigment_recipe - only allow 3+ pigments
# ad12f79 Fix duplicate recipe append
# 5b596f7 Remove 2-color fallback completely - require minimum 3 colors in all recipes
```

### Step 3: Verify Code Changes
```bash
# Check that 2-pigment code is removed
grep -n "Try 2 pigments\|find_best_two_pigment_recipe" backend/paint_manager.py
# Should return NOTHING (no matches)

# Check that 3+ pigment requirement is in place
grep -n "REQUIRED minimum\|best_pigment_count >= 3" backend/paint_manager.py
# Should show the requirement check
```

### Step 4: Restart Backend
```bash
sudo systemctl restart backend.service
sudo systemctl status backend.service
```

### Step 5: Check Backend Logs
```bash
sudo journalctl -u backend.service -n 50 --no-pager
# Look for any errors
```

### Step 6: Test Recipe Generation
1. Go to the app in browser
2. Upload an image
3. Select a paint library with 3+ paints
4. Generate recipes
5. **VERIFY**: All recipes should have 3+ colors (no 2-color recipes)

## If Still Getting 2-Color Recipes

### Check 1: Is the server code updated?
```bash
cd /opt/layerpainter
git log --oneline -1
# Should show: 95cc8dc
```

### Check 2: Is the backend service restarted?
```bash
sudo systemctl status backend.service
# Should show: active (running)
```

### Check 3: Check backend code directly
```bash
cd /opt/layerpainter
grep -A 5 "Try 3 pigments - REQUIRED" backend/paint_manager.py
# Should show the 3-pigment requirement code
```

### Check 4: Clear browser cache
- Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
- Or clear browser cache completely

### Check 5: Test API directly
```bash
# Get a session ID first, then test recipe generation
curl -X POST "http://localhost:8000/api/paint/recipes" \
  -F "palette=[{\"index\":0,\"hex\":\"#FF0000\"}]" \
  -F "library_group=default"
# Check the response - should NOT have type: 'two_pigment'
```

## Expected Behavior
- ✅ All recipes use 3+ colors when 3+ paints are available
- ✅ No 2-color recipes should ever appear
- ✅ If only 1-2 paints available, will use one-pigment recipe (not 2-pigment)
