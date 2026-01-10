# Clear Old Recipe Cache

## Problem
Old recipes from the previous algorithm are cached and being used instead of ChatGPT-generated recipes.

## Solution 1: Use "Force Regenerate" Button (Easiest)

On the frontend, click the **"Force Regenerate"** button (orange button next to "Generate Recipes"). This will:
- Ignore cached recipes
- Generate new recipes from ChatGPT
- Update the cache with the new ChatGPT recipes

## Solution 2: Clear Cache Files on Server (If Force Regenerate Doesn't Work)

SSH into your Ubuntu server and delete the cache files:

```bash
# Navigate to the cache directory
cd /opt/layerpainter/data/paint/recipes_cache

# List cache files
ls -la

# Delete all cache files (this will force regeneration of all recipes)
rm -f *.json

# Or delete cache for a specific library group (e.g., "default")
rm -f default_recipes.json

# Restart backend to ensure clean state
sudo systemctl restart backend.service
```

## Solution 3: Verify Backend is Using ChatGPT

Check the backend logs to confirm it's calling ChatGPT:

```bash
sudo journalctl -u backend.service -n 100 --no-pager | grep -i "chatgpt\|generating\|recipe"
```

You should see messages like:
- `Generating new recipe for color #XXXXXX in group default`
- `Force regenerating recipe for color #XXXXXX in group default`

If you see old algorithm messages, the backend code hasn't been updated. Pull the latest code:

```bash
cd /opt/layerpainter
git pull origin main
cd backend
source venv/bin/activate
pip install -r requirements.txt  # Make sure openai package is installed
deactivate
sudo systemctl restart backend.service
```

## Verify ChatGPT is Working

After clearing cache and regenerating:
1. Check backend logs for ChatGPT API calls
2. Recipes should show natural language instructions (not just percentages)
3. Recipes should NOT show "Error: X.XX ΔE" (that's from the old algorithm)

## Note

The cache structure changed when we switched to ChatGPT. Old cached recipes from the previous algorithm may still be in the cache files. Using "Force Regenerate" will replace them with ChatGPT recipes.
