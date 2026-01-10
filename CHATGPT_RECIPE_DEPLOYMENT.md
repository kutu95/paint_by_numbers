# ChatGPT Recipe Generation - Deployment Instructions

## Overview
Recipe generation has been completely replaced with ChatGPT API integration. The app now asks ChatGPT to generate mixing instructions based on the target color hex value and available paints.

## Changes Made

### Backend
- Added `openai>=1.0.0` to `requirements.txt`
- Replaced `/api/paint/recipes/from-palette` endpoint to use ChatGPT API
- Removed dependency on complex recipe generation algorithms
- ChatGPT receives:
  - Target color hex value
  - List of available paints with their hex colors
  - Instructions to provide precise mixing percentages

### Frontend
- Updated `formatRecipe()` to handle new `chatgpt` recipe type
- Removed error display (ΔE) for ChatGPT recipes
- ChatGPT instructions are displayed as-is to the user

## Environment Variable Required

**CRITICAL**: You must set the `OPENAI_API_KEY` environment variable on your server.

### On Ubuntu Server:

```bash
# 1. Get your OpenAI API key from https://platform.openai.com/api-keys
# 2. Add it to the backend service environment

# Option A: Edit the service file directly
sudo nano /etc/systemd/system/backend.service

# Add this line in the [Service] section (replace YOUR_KEY_HERE with your actual key):
Environment="OPENAI_API_KEY=sk-YOUR_KEY_HERE"

# Save and exit (Ctrl+X, Y, Enter)

# Option B: Use a .env file (if using systemd environment file)
sudo nano /opt/layerpainter/backend/.env
# Add: OPENAI_API_KEY=sk-YOUR_KEY_HERE
# Then update backend.service to load it:
# EnvironmentFile=/opt/layerpainter/backend/.env

# 3. Reload systemd and restart the backend
sudo systemctl daemon-reload
sudo systemctl restart backend.service

# 4. Verify it's set
sudo systemctl show backend.service | grep OPENAI_API_KEY
```

## Deployment Steps

### 1. Pull Latest Code
```bash
cd /opt/layerpainter
git pull origin main
```

### 2. Install OpenAI Package
```bash
cd backend
source venv/bin/activate  # or: . venv/bin/activate
pip install openai>=1.0.0
pip freeze > requirements.txt  # Update lock file
deactivate
```

### 3. Set Environment Variable
```bash
# See "Environment Variable Required" section above
sudo nano /etc/systemd/system/backend.service
# Add: Environment="OPENAI_API_KEY=sk-YOUR_KEY_HERE"
sudo systemctl daemon-reload
```

### 4. Restart Backend
```bash
sudo systemctl restart backend.service
sudo systemctl status backend.service
```

### 5. Check Logs
```bash
# Verify no errors
sudo journalctl -u backend.service -n 50 --no-pager

# Test recipe generation by uploading an image and checking recipes
```

### 6. Rebuild Frontend (if needed)
```bash
cd /opt/layerpainter/frontend
npm run build
sudo systemctl restart frontend.service
```

## Testing

1. Upload an image and generate layers
2. Select a paint library group
3. Generate recipes - you should see ChatGPT-generated mixing instructions
4. Recipes should show natural language instructions with percentages

## Troubleshooting

### "OpenAI API key not configured" error
- Verify `OPENAI_API_KEY` is set: `sudo systemctl show backend.service | grep OPENAI_API_KEY`
- Check backend logs: `sudo journalctl -u backend.service -n 50`
- Ensure the service was restarted after setting the variable

### "Failed to generate recipe" errors
- Check OpenAI API key is valid
- Check you have API credits at https://platform.openai.com/usage
- Check backend logs for detailed error messages
- Verify internet connectivity from the server

### Recipes not showing
- Clear browser cache
- Check browser console for errors
- Verify frontend was rebuilt: `ls -la /opt/layerpainter/frontend/.next`

## Cost Considerations

ChatGPT API calls are made per recipe (one per palette color). Using `gpt-4o-mini`:
- Approximately $0.15 per 1M input tokens
- Approximately $0.60 per 1M output tokens
- Typical recipe generation: ~500 input tokens + ~100 output tokens per color
- 16 colors ≈ ~8000 input + ~1600 output tokens ≈ **~$0.003 per palette**

The model used is `gpt-4o-mini` which is the most cost-effective option.

## Notes

- The old recipe generation code in `paint_manager.py` is no longer used but remains in the codebase for reference
- All recipe generation now goes through ChatGPT
- ChatGPT provides natural language instructions that are more intuitive than formulaic percentages
