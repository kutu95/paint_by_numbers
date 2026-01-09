# Cloudflare Tunnel Setup for LayerPainter

## Current App Configuration
- **Frontend**: http://192.168.0.146:3003
- **Backend API**: http://192.168.0.146:8000
- **Tunnel ID**: b2be279d-ebd1-41c7-8b33-c6e64b24547d.cfargotunnel.com

## Setup Steps

### Option 1: Add to Existing Tunnel Config (Recommended)

If you're using a self-hosted cloudflared with a config file:

1. **Find your tunnel config file** (usually one of these):
   ```bash
   sudo cat /etc/cloudflared/config.yml
   # OR
   cat ~/.cloudflared/config.yml
   # OR check systemd service
   sudo systemctl cat cloudflared
   ```

2. **Add LayerPainter routes to your config.yml**:
   
   ```yaml
   tunnel: b2be279d-ebd1-41c7-8b33-c6e64b24547d
   credentials-file: /etc/cloudflared/credentials.json  # or your credentials file path
   
   ingress:
     # Your existing routes here...
     
     # LayerPainter Frontend
     - hostname: layerpainter.yourdomain.com  # Replace with your domain
       service: http://localhost:3003
     
     # LayerPainter Backend API
     - hostname: layerpainter-api.yourdomain.com  # Replace with your domain
       service: http://localhost:8000
     
     # OR use path-based routing (if you prefer subdirectories)
     # - hostname: yourdomain.com
     #   path: /layerpainter/*
     #   service: http://localhost:3003
     # - hostname: yourdomain.com
     #   path: /layerpainter/api/*
     #   service: http://localhost:8000
     
     # Catch-all rule (must be last)
     - service: http_status:404
   ```

3. **Restart cloudflared**:
   ```bash
   sudo systemctl restart cloudflared
   sudo systemctl status cloudflared
   ```

### Option 2: Using Cloudflare Dashboard (Managed Tunnel)

If your tunnel is managed via Cloudflare dashboard:

1. Go to **Cloudflare Dashboard** → **Zero Trust** → **Networks** → **Tunnels**
2. Click on your tunnel: `b2be279d-ebd1-41c7-8b33-c6e64b24547d`
3. Click **Configure** → **Public Hostname**
4. Add two public hostnames:

   **Frontend:**
   - Subdomain: `layerpainter` (or your choice)
   - Domain: Select your domain
   - Service Type: HTTP
   - Service URL: `http://localhost:3003`
   
   **Backend API:**
   - Subdomain: `layerpainter-api` (or your choice)
   - Domain: Select your domain
   - Service Type: HTTP
   - Service URL: `http://localhost:8000`

### Option 3: Path-Based Routing (Single Domain)

If you want everything under one domain with paths:

```yaml
ingress:
  # LayerPainter Frontend (root or /app)
  - hostname: yourdomain.com
    path: /layerpainter*
    service: http://localhost:3003
  
  # LayerPainter Backend API
  - hostname: yourdomain.com
    path: /layerpainter-api/*
    service: http://localhost:8000
  
  # Catch-all
  - service: http_status:404
```

**Note**: For path-based routing, you'll need to update the frontend to proxy API requests or configure Next.js rewrites.

## Update CORS Configuration

After setting up the tunnel, update the backend CORS to allow your tunnel domain:

1. **Find your tunnel URL** (e.g., `https://layerpainter.yourdomain.com`)

2. **Update backend service file**:
   ```bash
   sudo nano /etc/systemd/system/backend.service
   ```

3. **Update CORS_ORIGINS**:
   ```ini
   Environment="CORS_ORIGINS=https://layerpainter.yourdomain.com,http://192.168.0.146:3003,http://localhost:3003"
   ```

4. **Restart backend**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart backend.service
   ```

## Update Frontend API Base URL

If using a subdomain for the API:

1. **Update frontend service file**:
   ```bash
   sudo nano /etc/systemd/system/frontend.service
   ```

2. **Update NEXT_PUBLIC_API_BASE_URL**:
   ```ini
   Environment="NEXT_PUBLIC_API_BASE_URL=https://layerpainter-api.yourdomain.com"
   ```

3. **Rebuild and restart frontend**:
   ```bash
   cd /opt/layerpainter/frontend
   export NEXT_PUBLIC_API_BASE_URL=https://layerpainter-api.yourdomain.com
   npm run build
   sudo systemctl restart frontend.service
   ```

## Alternative: Use Environment Variables

You can also set these in `/etc/environment` or use a `.env` file for Next.js.

## Verify Setup

1. **Check tunnel is running**:
   ```bash
   sudo systemctl status cloudflared
   ```

2. **Check tunnel logs**:
   ```bash
   sudo journalctl -u cloudflared -n 50
   ```

3. **Test from browser**:
   - Visit: `https://layerpainter.yourdomain.com`
   - Should load the frontend
   - Check browser console for any CORS errors

4. **Test API**:
   ```bash
   curl https://layerpainter-api.yourdomain.com/docs
   ```

## Troubleshooting

### CORS Errors
- Ensure backend CORS_ORIGINS includes your tunnel domain
- Restart backend after changing CORS settings

### 502 Bad Gateway
- Check that frontend/backend services are running on localhost
- Verify port numbers in tunnel config match service ports

### Connection Refused
- Verify services are listening on localhost (not 0.0.0.0 or 192.168.0.146)
- Check firewall allows localhost connections

### Domain Not Working
- Wait a few minutes for DNS propagation
- Check Cloudflare DNS settings for your domain
- Verify tunnel is active in Cloudflare dashboard

## Example Complete Config

Here's an example of what your `/etc/cloudflared/config.yml` might look like:

```yaml
tunnel: b2be279d-ebd1-41c7-8b33-c6e64b24547d
credentials-file: /etc/cloudflared/credentials.json

ingress:
  # Existing app 1 (example)
  - hostname: app1.yourdomain.com
    service: http://localhost:3000
  
  # Existing app 2 (example)
  - hostname: app2.yourdomain.com
    service: http://localhost:3001
  
  # LayerPainter Frontend
  - hostname: layerpainter.yourdomain.com
    service: http://localhost:3003
  
  # LayerPainter Backend API
  - hostname: layerpainter-api.yourdomain.com
    service: http://localhost:8000
  
  # Catch-all (must be last)
  - service: http_status:404
```
