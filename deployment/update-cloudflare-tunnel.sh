#!/bin/bash
# Script to update Cloudflare tunnel config with LayerPainter routes
# Usage: ./update-cloudflare-tunnel.sh [domain]
# Example: ./update-cloudflare-tunnel.sh landlife.au

DOMAIN=${1:-"landlife.au"}  # Default to landlife.au if not specified
CONFIG_FILE="/home/john/.cloudflared/config.yml"
BACKUP_FILE="${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"

echo "Updating Cloudflare tunnel config for domain: ${DOMAIN}"

# Create backup
echo "Creating backup at: ${BACKUP_FILE}"
cp "${CONFIG_FILE}" "${BACKUP_FILE}"

# Check if routes already exist
if grep -q "layerpainter.${DOMAIN}" "${CONFIG_FILE}"; then
    echo "Warning: LayerPainter routes already exist in config file"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Create updated config (add LayerPainter routes before catch-all)
cat > /tmp/new_config.yml << EOF
tunnel: farm-cashbook
credentials-file: /home/john/.cloudflared/b2be279d-ebd1-41c7-8b33-c6e64b24547d.json

ingress:
  - hostname: supabase.landlife.au
    service: http://localhost:54321
  - hostname: books.landlife.au
    service: http://localhost:3001
  - hostname: media.margies.app
    service: http://localhost:2342
    originRequest:
      httpHostHeader: media.margies.app
  - hostname: tryon.margies.app
    service: http://localhost:3002
  - hostname: layerpainter.${DOMAIN}
    service: http://localhost:3003
  - hostname: layerpainter-api.${DOMAIN}
    service: http://localhost:8000
  - service: http_status:404
EOF

echo "New config created. Review it:"
cat /tmp/new_config.yml
echo ""
read -p "Apply this config? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cp /tmp/new_config.yml "${CONFIG_FILE}"
    echo "Config updated! Restart cloudflared with: sudo systemctl restart cloudflared"
else
    echo "Aborted. Config not changed."
    rm /tmp/new_config.yml
fi
