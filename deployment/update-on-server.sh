#!/bin/bash
# Bring the running app in line with GitHub main.
# On the server: sudo bash /opt/layerpainter/deployment/update-on-server.sh
set -euo pipefail

APP=/opt/layerpainter
cd "$APP"

git -c safe.directory="$APP" fetch origin main
git -c safe.directory="$APP" reset --hard origin/main

chown -R www-data:www-data "$APP/backend" "$APP/frontend" "$APP/deployment"

if [ -f "$APP/frontend/.env.production" ]; then
  chown www-data:www-data "$APP/frontend/.env.production"
  chmod 600 "$APP/frontend/.env.production"
fi
if [ -f "$APP/backend/.env" ]; then
  chown www-data:www-data "$APP/backend/.env"
  chmod 600 "$APP/backend/.env"
fi

systemctl restart backend.service
sudo -u www-data env HOME=/var/www npm --prefix "$APP/frontend" run build
systemctl restart frontend.service
systemctl is-active backend.service frontend.service
