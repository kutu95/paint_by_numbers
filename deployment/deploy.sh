#!/bin/bash
# This no longer copies a working tree onto the server.
# Push to GitHub, then on the server run:
#   sudo bash /opt/layerpainter/deployment/update-on-server.sh
echo "Push main to GitHub first. Then, on the server, run:" >&2
echo "  sudo bash /opt/layerpainter/deployment/update-on-server.sh" >&2
exit 1
