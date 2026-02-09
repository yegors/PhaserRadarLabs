#!/bin/bash
# Deploy PhaserRadarLabs to the Phaser Pi
# Usage: bash tools/deploy.sh

REMOTE_USER="analog"
REMOTE_HOST="phaser.local"
REMOTE_DIR="/home/analog/yolo"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Deploying Playground -> $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

tar --exclude='__pycache__' --exclude='*.pyc' -cf - -C "$PROJECT_DIR" Playground | ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_DIR && tar xf -"

echo "Done."
