#!/bin/bash
# Tasks.md Step 1: Gunicorn API startup script
# Loads environment and starts Gunicorn with Uvicorn workers

# Source environment (loads MODEL_HOME etc.)
source ~/.zshrc

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Start Gunicorn with exact parameters from Tasks.md
exec gunicorn montage.api.app:app \
     --worker-class uvicorn.workers.UvicornWorker \
     --bind 0.0.0.0:8080 --workers 2