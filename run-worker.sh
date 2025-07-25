#!/bin/bash
# Tasks.md Step 1: Celery worker startup script
# Loads environment and starts Celery worker with exact specifications

# Source environment (loads MODEL_HOME etc.)
source ~/.zshrc

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Start Celery worker with exact parameters from Tasks.md
exec celery -A montage.jobs.celery_app worker \
     --concurrency=4 --prefetch-multiplier=1 \
     --loglevel=info