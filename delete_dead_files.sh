#!/bin/bash
# Script to delete identified dead files
# Review carefully before running!

set -e

echo "Deleting montage/core/resource_watchdog.py..."
git rm montage/core/resource_watchdog.py
echo "Deleting montage/core/visual_tracker.py..."
git rm montage/core/visual_tracker.py
echo "Deleting montage/providers/smart_track.py..."
git rm montage/providers/smart_track.py
echo "Deleting montage/utils/video_effects.py..."
git rm montage/utils/video_effects.py
echo "Deleting montage/utils/ffmpeg_memory_manager.py..."
git rm montage/utils/ffmpeg_memory_manager.py
echo "Deleting montage/api/celery_app.py..."
git rm montage/api/celery_app.py
echo "Deleting montage/jobs/celery_app.py..."
git rm montage/jobs/celery_app.py

echo "Deleted {len(deletable)} files"
