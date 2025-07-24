"""
Celery app configuration for Montage video processing pipeline
Tasks.md Step 1: Real Celery worker implementation
"""
from celery import Celery
import os

# Configure Celery with Redis backend
from ..settings import settings
redis_url = settings.redis.url.get_secret_value()

app = Celery(
    'montage',
    broker=redis_url,
    backend=redis_url,
    include=['montage.jobs.tasks']
)

# Celery configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3000,  # 50 min soft limit
    worker_prefetch_multiplier=1,  # As specified in Tasks.md
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks to prevent memory leaks
)

# Import tasks to register them
from . import tasks