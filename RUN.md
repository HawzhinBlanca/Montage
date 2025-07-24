# Quick Start (Developer)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
redis-server --daemonize yes
celery -A montage.jobs.tasks worker -l info &
uvicorn montage.api.service:app --port 8000
# POST /process     → returns job_id
# GET  /status/{id} → progress
# GET  /download/{id} → mp4 when done

```