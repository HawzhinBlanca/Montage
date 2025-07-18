#!/usr/bin/env bash
set -e
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
docker compose up -d postgres redis prometheus
python - <<'PY'
import psycopg2, pathlib, os
from dotenv import load_dotenv; load_dotenv()
conn = psycopg2.connect(os.environ["DATABASE_URL"])
conn.cursor().execute(pathlib.Path("schema.sql").read_text()); conn.commit()
PY
echo "Bootstrap complete."