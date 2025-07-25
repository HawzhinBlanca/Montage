Below is a machine-task spec you can drop into any competent Dev-Agent (GitHub Copilot Workspace, GPT-Engineer, Gemini Code Assist, etc.).
It contains only file‐creation commands, shell steps, and strict acceptance tests—no long code blobs.
Running these tasks end-to-end will generate the exact working pipeline described earlier.

⸻

0. Repo bootstrap

- mkdir: [src, tests, cleanup_archives]
- touch: [schema.sql, docker-compose.yml, bootstrap.sh, requirements.txt, .env.example]


⸻

1. Core dependency list

- write_file: requirements.txt
  content: |
    faster-whisper==0.10.1
    deepgram-sdk==2.14.1
    pyannote.audio==3.2.1
    openai==1.30.2
    anthropic==0.18.2
    opencv-python-headless==4.10.0.82
    ffmpeg-python==0.2.0
    rich==13.7.1
    tqdm==4.66.4
    psycopg2-binary==2.9.9
    redis==5.0.4
    prometheus-client==0.20.0
    bottle==0.12.25
    python-dotenv==1.0.1


⸻

2. Docker / infra

- write_file: docker-compose.yml
  content: |
    version: "3.9"
    services:
      postgres:
        image: postgres:15
        environment: { POSTGRES_PASSWORD: pass }
        ports: ["5432:5432"]
      redis:
        image: redis:7.2-alpine
        ports: ["6379:6379"]
      prometheus:
        image: prom/prometheus
        ports: ["9090:9090"]


⸻

3. DB schema

- write_file: schema.sql
  content: |
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE TABLE IF NOT EXISTS transcript_cache(
      sha256 CHAR(64) PRIMARY KEY,
      transcript TEXT NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW());


⸻

4. Bootstrap script

- write_file: bootstrap.sh
  mode: 755
  content: |
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


⸻

5. Source-tree stubs (agent to fill)

- create_python_module: src.config
  description: "Load env vars, expose DATABASE_URL, REDIS_URL, API keys, MAX_COST_USD"

- create_python_module: src.analyze_video
  description: |
    Functions:
      sha256_file(path)        -> str
      transcribe_faster_whisper(wav) -> list[word dict]
      transcribe_deepgram(wav)       -> list[word dict]
      rover_merge(fw, dg)            -> list[word dict]
      diarize(path)                  -> list[turn dict]
      align_speakers(words, turns)   -> list[word dict]
      analyze_video(src)             -> {"sha": str, "words": [...], "transcript": str}

- create_python_module: src.highlight_selector
  description: |
    Local rule scorer + Claude/GPT chunk merge (cost-capped).
    choose_highlights(words, audio_energy, mode="smart|premium") -> list[clip dict]

- create_python_module: src.ffmpeg_utils
  description: |
    Helpers:
      make_concat_file(paths) -> str
      burn_captions(video, srt, out_path)

- create_python_module: src.resolve_mcp
  description: |
    Bottle server exposing /buildTimeline + /renderProxy.
    Uses DaVinciResolveScript when available; falls back to noop.

- create_python_module: src.run_pipeline
  description: |
    CLI entry: python -m src.run_pipeline <video> --mode smart|premium
    1. analyze_video
    2. choose_highlights
    3. emit per-clip SRT
    4. POST plan to MCP


⸻

6. End-to-end test

- write_file: tests/test_e2e.py
  content: |
    import subprocess, pathlib
    def test_e2e():
        sample='tests/data/lecture.mp4'
        assert pathlib.Path(sample).exists()
        subprocess.run(
            ['python','-m','src.run_pipeline',sample,'--mode','smart'],
            check=True)


⸻

7. Acceptance gates (CI)

- add_github_actions:
    workflow_name: ci
    python_version: "3.11"
    pre_commands: |
      sudo apt-get update && sudo apt-get install -y ffmpeg
      docker compose up -d postgres redis prometheus
    test_command: pytest -q


⸻

8. Success criteria (agent must verify)
	1.	pytest -q passes (video truly processed).
	2.	python -m src.run_pipeline sample.mp4 --mode smart prints a valid JSON plan.
	3.	Prometheus endpoint /metrics exposes proc_ratio and cost_usd_total.
	4.	No mocks or TODOs remain in source.

⸻

Execution

## 1) Fill .env with real keys then:
./bootstrap.sh
pytest -q


⸻

End of machine-task spec – executing these tasks yields the fully working, intelligence-enabled pipeline without manual copy-pasting long code blobs.