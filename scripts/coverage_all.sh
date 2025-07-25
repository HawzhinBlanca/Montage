#!/usr/bin/env bash
set -e
coverage erase
coverage run -m pytest -q
coverage run --append -m src.cli.run_pipeline tests/data/short.mp4
coverage run --append -m src.cli.run_pipeline /no/such/file.mp4 || true
coverage run --append -m src.cli.run_pipeline tests/data/corrupt.mp4 || true
coverage run --append -m src.cli.run_pipeline tests/data/long.mp4 &
PID=$!; sleep 2; kill -TERM $PID || true; wait || true
ulimit -m 500000          # 500 MB
coverage run --append -m src.cli.run_pipeline tests/data/huge_metadata.mp4 || true
coverage json -o cov.json
coverage report --fail-under=95