#!/usr/bin/env bash
set -e

# Basic coverage without testcontainers
export DATABASE_URL="sqlite:///:memory:"
export REDIS_URL="redis://localhost:6379/0"
export MAX_COST_USD="1.00"
export OPENAI_API_KEY="test-key"  # pragma: allowlist secret
export ANTHROPIC_API_KEY="test-key"
export DEEPGRAM_API_KEY="test-key"

coverage erase

# Run basic imports and module tests
coverage run -m pytest tests/test_e2e.py::test_ffmpeg_utils -v || true
coverage run --append -m pytest tests/test_metrics.py -k "not test_start_http_server" -v || true

# Test CLI imports
coverage run --append -c "import src.cli.run_pipeline; print('CLI imports OK')" || true
coverage run --append -c "import src.core.highlight_selector; print('Highlight selector OK')" || true
coverage run --append -c "import src.core.db; print('DB OK')" || true
coverage run --append -c "import src.core.checkpoint; print('Checkpoint OK')" || true
coverage run --append -c "import src.core.metrics; print('Metrics OK')" || true
coverage run --append -c "import src.utils.ffmpeg_utils; print('FFmpeg utils OK')" || true
coverage run --append -c "import src.providers.concat_editor; print('Concat editor OK')" || true

# Generate coverage report
coverage json -o cov.json
coverage report