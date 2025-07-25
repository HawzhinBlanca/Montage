#!/usr/bin/env bash
set -euo pipefail

# bootstrap env from conftest
export JWT_SECRET_KEY="test-key"
export DATABASE_URL="sqlite:///:memory:"

echo "--- Running CLI smoke test ---"
python -m montage.cli.run_pipeline --from-plan tests/assets/minimal.json -o out.mp4

echo "--- Verifying output with ffprobe ---"
ffprobe -v error -show_entries format=duration out.mp4

echo "✅ Smoke test passed"