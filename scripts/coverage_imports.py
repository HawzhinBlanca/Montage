#!/usr/bin/env python3
"""Generate coverage by importing all modules"""

import os
import sys
import coverage

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock environment
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["MAX_COST_USD"] = "1.00"
os.environ["OPENAI_API_KEY"] = "test-key"  # pragma: allowlist secret
os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["DEEPGRAM_API_KEY"] = "test-key"

# Start coverage
cov = coverage.Coverage()
cov.erase()
cov.start()

# Import all modules
modules_to_import = [
    "src.config",
    "src.cli.run_pipeline",
    "src.core.analyze_video",
    "src.core.checkpoint",
    "src.core.db",
    "src.core.errors",
    "src.core.highlight_selector",
    "src.core.metrics",
    "src.core.smart_video_editor",
    "src.core.user_success_metrics",
    "src.providers.audio_normalizer",
    "src.providers.color_converter",
    "src.providers.concat_editor",
    "src.providers.resolve_mcp",
    "src.providers.smart_crop",
    "src.providers.smart_track",
    "src.providers.speaker_diarizer",
    "src.providers.transcript_analyzer",
    "src.providers.video_processor",
    "src.utils.budget_decorator",
    "src.utils.budget_guard",
    "src.utils.cleanup_manager",
    "src.utils.ffmpeg_utils",
    "src.utils.monitoring_integration",
    "src.utils.retry_utils",
    "src.utils.secret_loader",
    "src.utils.video_validator",
]

for module in modules_to_import:
    try:
        exec(f"import {module}")
        print(f"✓ Imported {module}")
    except Exception as e:
        print(f"✗ Failed to import {module}: {e}")

# Stop coverage and save
cov.stop()
cov.save()
cov.json_report(outfile="cov.json")
cov.report()

print("\nGenerated cov.json")
