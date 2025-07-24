#!/usr/bin/env python3
"""Generate stubs_report.md listing files that still contain obvious skeleton code.
Criteria:
  * Line containing only `pass` (after stripping) inside project code (not tests).
  * Line containing `# TODO` or `raise NotImplementedError`.
Exit status is 1 if any critical stubs found.
"""
import pathlib
import re
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PATTERN = re.compile(r"^\s*(pass|raise\s+NotImplementedError|#\s*TODO)")

ignore_dirs = {"tests", "htmlcov", ".git", ".venv"}

critical = []
for py_file in PROJECT_ROOT.rglob("*.py"):
    if any(part in ignore_dirs for part in py_file.parts):
        continue
    rel = py_file.relative_to(PROJECT_ROOT)
    try:
        for lineno, line in enumerate(py_file.read_text("utf8").splitlines(), 1):
            if PATTERN.match(line):
                critical.append(f"{rel}:{lineno}: {line.strip()}")
    except Exception:
        continue

REPORT = PROJECT_ROOT / "stubs_report.md"
if critical:
    REPORT.write_text("# Stub Report\n\n" + "\n".join(critical) + "\n")
    print("Stubs found – see stubs_report.md", file=sys.stderr)
    for line in critical:
        print(line)
    sys.exit(1)
else:
    if REPORT.exists():
        REPORT.unlink()
    print("No stubs found") 