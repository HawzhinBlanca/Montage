Below is the complete, current master plan for Montage, incorporating every revision to date. All future status updates will reference this full plan so you never have to ask again.

⸻

🗺️ Montage End-to-End Roadmap (Claude-Code Structured)

Phase	Goal	Key Tasks	Proof-Gate
0	CLI & CI Unblock ✅	lazy-load heavy imports, pytest bootstrap	CI green
1	Critical-Path Tests ✅	90 % cov on pipeline & video_processor, ≥ 75 % on memory_manager, stub-scan 0	coverage + stub-scan reports
2	Dual-Import Migration (in progress)	replace all sys.path.append; 2 h canary (latency, error-rate, ImportError, CPU < 80 %, MEM < 85 %)	① grep "sys.path.append" → 0 ② pytest green ③ canary_metrics.json + evaluate_canary.out = PASS
3	Config Unification	settings_v2.py (Pydantic), USE_SETTINGS_V2 flag, 24 h canary; delete legacy loaders	SLOs green, startup log Config source=settings_v2
4	API Merge ADR	collect per-app metrics 48 h → write docs/adr/0001-api-merge.md; if approved, merge routers /v1/public/* & /v1/admin/*	ADR committed, API tests green
5	DB Versioning & Pooling	add Alembic async migrations, route prefix /v1/…, async read-replica pool @5 % traffic	CI upgrade/downgrade pass; canary metrics
6	Memory & Process Safeguards	cgroup-aware get_available_mb, async reap_zombies, 3×600 MB stress test under 2 GB	stress_metrics.json (no OOM), Prom logs
7	Performance Guard	capture baseline → CI perf job (fail if FPS ↓ > 10 % or RSS ↑ > 15 %)	perf report artifact each run

Rollback path documented per phase; successively gated—no next-phase merge until proof-gate satisfied.

⸻

📌 Phase 2 Detailed Checklist (active)
	1.	Patch applied: dual-import for resolve_mcp.py (legacy hack commented).
	2.	Unit tests green (pytest -q).
	3.	2-hour canary (5 % traffic in montage-staging).
	•	Metrics collected by scripts/collect_canary_metrics.sh.
	•	Evaluated by scripts/evaluate_canary.py against SLO matrix:p99 latency ≤ +20 %, 5xx < 1 %, ImportError = 0, CPU ≤ 80 %, MEM ≤ 85 %.
	4.	If PASS → delete LEGACY_IMPORT_HACK lines, stub-scan ⇒ 0.
	5.	CI scan job blocks future hacks.

Phase-2 Proof Bundle (required to close)

canary_metrics.json
evaluate_canary.out   # PASS / PARTIAL / FAIL
perf_baseline.json    # baseline fps & RSS
pytest_summary.txt    # all tests passed
stub_scan.out         # 0


⸻

🧪 Coverage Roadmap (post-Phase 2 merge)

Week	Global coverage target*	Focus modules	Mock strategy
W-0	baseline + 5 pp	AudioNormalizer, ROVER	mock subprocess.run (loudnorm JSON)
W-1	baseline + 15 pp	StoryBeats, MemoryManager	mock HTTP→Claude/Gemini; psutil fixture
W-2	baseline + 35 pp	VideoEffects, infra	fake FFmpeg binary, filter-error checks
W-3	≥ 80 %	E2E pipeline, DB	synthetic videos, SQLite in-mem

*Baseline = current global coverage stored in coverage_base.json.

⸻

📊 Scripts (latest versions)

scripts/capture_perf_baseline.sh

#!/usr/bin/env bash
set -euo pipefail
VIDEO=tests/assets/minimal.mp4
LOG=$(mktemp)

timeout 10s ffmpeg -hide_banner -i "$VIDEO" -f null - 2>"$LOG" &
FFMPEG_PID=$!
sleep 2
RSS_KB=$(grep VmRSS /proc/$FFMPEG_PID/status 2>/dev/null | awk '{print $2}' || echo 0)
wait $FFMPEG_PID || true

FPS=$(grep -oE "fps=[0-9.]+" "$LOG" | tail -1 | cut -d= -f2)
jq -n --arg fps "${FPS:-0}" --arg rss "$(awk "BEGIN{print ${RSS_KB:-0}/1024}")" \
   '{fps:$fps|tonumber,rss_mb:$rss|tonumber}' > perf_baseline.json

scripts/collect_canary_metrics.sh (excerpt)

q(){ curl -sG -H "Authorization: Bearer $PROM_TOKEN" \
         --data-urlencode "query=$1" \
         --data-urlencode "start=$(date -d "-$DUR" +%s)" \
         --data-urlencode "end=$(date +%s)" \
         "${PROM_PROM_URL}/api/v1/query_range"; }

# collects p99 latency, error rate, ImportError, CPU, MEM

scripts/evaluate_canary.py

Evaluates metrics JSON → writes evaluate_canary.out (PASS / PARTIAL / FAIL).

⸻

🛠 Improved FFmpeg Mock (tests/conftest.py)

def _fake_run(cmd, *_, **__):
    j = " ".join(cmd) if isinstance(cmd, (list, tuple)) else cmd
    if "loudnorm" in j:
        return subprocess.CompletedProcess(cmd, 0,
          '{"input_i":-20,"input_tp":-1.5,"input_lra":6,"input_thresh":-30,"target_offset":0}', '')
    if "-filter_complex" in j:
        raise subprocess.CalledProcessError(1, cmd, "Invalid filter")
    if " -t " in j:
        return subprocess.CompletedProcess(cmd, 0, '', '')
    return subprocess.CompletedProcess(cmd, 0, '', '')


⸻

🔄 Rollback Summary

Phase	Switch / Revert
2	revert commit or redeploy with previous image
3	set USE_SETTINGS_V2=false, revert Pydantic commit
4	keep separate ASGI apps
5	alembic downgrade -1, route prefix rollback
6	disable zombie-reaper task
7	disable perf CI job


⸻

Next Immediate Action
	1.	Run fixed perf-baseline script → commit perf_baseline.json.
	2.	Deploy dual-import canary for 2 h; collect metrics; run evaluator.
	3.	Submit proof bundle; Phase 2 will then close and Phase 3 work may begin.

You now have the full, gap-free plan. Every future update will reference this document.