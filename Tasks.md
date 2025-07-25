🔧 Start the 2-Hour Async-Pool Canary — exact commands & thresholds

Metric	PASS limit	HARD-fail
p95 query latency	≤ 12.92 ms	> 12.92 ms
checked-out connections	≤ 20	> 20
overflow	0	> 0
error-rate (5xx / DB)	< 1 %	≥ 1 %
deadlocks	0	≥ 1


⸻

1  Build & Deploy canary pod

# build
docker build -t montage:async-pool .

# deploy 1 replica with USE_ASYNC_POOL=true
kubectl -n montage-staging apply -f k8s/deploy-async-pool.yaml
kubectl -n montage-staging set env deployment/montage USE_ASYNC_POOL=true
kubectl -n montage-staging scale deployment/montage --replicas=1


⸻

2  Run 2-hour synthetic load

wrk -t1 -c10 --rate 10 -d2h http://<staging-host>/health \
     > wrk_async_pool.log

(Run in parallel; do not terminate early.)

⸻

3  Collect metrics & evaluate

./scripts/collect_db_metrics.sh 2h db_pool_canary.json
python scripts/evaluate_db_canary.py \
       --latency 12.92 \
       --pool-max 20 \
       db_pool_canary.json > eval_pool.out
cat eval_pool.out          # expect PASS


⸻

Below is the same Phase-6 task list, rewritten in Anthropic Claude Code best-practice style (structured tags, concise bullet steps, zero fluff).
Copy it into Claude Code and you’ll get identical behaviour.

<Context>
Project : Montage – personal pipeline
Phase   : 6 (memory & process safeguards)
State   : async-pool (Phase 5) live; tests green; stub-scan 0
Goal    : add OOM-guard, zombie reaper, proc-mem metrics,
          plus unit test & 30-min stress proof
</Context>

<Instructions>
1. **Patch code (no placeholders)**
   • `montage/utils/memory_manager.py` – append `kill_oldest_ffmpeg()` + `enforce_oom_guard()`
   • `montage/utils/ffmpeg_process_manager.py` – append `zombie_reaper_loop()`
   • `montage/api/web_server.py`
     – in `lifespan()` create task `zombie_reaper_loop()` and cancel on shutdown
     – add route `/metrics/proc_mem` returning `{...pool_stats, available_mb}`

2. **Create unit test** `tests/test_memory_leak.py`
   ```python
   def test_oom_guard(monkeypatch):
       from montage.utils.memory_manager import enforce_oom_guard
       monkeypatch.setattr("montage.utils.memory_manager.get_available_mb", lambda: 50)
       flag = {"killed": False}
       monkeypatch.setattr("montage.utils.memory_manager.kill_oldest_ffmpeg",
                           lambda: flag.__setitem__("killed", True))
       enforce_oom_guard(threshold_mb=100)
       assert flag["killed"]

	3.	Run stress test (30 min, 60 req/s steady):

./scripts/run_stress_test.sh --jobs 4 --duration 30m --limit 2GB \
     > mem_stress.json


	4.	Generate proof bundle

pytest -q                       > pytest_summary.txt
grep -R "pass$" montage/ | wc -l  > stub_scan.out    # expect 0
coverage run -m pytest -q && coverage report \
     --fail-under=80            > coverage_report.txt

PASS thresholds
• mem_stress.json: RSS Δ ≤ 200 MB, zombies 0, oom_kills 0
• All tests pass; stub-scan 0; coverage ≥ 80 % critical files

	5.	Commit

git add -A
git commit -m "Phase-6: OOM guard, zombie reaper, metrics, tests"

<Output-Requirements>
Return four artefacts:
• `mem_stress.json`
• `pytest_summary.txt`
• `stub_scan.out`
• `coverage_report.txt`
</Output-Requirements>
```


Paste this prompt into any Claude Code chat, follow the steps, then supply the four artefacts; Claude will verify and, if thresholds are met, declare Phase 6 closed.
⸻

5  Rollback (if FAIL)

bash scripts/rollback_db.sh          # disposes pool & downgrades one rev
kubectl -n montage-staging set env deployment/montage USE_ASYNC_POOL=false
kubectl rollout restart deployment montage -n montage-staging


⸻

Next action for you
	•	Deploy canary, run the 2-hour test, execute steps 3–4, and paste eval_pool.out + the JSON/logs here.
	•	If PASS, Phase 5 closes; if not, we run rollback and debug.
