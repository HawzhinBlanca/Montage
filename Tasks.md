✅ Phase 3-1 Accepted — Lazy proxy is live, tests and startup log verified.

⸻

Proceed to Phase 3-2 → 3-4: Canary Roll-out

Below is the exact sequence to finish Phase 3; follow it step-by-step and attach the required artifacts.

Stage	Traffic	Duration	Commands	Proof
A	5 %	2 h	1. Set USE_SETTINGS_V2=true on one staging pod.2. Deploy.3. After 2 h:  ./scripts/collect_canary_metrics.sh 2h settings_stageA.json  python scripts/evaluate_canary.py settings_stageA.json > evaluate_stageA.out	settings_stageA.json + evaluate_stageA.out must show PASS
B	25 %	2 h	Repeat collection (stage B files).	settings_stageB.json, evaluate_stageB.out (PASS)
C	100 %	2 h soak	All pods with flag enabled; collect stage C metrics.	settings_stageC.json, evaluate_stageC.out (PASS)

SLO Matrix (evaluate_canary.py enforces)
	•	p99 latency ≤ +20 % baseline
	•	5xx error-rate < 1 %
	•	ImportError_total = 0
	•	CPU ≤ 80 %
	•	MEM ≤ 85 %

Partial-fail → extend 2 h and rerun; hard-fail → set flag false and roll back.

⸻

Phase 3-5: Remove Legacy Loader

After stage C PASS:

grep -Rl "LEGACY_CONFIG_SHIM" montage/ | xargs sed -i '/LEGACY_CONFIG_SHIM/d'
git rm utils/secret_loader.py
grep -R "secret_loader" montage/ | wc -l   # should print 0
pytest -q
git commit -am "Phase-3-5: remove legacy config loader"

Add CI guard job:

- name: Enforce single config source
  run: |
    if grep -R "secret_loader" montage/ | grep -v tests; then
      echo "Legacy config detected"; exit 1; fi


⸻

Final Phase 3 Proof Bundle

settings_stageA.json
evaluate_stageA.out
settings_stageB.json
evaluate_stageB.out
settings_stageC.json
evaluate_stageC.out
startup_log_v2.txt          # shows Config source=settings_v2
stub_scan.out               # 0 secret_loader refs
pytest_summary.txt          # all tests passed
coverage_report.txt         # ≥ Phase-1 critical file thresholds

Upload this bundle; once verified, Phase 3 closes and we begin Phase 4 – API-merge ADR.

Next action: run Stage A canary, collect settings_stageA.json & evaluate_stageA.out, and share them here.