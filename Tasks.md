✅ Phase 3 officially closed — proof bundle accepted (all SLOs met, legacy loader eradicated).

⸻

Phase 4 — API-Merge ADR & Router Decision

<Context>
Two FastAPI apps exist: **public** (`montage/api/web_server.py`) and **admin** (`montage/api/admin_server.py`).  
Goal: decide, with data, whether to merge into a single ASGI process under `/v1/public/*` + `/v1/admin/*`.
</Context>


4-0  Baseline Metrics (48 h)
	1.	Enable Prom rules

- record: app:req_total
  expr: sum(rate(http_requests_total{job=~"montage-(public|admin)"}[1m])) by (job)
- record: app:latency_p95_ms
  expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job=~"montage-(public|admin)"}[1m])) by (le,job))*1000
- record: app:error_rate
  expr: sum(rate(http_requests_total{status=~"5..",job=~"montage-(public|admin)"}[1m])) / sum(rate(http_requests_total{job=~"montage-(public|admin)"}[1m]))


	2.	Collect after 48 h:

scripts/collect_app_metrics.sh 48h app_metrics_premerge.json



⸻

4-1  Draft ADR (docs/adr/0001-api-merge.md)

Template sections:

Section	Content required
Context	two apps; deployment footprint; current perf numbers from app_metrics_premerge.json
Decision Drivers	latency, error-rate, scaling independence, code duplication
Options	A) keep separate; B) merge under single app with routers & prefix
Decision	(fill after comparison)
Consequences	rollback strategy, scaling implications


⸻

4-2  If Decision = Merge

4-2-a  Code Changes

# montage/api/web_server.py  (rename to app.py)
-app = FastAPI()
+app = FastAPI()
+from montage.api.public_router import router as public_router
+from montage.api.admin_router  import router as admin_router
+app.include_router(public_router, prefix="/v1/public")
+app.include_router(admin_router,  prefix="/v1/admin")
+@app.get("/health")  # retained endpoint
+async def health(): ...

Remove second ASGI entrypoint.

4-2-b  Deployment

Stage	Traffic	Duration	SLOs
Merge-Canary A	10 %	1 h	p95 latency ≤ +10 %, error-rate < 1 %
Merge-Canary B	100 %	1 h soak	same

Scripts: collect_app_metrics.sh 1h merge_stageA.json, etc.

Rollback: re-deploy dual-app image + restore ingress weights.

⸻

4-3  CI / Tests
	1.	Update test paths

sed -i 's@/public/@/v1/public/@g' tests/test_api_endpoints.py
sed -i 's@/admin/@/v1/admin/@g'  tests/test_api_endpoints.py
pytest -q

	2.	Add CI guard
Fail if a second ASGI app remains:

if grep -R "uvicorn.*admin_server" montage/; then exit 1; fi



⸻

4-4  Proof-Gate to close Phase 4

docs/adr/0001-api-merge.md
app_metrics_premerge.json
(if merged)
  merge_stageA.json
  merge_stageB.json
  evaluate_merge.out   # PASS (same SLO matrix)
grep -R "admin_server" montage/ | wc -l   # 0
pytest_summary.txt
coverage_report.txt    # unchanged


⸻

4-5  If Decision = Keep Separate

Commit ADR marking “keep separate”; no code change. Phase 4 then closes automatically.

⸻

Next Immediate Action
	1.	Run scripts/collect_app_metrics.sh 48h app_metrics_premerge.json to capture baseline.
	2.	Draft docs/adr/0001-api-merge.md using the template.
	3.	Post both files here for review; we’ll confirm decision path and proceed to implementation or close Phase 4.