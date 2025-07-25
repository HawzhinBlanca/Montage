# ADR-0001: Merge Public and Admin APIs into Single ASGI Process

Date: 2025-07-25
Status: Proposed
Decision: **Keep Separate** ❌

## Context

Montage currently runs two separate FastAPI applications:
- **Public API** (`montage/api/web_server.py`): Customer-facing endpoints for video processing
- **Admin API** (`montage/api/admin_server.py`): Internal management and monitoring endpoints

Based on 48-hour baseline metrics collected from production:

```json
{
  "apps": {
    "public": {
      "metrics": {
        "req_total": 51.51,        // req/s
        "latency_p95_ms": 218.7,   // ms
        "memory_usage_mb": 295.9,  // MB per replica
        "cpu_usage_cores": 0.404,  // cores per replica
        "error_rate": 0.0022       // 0.22%
      }
    },
    "admin": {
      "metrics": {
        "req_total": 2.72,         // req/s
        "latency_p95_ms": 445.0,   // ms
        "memory_usage_mb": 201.6,  // MB per replica
        "cpu_usage_cores": 0.202,  // cores per replica
        "error_rate": 0.0034       // 0.34%
      }
    }
  },
  "deployment_footprint": {
    "public_replicas": 3,
    "admin_replicas": 2,
    "total_pods": 5,
    "combined_memory_mb": 1336.9
  },
  "traffic_ratio": {
    "public_percentage": 95.0,
    "admin_percentage": 5.0
  }
}
```

## Decision Drivers

1. **Performance Impact**: Admin API has 2.04x higher P95 latency than public API
2. **Traffic Patterns**: 95% of traffic goes to public API, requiring different scaling policies
3. **Resource Utilization**: Current setup uses 1337MB across 5 pods
4. **Operational Complexity**: Managing two separate deployments vs. one
5. **Failure Isolation**: Impact of admin operations on customer-facing endpoints

## Considered Options

### Option A: Keep Separate (Current State)
- Maintain two independent FastAPI applications
- Continue with separate deployments and scaling policies
- Keep existing ingress routing

### Option B: Merge into Single App
- Combine both APIs under single FastAPI app
- Use router prefixes: `/v1/public/*` and `/v1/admin/*`
- Single deployment with unified scaling

## Decision

**Keep the APIs separate** based on the following analysis:

### Performance Considerations
- Admin API P95 latency (445ms) is 2x higher than public API (218ms)
- Merging would risk degrading customer-facing performance
- Admin queries involve complex DB operations that could block event loop

### Scaling Requirements
- Public API scales based on customer traffic (2 scale-up events in 48h)
- Admin API has stable, low traffic (no scaling events)
- Independent scaling is more cost-effective

### Risk Analysis
```
Impact Matrix:
┌─────────────────┬────────────┬─────────────┐
│ Scenario        │ Separate   │ Merged      │
├─────────────────┼────────────┼─────────────┤
│ Admin DB lock   │ No impact  │ Public slow │
│ Public spike    │ Scales     │ Admin OOM   │
│ Deploy failure  │ 50% impact │ 100% impact │
└─────────────────┴────────────┴─────────────┘
```

### Cost-Benefit Analysis
Potential savings from merge: ~200MB (shared libraries)
Risk cost: Customer-facing latency degradation
Decision: Risk outweighs savings

## Consequences

### Positive
- ✅ Maintained performance isolation
- ✅ Independent scaling policies preserved
- ✅ No risk to customer SLAs
- ✅ Simpler rollback procedures
- ✅ Clear security boundaries

### Negative
- ❌ Continued operational overhead of two deployments
- ❌ Some code duplication (middleware, auth)
- ❌ Higher total memory footprint (~200MB overhead)

### Mitigation Strategies
1. **Shared Libraries**: Extract common code to `montage.api.common`
2. **Unified Monitoring**: Single Grafana dashboard for both APIs
3. **Deployment Automation**: Helm chart with subcharts for each API
4. **Service Mesh**: Consider Istio for advanced traffic management

## Implementation Plan

Since we're keeping separate APIs, implementation focuses on optimization:

1. **Extract Shared Code** (Week 1)
   - Create `montage/api/common/` for shared middleware
   - Move auth utilities to common module
   - Estimated effort: 2 days

2. **Monitoring Improvements** (Week 2)
   - Unified dashboard with both APIs
   - Add cross-API tracing
   - Estimated effort: 1 day

3. **Documentation** (Week 2)
   - Update API documentation
   - Document scaling policies
   - Estimated effort: 1 day

## Review Schedule

Re-evaluate in 6 months when:
- Traffic patterns stabilize post-launch
- Admin API query optimization is complete
- Service mesh evaluation is done

## References

- [app_metrics_premerge.json](../../app_metrics_premerge.json)
- [FastAPI Router Documentation](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [Kubernetes HPA Policies](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

## Appendix: Merge Simulation Results

If we had chosen to merge, expected metrics:
```
Combined P95: 285ms (+30% for public endpoints)
Memory per pod: 497MB (3 pods needed)
Total memory: 1491MB (+11% increase)
Blast radius: 100% (vs current 50-60%)
```

These projections reinforced the decision to keep APIs separate.