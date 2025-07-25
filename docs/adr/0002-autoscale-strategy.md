# ADR-0002: Auto-Scale Strategy for Montage Personal Pipeline

## Status
Accepted

## Context
Montage needs to handle variable creator workloads efficiently:
- Peak usage during content creation sessions (evenings/weekends)
- Idle periods between uploads
- Burst processing when multiple videos queue up
- Cost optimization for personal/small team use

Current baseline metrics:
- P95 latency: 74.65ms (API), ~9s (video processing)
- Throughput: 850 req/min sustained
- CPU usage: 36.4% average, 78.9% peak
- Memory: ~1.2GB RSS with 8-10% growth per session

## Decision
Implement Horizontal Pod Autoscaler (HPA) with custom metrics:

### Scaling Configuration
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: montage-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: montage
  minReplicas: 1
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 65  # Scale up at 65% CPU
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70  # Scale up at 70% memory
  - type: Pods
    pods:
      metric:
        name: p95_latency_ms
      target:
        type: AverageValue
        averageValue: "80"  # Scale if P95 > 80ms
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60  # Quick scale-up
      policies:
      - type: Percent
        value: 100  # Double pods
        periodSeconds: 60
      - type: Pods
        value: 2    # Add 2 pods max
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # Slow scale-down
      policies:
      - type: Percent
        value: 25   # Remove 25% of pods
        periodSeconds: 60
```

### Scaling Triggers
1. **CPU-based**: Scale up when average CPU > 65%
2. **Memory-based**: Scale up when average memory > 70%
3. **Latency-based**: Scale up when P95 latency > 80ms
4. **Queue-based**: Scale up when Celery queue depth > 5 jobs

### Pod Distribution Strategy
- **1 pod**: Idle/overnight (baseline)
- **2-3 pods**: Normal usage (1-2 concurrent users)
- **4-6 pods**: Peak hours (multiple uploads)
- **7-8 pods**: Burst processing (batch jobs)

### Resource Requests/Limits
```yaml
resources:
  requests:
    cpu: "500m"      # 0.5 CPU cores
    memory: "1Gi"    # 1GB RAM
  limits:
    cpu: "2000m"     # 2 CPU cores
    memory: "2Gi"    # 2GB RAM
```

## Consequences

### Positive
- **Cost Efficient**: Scales to zero during idle
- **Performance**: Maintains P95 < 85ms under load
- **Reliability**: No single point of failure
- **Flexibility**: Handles burst workloads

### Negative
- **Cold Starts**: First request after scale-down slower
- **Complexity**: Requires metrics server + HPA
- **State**: Must use external Redis/DB for sessions

### Mitigation
- Keep minimum 1 replica for personal use
- Pre-warm pods during expected peak times
- Use init containers for fast startup

## Implementation Notes

### Phase 1: Basic HPA (CPU/Memory)
```bash
kubectl autoscale deployment montage \
  --min=1 --max=4 \
  --cpu-percent=65
```

### Phase 2: Custom Metrics
- Deploy Prometheus + metrics-server
- Expose P95 latency via `/metrics` endpoint
- Configure HPA with custom metrics

### Phase 3: Predictive Scaling
- Analyze usage patterns
- Pre-scale for regular peak times
- Integrate with calendar API for scheduled events

## Monitoring
Track these KPIs:
- Scale-up/down frequency
- Pod startup time
- Request distribution across pods
- Cost per processed video

## References
- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Custom Metrics API](https://github.com/kubernetes/metrics)
- Phase 7 baseline performance: perf_base.json
