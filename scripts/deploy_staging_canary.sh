#!/usr/bin/env bash
set -euo pipefail

# Phase 2 Staging Canary Deployment Script
# Deploys dual-import migration to staging environment with 5% traffic

CANARY_VERSION="phase2-dual-import"
TRAFFIC_PERCENTAGE=5
DURATION_HOURS=2

echo "🚀 Phase 2 Staging Canary Deployment"
echo "Version: $CANARY_VERSION"
echo "Traffic: $TRAFFIC_PERCENTAGE%"
echo "Duration: ${DURATION_HOURS}h"
echo "=================================="

# Check prerequisites
echo "Checking prerequisites..."
command -v kubectl >/dev/null 2>&1 || { echo "kubectl required but not installed"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "docker required but not installed"; exit 1; }

# Build and tag canary image
echo "Building canary Docker image..."
docker build -t montage:$CANARY_VERSION .
docker tag montage:$CANARY_VERSION montage-registry/montage:$CANARY_VERSION

# Deploy canary to staging
echo "Deploying canary to staging..."
cat > canary-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: montage-canary
  namespace: montage-staging
  labels:
    app: montage
    version: canary
spec:
  replicas: 2
  selector:
    matchLabels:
      app: montage
      version: canary
  template:
    metadata:
      labels:
        app: montage
        version: canary
    spec:
      containers:
      - name: montage
        image: montage-registry/montage:$CANARY_VERSION
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "staging"
        - name: CANARY_VERSION
          value: "$CANARY_VERSION"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: montage-canary-service
  namespace: montage-staging
spec:
  selector:
    app: montage
    version: canary
  ports:
  - port: 8000
    targetPort: 8000
EOF

kubectl apply -f canary-deployment.yaml

# Configure traffic split
echo "Configuring $TRAFFIC_PERCENTAGE% traffic split..."
cat > traffic-split.yaml << EOF
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: montage-rollout
  namespace: montage-staging
spec:
  strategy:
    canary:
      canaryService: montage-canary-service
      stableService: montage-stable-service
      trafficRouting:
        istio:
          virtualService:
            name: montage-vs
          destinationRule:
            name: montage-dr
            canarySubsetName: canary  
            stableSubsetName: stable
      steps:
      - setWeight: $TRAFFIC_PERCENTAGE
      - pause: {duration: ${DURATION_HOURS}h}
      analysis:
        templates:
        - templateName: success-rate
        - templateName: response-time
        args:
        - name: service-name
          value: montage-canary-service
EOF

kubectl apply -f traffic-split.yaml

echo "✅ Canary deployment initiated"
echo "Traffic split: $TRAFFIC_PERCENTAGE% to canary"
echo "Monitoring for ${DURATION_HOURS} hours..."

# Wait for deployment to be ready
kubectl wait --for=condition=available deployment/montage-canary -n montage-staging --timeout=300s

echo "🎯 Canary is live and receiving traffic"
echo "Metrics collection will run for ${DURATION_HOURS} hours"
echo "Use 'kubectl logs -f deployment/montage-canary -n montage-staging' to monitor"

# Setup monitoring
echo "Starting metrics collection..."
./scripts/collect_canary_metrics.sh &
METRICS_PID=$!

# Record deployment info
cat > canary_deployment_info.json << EOF
{
  "deployment_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "version": "$CANARY_VERSION",
  "traffic_percentage": $TRAFFIC_PERCENTAGE,
  "duration_hours": $DURATION_HOURS,
  "namespace": "montage-staging",
  "metrics_pid": $METRICS_PID,
  "status": "active"
}
EOF

echo "📊 Deployment info saved to canary_deployment_info.json"
echo "🕒 Canary will run for ${DURATION_HOURS} hours"
echo "Monitor with: ./scripts/monitor_canary.sh"