#!/bin/bash
# Deployment script for AI Video Processing Pipeline

set -e

# Configuration
ENVIRONMENT=${1:-staging}
CLUSTER_NAME="video-pipeline"
REGION="us-west-2"
NAMESPACE="video-pipeline"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Validate environment
validate_environment() {
    log "Validating environment: $ENVIRONMENT"
    
    if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
        error "Invalid environment. Must be 'staging' or 'production'"
    fi
    
    # Check required tools
    command -v aws >/dev/null 2>&1 || error "AWS CLI is required"
    command -v kubectl >/dev/null 2>&1 || error "kubectl is required"
    command -v helm >/dev/null 2>&1 || error "Helm is required"
    command -v terraform >/dev/null 2>&1 || error "Terraform is required"
}

# Setup AWS and Kubernetes configuration
setup_config() {
    log "Setting up AWS and Kubernetes configuration"
    
    # Update kubeconfig
    if [ "$ENVIRONMENT" = "production" ]; then
        aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME
    else
        aws eks update-kubeconfig --region $REGION --name "${CLUSTER_NAME}-${ENVIRONMENT}"
    fi
    
    # Verify cluster access
    kubectl cluster-info || error "Failed to connect to Kubernetes cluster"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log "Deploying infrastructure with Terraform"
    
    cd infrastructure/terraform
    
    # Initialize Terraform
    terraform init
    
    # Select workspace
    terraform workspace select $ENVIRONMENT || terraform workspace new $ENVIRONMENT
    
    # Plan deployment
    terraform plan -var="environment=$ENVIRONMENT" -out=tfplan
    
    # Apply infrastructure
    log "Applying Terraform changes..."
    terraform apply tfplan
    
    # Save outputs
    terraform output -json > terraform-outputs.json
    
    cd ../..
}

# Deploy Kubernetes resources
deploy_kubernetes() {
    log "Deploying Kubernetes resources"
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply base resources
    kubectl apply -f infrastructure/k8s/namespace.yaml
    kubectl apply -f infrastructure/k8s/configmap.yaml
    
    # Handle secrets
    if kubectl get secret video-pipeline-secrets -n $NAMESPACE >/dev/null 2>&1; then
        warn "Secrets already exist, skipping creation"
    else
        log "Creating secrets (you'll need to update these manually)"
        kubectl apply -f infrastructure/k8s/configmap.yaml
    fi
    
    # Deploy applications
    kubectl apply -f infrastructure/k8s/deployment.yaml
    kubectl apply -f infrastructure/k8s/service.yaml
    kubectl apply -f infrastructure/k8s/ingress.yaml
    kubectl apply -f infrastructure/k8s/hpa.yaml
    
    # Wait for deployments
    log "Waiting for deployments to be ready..."
    kubectl rollout status deployment/video-pipeline-api -n $NAMESPACE --timeout=600s
    kubectl rollout status deployment/video-pipeline-worker -n $NAMESPACE --timeout=600s
    
    log "Kubernetes deployment completed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack"
    
    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Deploy Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values infrastructure/helm/prometheus-values.yaml \
        --wait
    
    # Deploy custom Grafana dashboards
    kubectl apply -f monitoring/grafana/dashboards/ -n monitoring
    
    log "Monitoring stack deployed"
}

# Run health checks
health_checks() {
    log "Running health checks"
    
    # Wait a bit for everything to stabilize
    sleep 30
    
    # Check pod status
    kubectl get pods -n $NAMESPACE
    
    # Check services
    kubectl get services -n $NAMESPACE
    
    # Test API health endpoint
    API_URL=$(kubectl get ingress video-pipeline-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    if [ -n "$API_URL" ]; then
        log "Testing API health at $API_URL"
        
        # Wait for DNS propagation
        sleep 60
        
        if curl -f "https://$API_URL/health" >/dev/null 2>&1; then
            log "✅ API health check passed"
        else
            warn "❌ API health check failed"
        fi
        
        # Test metrics endpoint
        if curl -f "https://metrics.$API_URL/metrics" >/dev/null 2>&1; then
            log "✅ Metrics endpoint accessible"
        else
            warn "❌ Metrics endpoint failed"
        fi
    else
        warn "Could not determine API URL"
    fi
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        error "Deployment failed. Check logs above."
    fi
}

trap cleanup EXIT

# Main deployment flow
main() {
    log "Starting deployment to $ENVIRONMENT environment"
    
    validate_environment
    setup_config
    
    if [ "$ENVIRONMENT" = "production" ]; then
        read -p "⚠️  You are deploying to PRODUCTION. Are you sure? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            error "Deployment cancelled"
        fi
    fi
    
    # Deploy infrastructure first
    deploy_infrastructure
    
    # Deploy Kubernetes resources
    deploy_kubernetes
    
    # Deploy monitoring
    deploy_monitoring
    
    # Run health checks
    health_checks
    
    log "🎉 Deployment to $ENVIRONMENT completed successfully!"
    
    if [ "$ENVIRONMENT" = "staging" ]; then
        log "Staging URL: https://staging.video-pipeline.example.com"
    else
        log "Production URL: https://api.video-pipeline.example.com"
    fi
    
    log "Monitoring: https://grafana.video-pipeline.example.com"
}

# Run main function
main "$@"