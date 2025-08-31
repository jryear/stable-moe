#!/bin/bash
# MoE Routing Production Deployment Script
# Deploys the 4.72x stability improvement controller

set -euo pipefail

# Configuration
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-production}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-latest}
API_PORT=${API_PORT:-8000}
DASHBOARD_PORT=${DASHBOARD_PORT:-8501}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    command -v docker >/dev/null 2>&1 || error "Docker is not installed"
    command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is not installed"
    command -v curl >/dev/null 2>&1 || error "curl is not installed"
    
    # Check Docker daemon
    docker info >/dev/null 2>&1 || error "Docker daemon is not running"
    
    log "✅ All dependencies are available"
}

# Pre-deployment validation
validate_configuration() {
    log "Validating deployment configuration..."
    
    # Check if ports are available
    if lsof -Pi :${API_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
        warn "Port ${API_PORT} is already in use"
    fi
    
    if lsof -Pi :${DASHBOARD_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
        warn "Port ${DASHBOARD_PORT} is already in use"
    fi
    
    # Check if required files exist
    [ -f "requirements.txt" ] || error "requirements.txt not found"
    [ -f "src/api/server.py" ] || error "API server not found"
    [ -f "src/monitoring/dashboard.py" ] || error "Dashboard not found"
    
    log "✅ Configuration validated"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # Build API image
    log "Building MoE Routing API image..."
    docker build -f deployment/docker/Dockerfile -t moe-routing-api:${DOCKER_IMAGE_TAG} .
    
    # Build Dashboard image
    log "Building Dashboard image..."
    docker build -f deployment/docker/Dockerfile.dashboard -t moe-routing-dashboard:${DOCKER_IMAGE_TAG} .
    
    log "✅ Docker images built successfully"
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    # Create necessary directories
    mkdir -p deployment/docker/logs
    mkdir -p deployment/docker/config
    
    # Start services
    cd deployment/docker
    docker-compose up -d
    cd ../..
    
    log "✅ Services deployed"
}

# Health checks
wait_for_services() {
    log "Waiting for services to start..."
    
    # Wait for API
    local api_ready=false
    local dashboard_ready=false
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        attempt=$((attempt + 1))
        
        # Check API health
        if ! $api_ready && curl -f http://localhost:${API_PORT}/health >/dev/null 2>&1; then
            log "✅ API is healthy"
            api_ready=true
        fi
        
        # Check Dashboard health
        if ! $dashboard_ready && curl -f http://localhost:${DASHBOARD_PORT} >/dev/null 2>&1; then
            log "✅ Dashboard is accessible"
            dashboard_ready=true
        fi
        
        if $api_ready && $dashboard_ready; then
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            error "Services failed to start within timeout"
        fi
        
        sleep 10
    done
}

# Validate 4.72x improvement
validate_improvement() {
    log "Validating 4.72x stability improvement..."
    
    local validation_result
    validation_result=$(curl -s -X POST http://localhost:${API_PORT}/validate)
    
    local status
    status=$(echo "$validation_result" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
    
    if [[ "$status" == "PASSED" ]]; then
        log "✅ 4.72x improvement validation PASSED"
        
        local improvement_factor
        improvement_factor=$(echo "$validation_result" | python3 -c "import sys, json; print(json.load(sys.stdin)['validation_results']['improvement_factor'])")
        log "Measured improvement factor: ${improvement_factor}x"
    else
        error "4.72x improvement validation FAILED"
    fi
}

# Show deployment summary
show_summary() {
    log "Deployment completed successfully!"
    echo
    echo "🎯 MoE Routing Stability System (4.72x Improvement) is now running:"
    echo
    echo "  📡 API Server:     http://localhost:${API_PORT}"
    echo "  📊 Dashboard:      http://localhost:${DASHBOARD_PORT}"
    echo "  🏥 Health Check:   http://localhost:${API_PORT}/health"
    echo "  📈 Metrics:        http://localhost:${API_PORT}/metrics"
    echo
    echo "Key endpoints:"
    echo "  POST /route        - Apply 4.72x stability routing"
    echo "  POST /validate     - Validate improvement factor"
    echo "  GET  /recent-metrics - Get routing stability data"
    echo
    echo "To stop services: cd deployment/docker && docker-compose down"
    echo "To view logs:     cd deployment/docker && docker-compose logs -f"
}

# Cleanup on failure
cleanup() {
    if [[ $? -ne 0 ]]; then
        error "Deployment failed. Cleaning up..."
        cd deployment/docker 2>/dev/null && docker-compose down 2>/dev/null || true
    fi
}

trap cleanup EXIT

# Main deployment flow
main() {
    log "Starting MoE Routing Production Deployment (4.72x Improvement)"
    
    check_dependencies
    validate_configuration
    build_images
    deploy_services
    wait_for_services
    validate_improvement
    show_summary
    
    log "🚀 Deployment successful!"
}

# Handle command line arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    stop)
        log "Stopping services..."
        cd deployment/docker && docker-compose down
        log "✅ Services stopped"
        ;;
    status)
        log "Checking service status..."
        cd deployment/docker && docker-compose ps
        ;;
    logs)
        log "Showing service logs..."
        cd deployment/docker && docker-compose logs -f
        ;;
    clean)
        log "Cleaning up deployment..."
        cd deployment/docker && docker-compose down -v
        docker rmi moe-routing-api:${DOCKER_IMAGE_TAG} 2>/dev/null || true
        docker rmi moe-routing-dashboard:${DOCKER_IMAGE_TAG} 2>/dev/null || true
        log "✅ Cleanup completed"
        ;;
    validate)
        validate_improvement
        ;;
    *)
        echo "Usage: $0 {deploy|stop|status|logs|clean|validate}"
        echo
        echo "Commands:"
        echo "  deploy   - Deploy the MoE routing system"
        echo "  stop     - Stop all services"
        echo "  status   - Show service status"
        echo "  logs     - Show service logs"
        echo "  clean    - Remove all containers and images"
        echo "  validate - Test 4.72x improvement validation"
        exit 1
        ;;
esac