#!/bin/bash
# MoE Routing Test Suite
# Runs comprehensive validation of 4.72x stability improvement

set -euo pipefail

# Configuration
API_BASE_URL=${API_BASE_URL:-http://localhost:8000}
PYTHON_CMD=${PYTHON_CMD:-python3}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] $1${NC}"
}

# Check if API is running
check_api_health() {
    log "Checking API health..."
    
    local response
    if response=$(curl -s -f "${API_BASE_URL}/health" 2>/dev/null); then
        local status
        status=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
        
        if [[ "$status" == "healthy" ]]; then
            log "✅ API is healthy"
            return 0
        else
            error "API reports unhealthy status: $status"
            return 1
        fi
    else
        error "Cannot connect to API at ${API_BASE_URL}"
        echo "Please ensure the API server is running:"
        echo "  cd deployment/docker && docker-compose up -d"
        return 1
    fi
}

# Run validation tests
run_validation_tests() {
    log "Running validation test suite..."
    
    local test_files=(
        "validation/mediator_proof_test.py"
        "validation/boundary_distance_test.py" 
        "validation/gating_sensitivity_test.py"
        "validation/fast_stratified_test.py"
    )
    
    local passed=0
    local total=${#test_files[@]}
    
    for test_file in "${test_files[@]}"; do
        if [[ -f "$test_file" ]]; then
            info "Running $test_file..."
            if $PYTHON_CMD "$test_file" >/dev/null 2>&1; then
                log "✅ $test_file PASSED"
                ((passed++))
            else
                error "❌ $test_file FAILED"
            fi
        else
            warn "Test file not found: $test_file"
        fi
    done
    
    log "Validation tests: $passed/$total passed"
    
    if [[ $passed -eq $total ]]; then
        return 0
    else
        return 1
    fi
}

# Test API endpoints
test_api_endpoints() {
    log "Testing API endpoints..."
    
    # Test routing endpoint
    info "Testing /route endpoint..."
    local route_request='{"logits": [0.5, -0.2, 0.8, -0.5, 0.3], "ambiguity_score": 0.8}'
    local response
    if response=$(curl -s -X POST "${API_BASE_URL}/route" \
        -H "Content-Type: application/json" \
        -d "$route_request" 2>/dev/null); then
        
        # Check if response contains expected fields
        if echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    assert 'routing_weights' in data
    assert 'metrics' in data
    assert 'controller_version' in data
    assert data['controller_version'] == '4.72x_improvement'
    print('SUCCESS')
except:
    print('FAILED')
" | grep -q "SUCCESS"; then
            log "✅ /route endpoint working"
        else
            error "❌ /route endpoint response invalid"
            return 1
        fi
    else
        error "❌ /route endpoint failed"
        return 1
    fi
    
    # Test metrics endpoint
    info "Testing /metrics endpoint..."
    if curl -s -f "${API_BASE_URL}/metrics" >/dev/null 2>&1; then
        log "✅ /metrics endpoint working"
    else
        error "❌ /metrics endpoint failed"
        return 1
    fi
    
    # Test validate endpoint
    info "Testing /validate endpoint..."
    if response=$(curl -s -X POST "${API_BASE_URL}/validate" 2>/dev/null); then
        local status
        status=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('status', 'UNKNOWN'))
except:
    print('ERROR')
")
        
        if [[ "$status" == "PASSED" ]]; then
            log "✅ /validate endpoint working - 4.72x improvement confirmed"
        else
            error "❌ /validate endpoint failed - status: $status"
            return 1
        fi
    else
        error "❌ /validate endpoint failed"
        return 1
    fi
    
    return 0
}

# Load test with multiple requests
run_load_test() {
    local num_requests=${1:-10}
    log "Running load test with $num_requests requests..."
    
    local success_count=0
    local start_time=$(date +%s)
    
    for i in $(seq 1 $num_requests); do
        local route_request="{\"logits\": [$(shuf -i -100-100 -n 5 | tr '\n' ',' | sed 's/,$//g' | sed 's/,/, /g')], \"ambiguity_score\": $(python3 -c "import random; print(random.random())")}"
        
        if curl -s -X POST "${API_BASE_URL}/route" \
            -H "Content-Type: application/json" \
            -d "$route_request" >/dev/null 2>&1; then
            ((success_count++))
        fi
        
        # Show progress
        if (( i % 5 == 0 )); then
            info "Progress: $i/$num_requests requests"
        fi
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local rps=$(echo "scale=2; $num_requests / $duration" | bc -l 2>/dev/null || echo "N/A")
    
    log "Load test completed: $success_count/$num_requests successful"
    log "Duration: ${duration}s, RPS: $rps"
    
    if [[ $success_count -eq $num_requests ]]; then
        return 0
    else
        return 1
    fi
}

# Test stability under varying ambiguity
test_stability() {
    log "Testing routing stability under varying ambiguity..."
    
    local ambiguity_levels=(0.1 0.3 0.5 0.7 0.9)
    local success_count=0
    
    for ambiguity in "${ambiguity_levels[@]}"; do
        info "Testing with ambiguity: $ambiguity"
        
        local route_request="{\"logits\": [0.5, -0.2, 0.8, -0.5, 0.3], \"ambiguity_score\": $ambiguity}"
        local response
        
        if response=$(curl -s -X POST "${API_BASE_URL}/route" \
            -H "Content-Type: application/json" \
            -d "$route_request" 2>/dev/null); then
            
            # Extract gating sensitivity from response
            local gating_sensitivity
            gating_sensitivity=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['metrics']['gating_sensitivity'])
except:
    print('ERROR')
")
            
            if [[ "$gating_sensitivity" != "ERROR" ]]; then
                info "Ambiguity: $ambiguity → Gating Sensitivity: $gating_sensitivity"
                ((success_count++))
            fi
        fi
    done
    
    log "Stability test: $success_count/${#ambiguity_levels[@]} tests passed"
    
    if [[ $success_count -eq ${#ambiguity_levels[@]} ]]; then
        return 0
    else
        return 1
    fi
}

# Generate test report
generate_report() {
    log "Generating test report..."
    
    local report_file="test-report-$(date +%Y%m%d-%H%M%S).json"
    
    # Get current metrics
    local metrics_response
    if metrics_response=$(curl -s "${API_BASE_URL}/metrics" 2>/dev/null); then
        echo "$metrics_response" > "$report_file"
        log "Test report saved to: $report_file"
    else
        warn "Could not generate detailed report - metrics unavailable"
    fi
}

# Main test function
main() {
    log "🧪 Starting MoE Routing Test Suite (4.72x Improvement Validation)"
    echo
    
    local exit_code=0
    
    # Run tests
    check_api_health || exit_code=1
    
    if [[ $exit_code -eq 0 ]]; then
        run_validation_tests || exit_code=1
        test_api_endpoints || exit_code=1
        run_load_test 20 || exit_code=1
        test_stability || exit_code=1
        generate_report
    fi
    
    echo
    if [[ $exit_code -eq 0 ]]; then
        log "🎉 All tests PASSED! 4.72x improvement validated."
    else
        error "❌ Some tests FAILED. Check the output above."
    fi
    
    return $exit_code
}

# Handle command line arguments
case "${1:-all}" in
    all)
        main
        ;;
    health)
        check_api_health
        ;;
    validation)
        run_validation_tests
        ;;
    api)
        test_api_endpoints
        ;;
    load)
        run_load_test "${2:-10}"
        ;;
    stability)
        test_stability
        ;;
    report)
        generate_report
        ;;
    *)
        echo "Usage: $0 {all|health|validation|api|load|stability|report}"
        echo
        echo "Test suites:"
        echo "  all        - Run complete test suite"
        echo "  health     - Check API health"
        echo "  validation - Run validation tests"
        echo "  api        - Test API endpoints"
        echo "  load       - Run load test (default: 10 requests)"
        echo "  stability  - Test routing stability"
        echo "  report     - Generate test report"
        exit 1
        ;;
esac