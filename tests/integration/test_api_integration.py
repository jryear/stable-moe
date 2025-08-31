"""
API integration tests with automated testing scenarios
Tests full API functionality and automated workflows
"""

import pytest
import requests
import asyncio
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import subprocess
import signal
import os

from fastapi.testclient import TestClient
from src.api.server import app


class TestAPIIntegration:
    """Full API integration test suite"""
    
    @pytest.fixture(scope="class")
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    def test_api_health_check(self, client):
        """Test API health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "controller_status" in data
        assert "improvement_factor" in data
        assert data["improvement_factor"] == "4.72x validated"
    
    def test_routing_endpoint_basic(self, client):
        """Test basic routing functionality"""
        request_data = {
            "logits": [0.5, -0.2, 0.8, -0.5, 0.3],
            "ambiguity_score": 0.8,
            "request_id": "test_basic"
        }
        
        response = client.post("/route", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "routing_weights" in data
        assert "metrics" in data
        assert "request_id" in data
        assert "controller_version" in data
        assert data["controller_version"] == "4.72x_improvement"
        
        # Validate routing weights
        weights = data["routing_weights"]
        assert len(weights) == len(request_data["logits"])
        assert abs(sum(weights) - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
        
        # Validate metrics
        metrics = data["metrics"]
        required_metrics = [
            "gating_sensitivity", "winner_flip_rate", "boundary_distance",
            "routing_entropy", "latency_ms", "clarity_score", "beta", "lambda"
        ]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_routing_validation_errors(self, client):
        """Test routing endpoint validation"""
        # Missing required fields
        response = client.post("/route", json={})
        assert response.status_code == 422
        
        # Invalid logits (too few)
        response = client.post("/route", json={
            "logits": [0.5],
            "ambiguity_score": 0.5
        })
        assert response.status_code == 422
        
        # Invalid ambiguity score (out of range)
        response = client.post("/route", json={
            "logits": [0.5, -0.2, 0.8, -0.5, 0.3],
            "ambiguity_score": 1.5
        })
        assert response.status_code == 422
        
        # Invalid ambiguity score (negative)
        response = client.post("/route", json={
            "logits": [0.5, -0.2, 0.8, -0.5, 0.3],
            "ambiguity_score": -0.1
        })
        assert response.status_code == 422
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        # Make some routing requests first
        for i in range(5):
            client.post("/route", json={
                "logits": [0.5, -0.2, 0.8, -0.5, 0.3],
                "ambiguity_score": 0.5 + i * 0.1,
                "request_id": f"metrics_test_{i}"
            })
        
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        expected_fields = [
            "requests_processed", "avg_gating_sensitivity", "avg_winner_flip_rate",
            "avg_latency_ms", "status", "uptime_seconds"
        ]
        
        for field in expected_fields:
            assert field in data
    
    def test_recent_metrics_endpoint(self, client):
        """Test recent metrics endpoint"""
        # Make some routing requests
        for i in range(10):
            client.post("/route", json={
                "logits": [0.5, -0.2, 0.8, -0.5, 0.3],
                "ambiguity_score": 0.5,
                "request_id": f"recent_test_{i}"
            })
        
        response = client.get("/recent-metrics?limit=5")
        assert response.status_code == 200
        
        data = response.json()
        assert "metrics" in data
        assert "count" in data
        assert len(data["metrics"]) <= 5
        
        # Check metric structure
        if data["metrics"]:
            metric = data["metrics"][0]
            required_fields = [
                "gating_sensitivity", "winner_flip_rate", "boundary_distance",
                "routing_entropy", "latency_ms", "timestamp", "clarity_score"
            ]
            for field in required_fields:
                assert field in metric
    
    def test_validate_endpoint(self, client):
        """Test validation endpoint"""
        response = client.post("/validate")
        assert response.status_code == 200
        
        data = response.json()
        assert "validation_results" in data
        assert "status" in data
        assert "message" in data
        
        validation = data["validation_results"]
        assert "improvement_factor" in validation
        assert "meets_target" in validation
        assert "baseline_sensitivity" in validation
        assert "controlled_sensitivity" in validation
        
        # Should pass validation
        assert data["status"] == "PASSED"
        assert validation["meets_target"] is True
        assert validation["improvement_factor"] >= 4.0
    
    def test_reset_endpoint(self, client):
        """Test controller reset endpoint"""
        # Build up some state first
        for i in range(5):
            client.post("/route", json={
                "logits": [0.5, -0.2, 0.8, -0.5, 0.3],
                "ambiguity_score": 0.5,
                "request_id": f"reset_test_{i}"
            })
        
        # Reset controller
        response = client.post("/reset")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "controller_reset"
        
        # Verify state was reset by checking metrics
        metrics_response = client.get("/metrics")
        metrics_data = metrics_response.json()
        assert metrics_data["requests_processed"] == 0


class TestAutomatedWorkflows:
    """Test automated testing and monitoring workflows"""
    
    def test_load_testing_workflow(self, client):
        """Test automated load testing workflow"""
        start_time = time.time()
        
        # Simulate load testing
        num_requests = 100
        success_count = 0
        error_count = 0
        latencies = []
        
        test_cases = [
            {"logits": [0.5, -0.2, 0.8, -0.5, 0.3], "ambiguity": 0.2},
            {"logits": [0.1, 0.05, 0.12, 0.08, 0.11], "ambiguity": 0.9},
            {"logits": [0.3, 0.1, 0.4, -0.1, 0.2], "ambiguity": 0.5},
        ]
        
        for i in range(num_requests):
            test_case = test_cases[i % len(test_cases)]
            
            request_start = time.time()
            try:
                response = client.post("/route", json={
                    "logits": test_case["logits"],
                    "ambiguity_score": test_case["ambiguity"],
                    "request_id": f"load_test_{i}"
                })
                
                if response.status_code == 200:
                    success_count += 1
                    latencies.append((time.time() - request_start) * 1000)
                else:
                    error_count += 1
                    
            except Exception:
                error_count += 1
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert success_count >= num_requests * 0.95  # 95% success rate
        assert error_count <= num_requests * 0.05    # 5% error rate
        
        if latencies:
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            assert avg_latency < 100.0  # Average < 100ms
            assert p95_latency < 200.0  # P95 < 200ms
        
        # Throughput check
        rps = num_requests / total_time
        assert rps > 10.0  # At least 10 RPS
    
    def test_concurrent_request_workflow(self, client):
        """Test concurrent request handling"""
        def make_request(request_id: int) -> Dict[str, Any]:
            """Make a single request"""
            try:
                response = client.post("/route", json={
                    "logits": np.random.normal(0, 0.5, 5).tolist(),
                    "ambiguity_score": np.random.uniform(0.1, 0.9),
                    "request_id": f"concurrent_{request_id}"
                })
                
                return {
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "data": response.json() if response.status_code == 200 else None
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        # Run concurrent requests
        num_workers = 10
        requests_per_worker = 20
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers * requests_per_worker):
                future = executor.submit(make_request, i)
                futures.append(future)
            
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        success_rate = len(successful) / len(results)
        assert success_rate >= 0.95  # 95% success rate
        
        # Verify response consistency
        for result in successful:
            data = result["data"]
            assert "routing_weights" in data
            assert "controller_version" in data
            assert data["controller_version"] == "4.72x_improvement"
    
    def test_stability_monitoring_workflow(self, client):
        """Test automated stability monitoring workflow"""
        # Collect baseline metrics
        baseline_metrics = []
        
        for i in range(20):
            response = client.post("/route", json={
                "logits": [0.5, -0.2, 0.8, -0.5, 0.3],
                "ambiguity_score": 0.8,  # High ambiguity
                "request_id": f"stability_baseline_{i}"
            })
            
            if response.status_code == 200:
                data = response.json()
                baseline_metrics.append(data["metrics"])
        
        # Analyze stability
        gating_sensitivities = [m["gating_sensitivity"] for m in baseline_metrics]
        flip_rates = [m["winner_flip_rate"] for m in baseline_metrics]
        
        # Stability checks
        avg_sensitivity = np.mean(gating_sensitivities)
        max_sensitivity = np.max(gating_sensitivities)
        avg_flip_rate = np.mean(flip_rates)
        std_sensitivity = np.std(gating_sensitivities)
        
        # Automated alerts (would trigger monitoring systems)
        stability_alerts = []
        
        if avg_sensitivity > 0.15:
            stability_alerts.append(f"High average gating sensitivity: {avg_sensitivity:.4f}")
        
        if max_sensitivity > 0.25:
            stability_alerts.append(f"Peak gating sensitivity too high: {max_sensitivity:.4f}")
        
        if avg_flip_rate > 0.08:
            stability_alerts.append(f"High winner flip rate: {avg_flip_rate:.4f}")
        
        if std_sensitivity > 0.05:
            stability_alerts.append(f"High sensitivity variance: {std_sensitivity:.4f}")
        
        # Should have minimal stability issues
        assert len(stability_alerts) == 0, f"Stability alerts: {stability_alerts}"
        
        # Performance targets for 4.72x improvement
        assert avg_sensitivity < 0.1, f"Failed target: avg sensitivity {avg_sensitivity:.4f} > 0.1"
        assert avg_flip_rate < 0.05, f"Failed target: avg flip rate {avg_flip_rate:.4f} > 0.05"
    
    def test_regression_testing_workflow(self, client):
        """Test automated regression testing workflow"""
        # Standard test cases for regression testing
        regression_test_cases = [
            {
                "name": "low_ambiguity_clear_winner",
                "logits": [0.9, -0.5, 0.1, -0.3, 0.2],
                "ambiguity": 0.2,
                "expected_sensitivity_max": 0.05,
                "expected_flip_rate_max": 0.02
            },
            {
                "name": "high_ambiguity_uncertain",
                "logits": [0.1, 0.05, 0.12, 0.08, 0.11],
                "ambiguity": 0.9,
                "expected_sensitivity_max": 0.12,
                "expected_flip_rate_max": 0.06
            },
            {
                "name": "medium_ambiguity_balanced",
                "logits": [0.3, 0.1, 0.4, -0.1, 0.2],
                "ambiguity": 0.5,
                "expected_sensitivity_max": 0.08,
                "expected_flip_rate_max": 0.04
            }
        ]
        
        regression_results = {}
        
        for test_case in regression_test_cases:
            # Run test case multiple times for statistical significance
            metrics_list = []
            
            for run in range(10):
                response = client.post("/route", json={
                    "logits": test_case["logits"],
                    "ambiguity_score": test_case["ambiguity"],
                    "request_id": f"regression_{test_case['name']}_{run}"
                })
                
                assert response.status_code == 200
                data = response.json()
                metrics_list.append(data["metrics"])
            
            # Analyze results
            avg_sensitivity = np.mean([m["gating_sensitivity"] for m in metrics_list])
            avg_flip_rate = np.mean([m["winner_flip_rate"] for m in metrics_list])
            
            regression_results[test_case["name"]] = {
                "avg_sensitivity": avg_sensitivity,
                "avg_flip_rate": avg_flip_rate,
                "passed_sensitivity": avg_sensitivity <= test_case["expected_sensitivity_max"],
                "passed_flip_rate": avg_flip_rate <= test_case["expected_flip_rate_max"]
            }
            
            # Regression assertions
            assert avg_sensitivity <= test_case["expected_sensitivity_max"], \
                f"{test_case['name']}: sensitivity {avg_sensitivity:.4f} > {test_case['expected_sensitivity_max']}"
            
            assert avg_flip_rate <= test_case["expected_flip_rate_max"], \
                f"{test_case['name']}: flip rate {avg_flip_rate:.4f} > {test_case['expected_flip_rate_max']}"
        
        # Validate overall regression test passed
        all_passed = all(
            result["passed_sensitivity"] and result["passed_flip_rate"] 
            for result in regression_results.values()
        )
        assert all_passed, f"Regression test failures: {regression_results}"
    
    def test_continuous_integration_workflow(self, client):
        """Test CI/CD workflow validation"""
        # Step 1: Health check
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        
        # Step 2: Improvement validation
        validate_response = client.post("/validate")
        assert validate_response.status_code == 200
        
        validate_data = validate_response.json()
        assert validate_data["status"] == "PASSED"
        assert validate_data["validation_results"]["improvement_factor"] >= 4.0
        
        # Step 3: Performance benchmarking
        benchmark_start = time.time()
        benchmark_requests = 50
        
        for i in range(benchmark_requests):
            response = client.post("/route", json={
                "logits": [0.5, -0.2, 0.8, -0.5, 0.3],
                "ambiguity_score": 0.5,
                "request_id": f"benchmark_{i}"
            })
            assert response.status_code == 200
        
        benchmark_time = time.time() - benchmark_start
        benchmark_rps = benchmark_requests / benchmark_time
        
        # CI/CD performance gates
        assert benchmark_rps > 20.0, f"Performance regression: {benchmark_rps:.1f} RPS < 20 RPS"
        
        # Step 4: Final metrics validation
        final_metrics = client.get("/metrics").json()
        assert final_metrics["requests_processed"] >= benchmark_requests
        assert final_metrics["status"] == "healthy"
        
        print(f"CI/CD Validation Passed:")
        print(f"  - Health: {health_data['status']}")
        print(f"  - Improvement: {validate_data['validation_results']['improvement_factor']:.2f}x")
        print(f"  - Performance: {benchmark_rps:.1f} RPS")
        print(f"  - Total Requests: {final_metrics['requests_processed']}")


class TestBackendIntegration:
    """Test integration with different LLM backends"""
    
    def test_mock_mlx_integration(self, client, mock_mlx_backend):
        """Test integration with MLX backend"""
        # This would test actual MLX integration in a real implementation
        # For now, testing the mock backend integration pattern
        
        # Simulate MLX-generated logits (typically more normalized)
        mlx_logits = await mock_mlx_backend.generate_logits("test prompt", 5)
        ambiguity = mock_mlx_backend.estimate_ambiguity("test prompt")
        
        response = client.post("/route", json={
            "logits": mlx_logits,
            "ambiguity_score": ambiguity,
            "request_id": "mlx_test"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "routing_weights" in data
        assert data["controller_version"] == "4.72x_improvement"
    
    def test_mock_vllm_integration(self, client, mock_vllm_backend):
        """Test integration with vLLM backend"""
        # Simulate vLLM-generated logits (can have wider range)
        vllm_logits = await mock_vllm_backend.generate_logits("test prompt", 5)
        ambiguity = mock_vllm_backend.estimate_ambiguity("test prompt")
        
        response = client.post("/route", json={
            "logits": vllm_logits,
            "ambiguity_score": ambiguity,
            "request_id": "vllm_test"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "routing_weights" in data
        assert data["controller_version"] == "4.72x_improvement"
    
    @pytest.mark.asyncio
    async def test_backend_comparison(self, client, mock_mlx_backend, mock_vllm_backend):
        """Compare stability across different backends"""
        backends = [
            ("mlx", mock_mlx_backend),
            ("vllm", mock_vllm_backend)
        ]
        
        backend_results = {}
        
        for backend_name, backend in backends:
            metrics_list = []
            
            for i in range(10):
                logits = await backend.generate_logits(f"test prompt {i}", 5)
                ambiguity = backend.estimate_ambiguity(f"test prompt {i}")
                
                response = client.post("/route", json={
                    "logits": logits,
                    "ambiguity_score": ambiguity,
                    "request_id": f"{backend_name}_test_{i}"
                })
                
                assert response.status_code == 200
                data = response.json()
                metrics_list.append(data["metrics"])
            
            # Analyze backend-specific results
            avg_sensitivity = np.mean([m["gating_sensitivity"] for m in metrics_list])
            avg_flip_rate = np.mean([m["winner_flip_rate"] for m in metrics_list])
            
            backend_results[backend_name] = {
                "avg_sensitivity": avg_sensitivity,
                "avg_flip_rate": avg_flip_rate,
                "call_count": backend.call_count
            }
        
        # Both backends should achieve good stability
        for backend_name, results in backend_results.items():
            assert results["avg_sensitivity"] < 0.15, \
                f"{backend_name} sensitivity too high: {results['avg_sensitivity']:.4f}"
            assert results["avg_flip_rate"] < 0.08, \
                f"{backend_name} flip rate too high: {results['avg_flip_rate']:.4f}"