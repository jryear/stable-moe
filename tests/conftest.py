"""
Pytest configuration and fixtures for MoE Routing tests
"""

import pytest
import numpy as np
import asyncio
from typing import List, Dict, Generator
from unittest.mock import Mock, patch
import tempfile
import os

from src.core.production_controller import ProductionClarityController, RoutingMetrics
from src.api.server import app
from fastapi.testclient import TestClient

@pytest.fixture
def controller():
    """Create a fresh controller instance for testing"""
    return ProductionClarityController()

@pytest.fixture
def api_client():
    """Create FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def sample_logits():
    """Standard test logits"""
    return [0.5, -0.2, 0.8, -0.5, 0.3]

@pytest.fixture
def high_ambiguity_logits():
    """Logits that create high ambiguity scenario"""
    return [0.1, 0.05, 0.12, 0.08, 0.11]

@pytest.fixture
def low_ambiguity_logits():
    """Logits with clear winner (low ambiguity)"""
    return [0.9, -0.5, 0.1, -0.3, 0.2]

@pytest.fixture
def ambiguity_levels():
    """Range of ambiguity scores for testing"""
    return [0.1, 0.3, 0.5, 0.7, 0.9]

@pytest.fixture
def mock_routing_metrics():
    """Mock RoutingMetrics object"""
    return RoutingMetrics(
        gating_sensitivity=0.0654,
        winner_flip_rate=0.0123,
        boundary_distance=0.458,
        routing_entropy=1.234,
        latency_ms=2.3,
        clarity_score=0.200,
        timestamp="2024-01-01T00:00:00Z",
        beta=0.85,
        lambda_val=0.92
    )

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def stability_test_data():
    """Generate test data for stability analysis"""
    np.random.seed(42)
    
    # Generate multiple logits with varying ambiguity
    test_cases = []
    for ambiguity in [0.1, 0.5, 0.9]:
        for _ in range(5):
            if ambiguity < 0.3:
                # Low ambiguity - clear winner
                logits = np.array([1.0, -0.5, 0.1, -0.3, 0.2])
                logits += np.random.normal(0, 0.1, 5)
            elif ambiguity > 0.7:
                # High ambiguity - similar values
                logits = np.random.normal(0, 0.1, 5)
            else:
                # Medium ambiguity
                logits = np.random.normal(0, 0.3, 5)
            
            test_cases.append({
                'logits': logits.tolist(),
                'ambiguity': ambiguity,
                'expected_sensitivity_range': (0.01, 0.2) if ambiguity > 0.7 else (0.001, 0.05)
            })
    
    return test_cases

@pytest.fixture
def performance_test_cases():
    """Generate performance test cases"""
    test_cases = []
    
    # Edge cases
    test_cases.extend([
        {
            'name': 'zeros',
            'logits': [0.0, 0.0, 0.0, 0.0, 0.0],
            'ambiguity': 0.5
        },
        {
            'name': 'extreme_values',
            'logits': [10.0, -10.0, 5.0, -5.0, 0.0],
            'ambiguity': 0.2
        },
        {
            'name': 'single_expert',
            'logits': [1.0, 0.0, 0.0, 0.0, 0.0],
            'ambiguity': 0.1
        },
        {
            'name': 'negative_logits',
            'logits': [-0.1, -0.2, -0.15, -0.25, -0.18],
            'ambiguity': 0.8
        }
    ])
    
    # Random test cases
    np.random.seed(123)
    for i in range(20):
        logits = np.random.normal(0, 1, 5).tolist()
        ambiguity = np.random.uniform(0.1, 0.9)
        test_cases.append({
            'name': f'random_{i}',
            'logits': logits,
            'ambiguity': ambiguity
        })
    
    return test_cases

class MockBackend:
    """Mock LLM backend for testing"""
    
    def __init__(self, backend_type="mock"):
        self.backend_type = backend_type
        self.call_count = 0
        
    async def generate_logits(self, prompt: str, num_experts: int = 5) -> List[float]:
        """Mock logit generation"""
        self.call_count += 1
        
        # Simulate different backend behaviors
        if self.backend_type == "mlx":
            # MLX-style logits (typically more normalized)
            return np.random.normal(0, 0.5, num_experts).tolist()
        elif self.backend_type == "vllm":
            # vLLM-style logits (can have wider range)
            return np.random.normal(0, 1.0, num_experts).tolist()
        else:
            # Default mock
            return [0.5, -0.2, 0.8, -0.5, 0.3][:num_experts]
    
    def estimate_ambiguity(self, prompt: str) -> float:
        """Mock ambiguity estimation"""
        # Simple heuristic based on prompt length
        return min(0.9, len(prompt) / 100.0)

@pytest.fixture
def mock_mlx_backend():
    """Mock MLX backend"""
    return MockBackend("mlx")

@pytest.fixture 
def mock_vllm_backend():
    """Mock vLLM backend"""
    return MockBackend("vllm")

@pytest.fixture
def validation_thresholds():
    """Standard validation thresholds for tests"""
    return {
        'max_gating_sensitivity': 0.2,
        'max_winner_flip_rate': 0.1,
        'min_improvement_factor': 3.0,
        'max_latency_ms': 100.0,
        'min_clarity_correlation': 0.5
    }