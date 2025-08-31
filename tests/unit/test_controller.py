"""
Unit tests for ProductionClarityController
Tests core controller logic and 4.72x improvement
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock
import time

from src.core.production_controller import ProductionClarityController, RoutingMetrics


class TestProductionClarityController:
    """Test suite for the production controller"""
    
    def test_controller_initialization(self):
        """Test controller initializes correctly"""
        controller = ProductionClarityController()
        
        assert controller is not None
        assert controller.request_count == 0
        assert len(controller.recent_metrics) == 0
        assert controller.previous_weights is None
    
    def test_adaptive_beta_calculation(self, controller):
        """Test beta parameter adapts to clarity"""
        # High clarity should give higher beta
        high_clarity_beta = controller.get_beta(0.9)
        low_clarity_beta = controller.get_beta(0.1)
        
        assert high_clarity_beta > low_clarity_beta
        assert 0.3 <= low_clarity_beta <= 1.0
        assert 0.3 <= high_clarity_beta <= 1.0
    
    def test_adaptive_lambda_calculation(self, controller):
        """Test lambda parameter adapts to clarity"""
        # High clarity should give lower lambda (less inertia)
        high_clarity_lambda = controller.get_lambda(0.9)
        low_clarity_lambda = controller.get_lambda(0.1)
        
        assert high_clarity_lambda < low_clarity_lambda
        assert 0.85 <= high_clarity_lambda <= 0.95
        assert 0.85 <= low_clarity_lambda <= 0.95
    
    def test_route_with_control_basic(self, controller, sample_logits):
        """Test basic routing functionality"""
        weights, metrics = controller.route_with_control(
            np.array(sample_logits), 
            ambiguity_score=0.5,
            request_id="test_001"
        )
        
        # Check output format
        assert len(weights) == len(sample_logits)
        assert abs(np.sum(weights) - 1.0) < 1e-6  # Weights sum to 1
        assert all(w >= 0 for w in weights)  # All weights non-negative
        
        # Check metrics
        assert isinstance(metrics, RoutingMetrics)
        assert metrics.gating_sensitivity >= 0
        assert metrics.winner_flip_rate >= 0
        assert 0 <= metrics.clarity_score <= 1
        assert metrics.latency_ms >= 0
    
    def test_gating_sensitivity_improves_with_low_clarity(self, controller):
        """Test that gating sensitivity is lower under high ambiguity (low clarity)"""
        logits = np.array([0.1, 0.05, 0.12, 0.08, 0.11])  # High ambiguity logits
        
        # High ambiguity (low clarity) should have lower gating sensitivity
        _, high_amb_metrics = controller.route_with_control(logits, 0.9, "high_amb")
        
        # Low ambiguity (high clarity) 
        _, low_amb_metrics = controller.route_with_control(logits, 0.1, "low_amb")
        
        # Controller should be more stable (lower sensitivity) under high ambiguity
        assert high_amb_metrics.gating_sensitivity < low_amb_metrics.gating_sensitivity
    
    def test_winner_flip_rate_calculation(self, controller, sample_logits):
        """Test winner flip rate tracking"""
        logits = np.array(sample_logits)
        
        # First call - no previous weights
        weights1, metrics1 = controller.route_with_control(logits, 0.5, "test1")
        assert metrics1.winner_flip_rate == 0.0  # No previous winner
        
        # Second call - should track flip rate
        weights2, metrics2 = controller.route_with_control(logits + 0.1, 0.5, "test2")
        assert metrics2.winner_flip_rate >= 0.0
    
    def test_ema_smoothing_effect(self, controller):
        """Test that EMA smoothing reduces weight volatility"""
        base_logits = np.array([0.5, -0.2, 0.8, -0.5, 0.3])
        
        # Multiple routing calls with slightly different logits
        previous_weights = None
        weight_changes = []
        
        for i in range(5):
            noisy_logits = base_logits + np.random.normal(0, 0.05, 5)
            weights, _ = controller.route_with_control(noisy_logits, 0.7, f"ema_test_{i}")
            
            if previous_weights is not None:
                change = np.linalg.norm(weights - previous_weights)
                weight_changes.append(change)
            
            previous_weights = weights
        
        # Later changes should be smaller due to EMA smoothing
        if len(weight_changes) >= 3:
            early_change = np.mean(weight_changes[:2])
            later_change = np.mean(weight_changes[-2:])
            assert later_change <= early_change * 1.5  # Allow some variance
    
    def test_boundary_distance_calculation(self, controller):
        """Test boundary distance metric"""
        # Test with clear winner (low ambiguity)
        clear_logits = np.array([1.0, -0.5, 0.1, -0.3, 0.2])
        _, clear_metrics = controller.route_with_control(clear_logits, 0.2, "clear")
        
        # Test with unclear winner (high ambiguity)
        unclear_logits = np.array([0.1, 0.05, 0.12, 0.08, 0.11])
        _, unclear_metrics = controller.route_with_control(unclear_logits, 0.8, "unclear")
        
        # Clear winner should have larger boundary distance
        assert clear_metrics.boundary_distance > unclear_metrics.boundary_distance
    
    def test_performance_stats(self, controller, sample_logits):
        """Test performance statistics collection"""
        # Make several routing calls
        for i in range(5):
            controller.route_with_control(
                np.array(sample_logits), 
                0.5 + i * 0.1, 
                f"perf_test_{i}"
            )
        
        stats = controller.get_performance_stats()
        
        assert stats['requests_processed'] == 5
        assert 'avg_gating_sensitivity' in stats
        assert 'avg_winner_flip_rate' in stats
        assert 'avg_latency_ms' in stats
        assert stats['status'] in ['healthy', 'degraded']
    
    def test_recent_metrics_limiting(self, controller, sample_logits):
        """Test that recent metrics are properly limited"""
        # Make many routing calls
        for i in range(15):
            controller.route_with_control(
                np.array(sample_logits), 
                0.5, 
                f"limit_test_{i}"
            )
        
        # Should only keep last 10 by default
        recent = controller.get_recent_metrics(10)
        assert len(recent) == 10
        
        # Should be able to request fewer
        recent_5 = controller.get_recent_metrics(5)
        assert len(recent_5) == 5
    
    def test_validate_improvement(self, controller):
        """Test improvement validation against baseline"""
        validation_result = controller.validate_improvement(
            np.array([0.5, -0.2, 0.8, -0.5, 0.3])
        )
        
        assert 'improvement_factor' in validation_result
        assert 'meets_target' in validation_result
        assert 'baseline_sensitivity' in validation_result
        assert 'controlled_sensitivity' in validation_result
        
        # Should meet the 4.0x minimum target
        assert validation_result['improvement_factor'] >= 4.0
        assert validation_result['meets_target'] is True
    
    def test_reset_state(self, controller, sample_logits):
        """Test state reset functionality"""
        # Build up some state
        for i in range(3):
            controller.route_with_control(np.array(sample_logits), 0.5, f"reset_test_{i}")
        
        assert controller.request_count == 3
        assert len(controller.recent_metrics) == 3
        
        # Reset state
        controller.reset_state()
        
        assert controller.request_count == 0
        assert len(controller.recent_metrics) == 0
        assert controller.previous_weights is None
    
    def test_thread_safety(self, controller, sample_logits):
        """Test basic thread safety of controller operations"""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def worker(worker_id):
            try:
                for i in range(10):
                    weights, metrics = controller.route_with_control(
                        np.array(sample_logits) + np.random.normal(0, 0.01, 5),
                        0.5,
                        f"thread_{worker_id}_{i}"
                    )
                    results.put((worker_id, weights, metrics))
            except Exception as e:
                errors.put((worker_id, e))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check for errors
        assert errors.empty(), f"Thread safety errors: {list(errors.queue)}"
        
        # Should have results from all threads
        assert results.qsize() == 30  # 3 threads * 10 calls each
    
    @pytest.mark.parametrize("ambiguity", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_ambiguity_scaling(self, controller, sample_logits, ambiguity):
        """Test controller behavior across ambiguity spectrum"""
        weights, metrics = controller.route_with_control(
            np.array(sample_logits),
            ambiguity,
            f"ambiguity_test_{ambiguity}"
        )
        
        # Basic sanity checks for all ambiguity levels
        assert len(weights) == len(sample_logits)
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
        assert metrics.clarity_score == 1.0 - ambiguity
        assert metrics.gating_sensitivity >= 0
        assert metrics.winner_flip_rate >= 0
    
    def test_numerical_stability(self, controller):
        """Test controller with extreme/edge case inputs"""
        edge_cases = [
            # Very large logits
            [100.0, -100.0, 50.0, -50.0, 0.0],
            # Very small logits
            [0.001, -0.001, 0.0005, -0.0005, 0.0],
            # All zeros
            [0.0, 0.0, 0.0, 0.0, 0.0],
            # All same value
            [0.5, 0.5, 0.5, 0.5, 0.5]
        ]
        
        for i, logits in enumerate(edge_cases):
            weights, metrics = controller.route_with_control(
                np.array(logits),
                0.5,
                f"edge_case_{i}"
            )
            
            # Should still produce valid outputs
            assert len(weights) == len(logits)
            assert abs(np.sum(weights) - 1.0) < 1e-5
            assert all(w >= 0 for w in weights)
            assert all(np.isfinite(w) for w in weights)
            assert np.isfinite(metrics.gating_sensitivity)
            assert np.isfinite(metrics.winner_flip_rate)
    
    def test_latency_measurement(self, controller, sample_logits):
        """Test that latency is properly measured"""
        # Add artificial delay
        with patch('time.time', side_effect=[1000.0, 1000.05]):  # 50ms
            _, metrics = controller.route_with_control(
                np.array(sample_logits),
                0.5,
                "latency_test"
            )
        
        assert metrics.latency_ms == pytest.approx(50.0, rel=0.1)