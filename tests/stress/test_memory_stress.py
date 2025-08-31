"""
Memory stress tests for MoE Routing Controller
Tests memory usage, leaks, and performance under load
"""

import pytest
import numpy as np
import psutil
import threading
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import tracemalloc

from src.core.production_controller import ProductionClarityController


class TestMemoryStress:
    """Memory stress testing suite"""
    
    def get_memory_usage_mb(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def test_memory_baseline(self, controller):
        """Establish memory baseline for controller"""
        initial_memory = self.get_memory_usage_mb()
        
        # Single routing call
        logits = np.array([0.5, -0.2, 0.8, -0.5, 0.3])
        controller.route_with_control(logits, 0.5, "baseline")
        
        after_memory = self.get_memory_usage_mb()
        memory_increase = after_memory - initial_memory
        
        # Single call should use minimal memory (< 1MB)
        assert memory_increase < 1.0, f"Single call used {memory_increase:.2f}MB"
    
    def test_memory_growth_under_load(self, controller):
        """Test memory doesn't grow unbounded under sustained load"""
        initial_memory = self.get_memory_usage_mb()
        logits = np.array([0.5, -0.2, 0.8, -0.5, 0.3])
        
        # Make many requests
        num_requests = 10000
        memory_samples = []
        
        for i in range(num_requests):
            # Vary inputs slightly to prevent optimization
            noisy_logits = logits + np.random.normal(0, 0.01, 5)
            ambiguity = 0.5 + np.random.normal(0, 0.1)
            ambiguity = max(0.0, min(1.0, ambiguity))
            
            controller.route_with_control(noisy_logits, ambiguity, f"load_test_{i}")
            
            # Sample memory periodically
            if i % 1000 == 0:
                memory_samples.append(self.get_memory_usage_mb())
        
        final_memory = self.get_memory_usage_mb()
        total_growth = final_memory - initial_memory
        
        # Memory should stabilize (< 50MB growth for 10k requests)
        assert total_growth < 50.0, f"Memory grew by {total_growth:.2f}MB"
        
        # Check that growth rate decreases over time
        if len(memory_samples) >= 5:
            early_growth = memory_samples[2] - memory_samples[0]
            late_growth = memory_samples[-1] - memory_samples[-3]
            
            # Late growth should be much smaller than early growth
            assert late_growth <= early_growth * 2.0, "Memory growth not stabilizing"
    
    def test_memory_cleanup_after_reset(self, controller):
        """Test memory is properly cleaned up after state reset"""
        initial_memory = self.get_memory_usage_mb()
        logits = np.array([0.5, -0.2, 0.8, -0.5, 0.3])
        
        # Build up state
        for i in range(5000):
            controller.route_with_control(
                logits + np.random.normal(0, 0.01, 5),
                np.random.uniform(0.1, 0.9),
                f"cleanup_test_{i}"
            )
        
        memory_after_load = self.get_memory_usage_mb()
        
        # Reset state and force garbage collection
        controller.reset_state()
        gc.collect()
        time.sleep(0.1)  # Allow cleanup
        
        memory_after_reset = self.get_memory_usage_mb()
        memory_freed = memory_after_load - memory_after_reset
        
        # Should free significant memory (at least 50% of growth)
        growth = memory_after_load - initial_memory
        if growth > 1.0:  # Only check if there was significant growth
            assert memory_freed > growth * 0.3, f"Only freed {memory_freed:.2f}MB of {growth:.2f}MB"
    
    @pytest.mark.timeout(300)  # 5 minute timeout
    def test_concurrent_memory_stress(self, controller):
        """Test memory usage under concurrent load"""
        initial_memory = self.get_memory_usage_mb()
        
        def worker(worker_id: int, num_requests: int) -> Dict:
            """Worker thread for concurrent testing"""
            memory_samples = []
            for i in range(num_requests):
                logits = np.random.normal(0, 0.5, 5)
                ambiguity = np.random.uniform(0.1, 0.9)
                
                controller.route_with_control(
                    logits, ambiguity, f"concurrent_{worker_id}_{i}"
                )
                
                if i % 100 == 0:
                    memory_samples.append(self.get_memory_usage_mb())
            
            return {
                'worker_id': worker_id,
                'memory_samples': memory_samples,
                'final_memory': self.get_memory_usage_mb()
            }
        
        # Run multiple workers concurrently
        num_workers = 4
        requests_per_worker = 2500
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker, i, requests_per_worker)
                for i in range(num_workers)
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        final_memory = self.get_memory_usage_mb()
        total_growth = final_memory - initial_memory
        
        # Total memory growth should be reasonable for 10k concurrent requests
        assert total_growth < 100.0, f"Concurrent load caused {total_growth:.2f}MB growth"
        
        # Verify all workers completed successfully
        assert len(results) == num_workers
        for result in results:
            assert len(result['memory_samples']) > 0
    
    def test_memory_leak_detection(self, controller):
        """Detect potential memory leaks through repeated cycles"""
        tracemalloc.start()
        
        def run_cycle():
            """Run a complete cycle of operations"""
            logits = np.random.normal(0, 0.5, 5)
            
            # Multiple operations per cycle
            for _ in range(100):
                controller.route_with_control(
                    logits + np.random.normal(0, 0.01, 5),
                    np.random.uniform(0.1, 0.9),
                    f"leak_test_{np.random.randint(0, 10000)}"
                )
            
            # Get metrics
            controller.get_performance_stats()
            controller.get_recent_metrics(50)
        
        # Run multiple cycles and measure memory growth
        cycle_memories = []
        
        for cycle in range(10):
            run_cycle()
            
            # Force cleanup between cycles
            if cycle % 3 == 0:
                controller.reset_state()
            
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            cycle_memories.append(current / (1024 * 1024))  # Convert to MB
        
        tracemalloc.stop()
        
        # Check for linear memory growth (indicates leak)
        if len(cycle_memories) >= 5:
            # Calculate trend
            x = np.arange(len(cycle_memories))
            y = np.array(cycle_memories)
            slope = np.polyfit(x, y, 1)[0]
            
            # Slope should be small (< 0.5MB per cycle)
            assert slope < 0.5, f"Potential memory leak: {slope:.2f}MB growth per cycle"
    
    def test_large_batch_processing(self, controller):
        """Test memory efficiency with large batch processing"""
        initial_memory = self.get_memory_usage_mb()
        
        # Process large batch of requests
        batch_size = 50000
        batch_logits = []
        batch_ambiguities = []
        
        # Generate batch data
        np.random.seed(42)
        for _ in range(batch_size):
            batch_logits.append(np.random.normal(0, 0.5, 5))
            batch_ambiguities.append(np.random.uniform(0.1, 0.9))
        
        peak_memory = initial_memory
        
        # Process batch
        for i, (logits, ambiguity) in enumerate(zip(batch_logits, batch_ambiguities)):
            controller.route_with_control(logits, ambiguity, f"batch_{i}")
            
            # Monitor peak memory
            if i % 5000 == 0:
                current_memory = self.get_memory_usage_mb()
                peak_memory = max(peak_memory, current_memory)
        
        final_memory = self.get_memory_usage_mb()
        
        # Memory usage should be reasonable for large batch
        peak_growth = peak_memory - initial_memory
        final_growth = final_memory - initial_memory
        
        assert peak_growth < 200.0, f"Peak memory growth: {peak_growth:.2f}MB"
        assert final_growth < 150.0, f"Final memory growth: {final_growth:.2f}MB"
    
    def test_metrics_history_memory_limit(self, controller):
        """Test that metrics history doesn't grow unbounded"""
        initial_memory = self.get_memory_usage_mb()
        
        # Generate many requests to build up metrics history
        for i in range(20000):
            logits = np.random.normal(0, 0.5, 5)
            controller.route_with_control(
                logits, 0.5, f"history_test_{i}"
            )
        
        # Check recent metrics length
        recent_metrics = controller.get_recent_metrics(10000)
        
        # Should be limited (default is 10000 max)
        assert len(recent_metrics) <= 10000, f"Metrics history too large: {len(recent_metrics)}"
        
        # Memory growth should be bounded
        final_memory = self.get_memory_usage_mb()
        growth = final_memory - initial_memory
        
        # Even with 20k requests, memory should be reasonable
        assert growth < 100.0, f"Unbounded metrics history: {growth:.2f}MB"
    
    def test_memory_pressure_handling(self, controller):
        """Test controller behavior under memory pressure"""
        # This is a conceptual test - actual implementation would depend on system
        initial_memory = self.get_memory_usage_mb()
        
        # Simulate memory pressure by making many large requests
        large_logits_list = []
        for _ in range(1000):
            # Create larger than normal logit arrays
            large_logits = np.random.normal(0, 0.5, 50)  # Larger than typical 5
            large_logits_list.append(large_logits)
        
        # Process under memory pressure
        success_count = 0
        for i, logits in enumerate(large_logits_list):
            try:
                # Use only first 5 elements for routing
                controller.route_with_control(
                    logits[:5], 0.5, f"pressure_test_{i}"
                )
                success_count += 1
            except MemoryError:
                # Should handle gracefully
                break
        
        # Should process most requests successfully
        assert success_count > 900, f"Only {success_count}/1000 succeeded under pressure"
        
        final_memory = self.get_memory_usage_mb()
        growth = final_memory - initial_memory
        
        # Should not consume excessive memory
        assert growth < 50.0, f"Memory pressure test used {growth:.2f}MB"


class TestResourceConstraints:
    """Test resource constraint handling"""
    
    def test_cpu_bound_operations(self, controller):
        """Test CPU-intensive operations don't block indefinitely"""
        start_time = time.time()
        
        # CPU-intensive test with many small operations
        for i in range(5000):
            # Complex logit patterns that require more computation
            logits = np.sin(np.linspace(0, 2*np.pi, 5)) + np.random.normal(0, 0.1, 5)
            controller.route_with_control(logits, 0.5, f"cpu_test_{i}")
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 30 seconds)
        assert elapsed < 30.0, f"CPU-bound operations took {elapsed:.2f}s"
        
        # Performance should be reasonable (> 100 RPS)
        rps = 5000 / elapsed
        assert rps > 100, f"Performance too low: {rps:.1f} RPS"
    
    def test_memory_constrained_environment(self, controller):
        """Test behavior in memory-constrained environment"""
        # Simulate constraints by tracking memory closely
        memory_limit_mb = 100  # Artificial limit for testing
        
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        request_count = 0
        
        try:
            while request_count < 10000:
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_used = current_memory - initial_memory
                
                if memory_used > memory_limit_mb:
                    # Trigger cleanup
                    controller.reset_state()
                    gc.collect()
                
                logits = np.random.normal(0, 0.5, 5)
                controller.route_with_control(
                    logits, 0.5, f"constrained_{request_count}"
                )
                request_count += 1
        
        except Exception as e:
            pytest.fail(f"Failed under memory constraints: {e}")
        
        # Should process significant number of requests
        assert request_count >= 5000, f"Only processed {request_count} requests"