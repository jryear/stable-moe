#!/usr/bin/env python3
"""
Clarity Controller V2 - Enhanced with Spike Guards and Mediation-Informed Design
Implements exact schedules specified for locking down the A* → G → L_⊥ control policy

Key Features:
- β(C): Sharpness control based on clarity
- λ(C): EMA inertia based on clarity  
- Spike guard: Hold β,λ for 5 ticks if ||∇w|| spikes 3 times
- Optional S-matrix freeze when A* > 0.7 unless judge improves
- Distance-to-vertex tracking (D = 1 - max_i w_i)
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Deque
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass

@dataclass
class ControlState:
    """State tracking for spike detection and S-matrix control"""
    gradient_history: Deque[float]  # Recent ||∇w|| values
    spike_count: int  # Consecutive spikes
    hold_until_tick: int  # Hold parameters until this tick
    s_matrix_frozen: bool  # S-matrix learning frozen
    last_judge_score: float  # For S-matrix freeze logic
    tick: int  # Current control tick

@dataclass
class EnhancedRoutingMetrics:
    """Enhanced metrics including distance-to-vertex and mediation tracking"""
    # Core metrics
    gating_sensitivity: float
    winner_flip_rate: float
    boundary_distance: float
    routing_entropy: float
    latency_ms: float
    timestamp: float
    
    # Enhanced metrics
    clarity_score: float
    beta: float
    lambda_val: float
    distance_to_vertex: float  # D = 1 - max_i w_i
    gradient_norm: float  # ||∇w||
    spike_detected: bool
    control_state: str  # 'normal', 'holding', 's_frozen'
    
    # Mediation tracking
    expected_G_reduction: float  # Based on β,λ settings
    mediation_ratio: Optional[float]  # If available from recent analysis

class ClarityControllerV2:
    """
    Enhanced Clarity Controller with Spike Guards and Mediation-Informed Control
    
    Validated Control Policy:
    - High A* → Lower β, Higher λ → Reduced G → Lower L_⊥
    - Spike guard prevents thrashing
    - S-matrix freeze prevents learning on ambiguous examples
    """
    
    def __init__(self,
                 # Core schedules (your exact specification)
                 beta_min: float = 0.3,
                 beta_max: float = 1.6, 
                 lambda_min: float = 0.2,
                 lambda_max: float = 0.8,
                 
                 # Spike detection parameters
                 spike_threshold: float = 2.5,  # ||∇w|| threshold
                 spike_memory: int = 5,  # Look back this many ticks
                 consecutive_spikes_trigger: int = 3,  # Trigger after this many
                 hold_duration: int = 5,  # Hold β,λ for this many ticks
                 
                 # S-matrix control
                 s_freeze_ambiguity_threshold: float = 0.7,
                 s_freeze_improvement_threshold: float = 0.05,
                 eta_base: float = 0.01,
                 k_eta: float = 0.75,
                 
                 # State management
                 metrics_buffer_size: int = 1000,
                 flip_history_size: int = 100):
        
        # Core schedules
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.lambda_min = lambda_min  
        self.lambda_max = lambda_max
        
        # Spike detection
        self.spike_threshold = spike_threshold
        self.spike_memory = spike_memory
        self.consecutive_spikes_trigger = consecutive_spikes_trigger
        self.hold_duration = hold_duration
        
        # S-matrix control
        self.s_freeze_ambiguity_threshold = s_freeze_ambiguity_threshold
        self.s_freeze_improvement_threshold = s_freeze_improvement_threshold
        self.eta_base = eta_base
        self.k_eta = k_eta
        
        # Thread-safe state
        self.lock = threading.Lock()
        
        # Routing state
        self.p_prev = None
        self.prev_winner = None
        self.flip_history = deque(maxlen=flip_history_size)
        self.metrics_buffer = deque(maxlen=metrics_buffer_size)
        
        # Control state
        self.control_state = ControlState(
            gradient_history=deque(maxlen=spike_memory),
            spike_count=0,
            hold_until_tick=0,
            s_matrix_frozen=False,
            last_judge_score=0.0,
            tick=0
        )
        
        # Performance tracking
        self.call_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        self.control_interventions = 0
        
        # Telemetry
        self.logger_enabled = True
        self.telemetry_path = Path("logs/clarity_controller_v2.jsonl")
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"ClarityControllerV2 initialized:")
        print(f"  β: [{beta_min:.2f}, {beta_max:.2f}]")
        print(f"  λ: [{lambda_min:.2f}, {lambda_max:.2f}]") 
        print(f"  Spike guard: {consecutive_spikes_trigger} spikes → hold {hold_duration} ticks")
        print(f"  S-matrix freeze: A* > {s_freeze_ambiguity_threshold}")
    
    def get_beta(self, clarity: float, force_value: Optional[float] = None) -> float:
        """
        Get β (gating sharpness) based on clarity
        β = β_min + (β_max - β_min) * C
        Higher clarity → Higher β → Sharper gating
        """
        if force_value is not None:
            return force_value
        return self.beta_min + (self.beta_max - self.beta_min) * clarity
    
    def get_lambda(self, clarity: float, force_value: Optional[float] = None) -> float:
        """
        Get λ (EMA update rate) based on clarity  
        λ = λ_min + (λ_max - λ_min) * C
        Higher clarity → Higher λ → Less inertia (more responsive)
        """
        if force_value is not None:
            return force_value
        return self.lambda_min + (self.lambda_max - self.lambda_min) * clarity
    
    def get_eta_s(self, clarity: float, frozen: bool = False) -> float:
        """
        Get S-matrix learning rate
        η_S = η_base * (1 + k_eta * C) if not frozen else 0
        """
        if frozen:
            return 0.0
        return self.eta_base * (1.0 + self.k_eta * clarity)
    
    def compute_gradient_norm(self, p_current: np.ndarray, p_prev: Optional[np.ndarray]) -> float:
        """Compute ||∇w|| for spike detection"""
        if p_prev is None:
            return 0.0
        return float(np.linalg.norm(p_current - p_prev))
    
    def detect_spike(self, gradient_norm: float) -> bool:
        """Detect if current gradient is a spike"""
        return gradient_norm > self.spike_threshold
    
    def update_control_state(self, gradient_norm: float, A_star: float, judge_score: Optional[float] = None):
        """Update control state based on gradient and conditions"""
        self.control_state.tick += 1
        self.control_state.gradient_history.append(gradient_norm)
        
        # Spike detection
        current_spike = self.detect_spike(gradient_norm)
        
        if current_spike:
            self.control_state.spike_count += 1
        else:
            self.control_state.spike_count = max(0, self.control_state.spike_count - 1)
        
        # Trigger hold if too many consecutive spikes
        if (self.control_state.spike_count >= self.consecutive_spikes_trigger and 
            self.control_state.tick > self.control_state.hold_until_tick):
            
            self.control_state.hold_until_tick = self.control_state.tick + self.hold_duration
            self.control_interventions += 1
            if self.logger_enabled:
                print(f"⚠️  SPIKE GUARD ACTIVATED: Holding β,λ for {self.hold_duration} ticks "
                      f"(spike count: {self.control_state.spike_count})")
        
        # S-matrix freeze logic
        old_s_frozen = self.control_state.s_matrix_frozen
        
        if A_star > self.s_freeze_ambiguity_threshold:
            # High ambiguity - consider freezing
            if judge_score is not None:
                improvement = judge_score - self.control_state.last_judge_score
                if improvement < self.s_freeze_improvement_threshold:
                    self.control_state.s_matrix_frozen = True
                else:
                    self.control_state.s_matrix_frozen = False
                self.control_state.last_judge_score = judge_score
            else:
                # No judge score - default to freeze on high ambiguity
                self.control_state.s_matrix_frozen = True
        else:
            # Low ambiguity - unfreeze
            self.control_state.s_matrix_frozen = False
        
        if old_s_frozen != self.control_state.s_matrix_frozen and self.logger_enabled:
            status = "FROZEN" if self.control_state.s_matrix_frozen else "UNFROZEN"
            print(f"🧊 S-MATRIX {status}: A*={A_star:.3f}")
    
    def get_control_mode(self) -> str:
        """Get current control mode string"""
        if self.control_state.tick <= self.control_state.hold_until_tick:
            return "holding"
        elif self.control_state.s_matrix_frozen:
            return "s_frozen"
        else:
            return "normal"
    
    def route_with_enhanced_control(self, 
                                   logits: np.ndarray,
                                   ambiguity_score: float,
                                   judge_score: Optional[float] = None,
                                   request_id: Optional[str] = None) -> Tuple[np.ndarray, EnhancedRoutingMetrics]:
        """
        Apply enhanced controlled routing with spike guards and S-matrix control
        
        Args:
            logits: Raw routing logits
            ambiguity_score: Ambiguity score A* [0, 1]
            judge_score: Optional quality score for S-matrix decisions
            request_id: Optional request identifier
            
        Returns:
            routing_weights: Controlled routing distribution
            metrics: Enhanced metrics including control state
        """
        start_time = time.time()
        
        try:
            # Input validation
            if not isinstance(logits, np.ndarray) or len(logits) == 0:
                raise ValueError("Invalid logits array")
            
            ambiguity_score = np.clip(ambiguity_score, 0, 1)
            clarity = 1.0 - ambiguity_score
            
            with self.lock:
                # Determine if we're in hold mode
                in_hold_mode = self.control_state.tick <= self.control_state.hold_until_tick
                
                # Get control parameters (with potential override)
                if in_hold_mode and hasattr(self, '_held_beta'):
                    # Use held values
                    beta = self._held_beta
                    lam = self._held_lambda
                else:
                    # Compute new values
                    beta = self.get_beta(clarity)
                    lam = self.get_lambda(clarity)
                    
                    # Store for potential hold
                    if not in_hold_mode:
                        self._held_beta = beta
                        self._held_lambda = lam
                
                # Apply temperature scaling  
                p_new = self._softmax(beta * logits)
                
                # Apply EMA with adaptive inertia
                if self.p_prev is None:
                    p_route = p_new.copy()
                    gradient_norm = 0.0
                else:
                    p_route = (1 - lam) * self.p_prev + lam * p_new
                    p_route = p_route / (np.sum(p_route) + 1e-9)
                    gradient_norm = self.compute_gradient_norm(p_route, self.p_prev)
                
                # Update control state
                self.update_control_state(gradient_norm, ambiguity_score, judge_score)
                
                # Compute enhanced metrics
                metrics = self._compute_enhanced_metrics(
                    p_route, p_new, ambiguity_score, clarity, beta, lam,
                    gradient_norm, start_time, request_id
                )
                
                # Update routing state
                self.p_prev = p_route.copy()
                
                # Update performance tracking
                self.call_count += 1
                self.total_latency += metrics.latency_ms
                self.metrics_buffer.append(metrics)
            
            return p_route, metrics
            
        except Exception as e:
            with self.lock:
                self.error_count += 1
            
            # Fallback routing
            fallback_routing = self._softmax(logits)
            fallback_metrics = EnhancedRoutingMetrics(
                gating_sensitivity=0.0,
                winner_flip_rate=0.0,
                boundary_distance=1.0 - np.max(fallback_routing),
                routing_entropy=-np.sum(fallback_routing * np.log(fallback_routing + 1e-9)),
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                clarity_score=clarity,
                beta=1.0,
                lambda_val=0.5,
                distance_to_vertex=1.0 - np.max(fallback_routing),
                gradient_norm=0.0,
                spike_detected=False,
                control_state="error",
                expected_G_reduction=0.0,
                mediation_ratio=None
            )
            
            return fallback_routing, fallback_metrics
    
    def _compute_enhanced_metrics(self, 
                                 p_route: np.ndarray,
                                 p_new: np.ndarray,
                                 ambiguity_score: float,
                                 clarity: float,
                                 beta: float,
                                 lam: float,
                                 gradient_norm: float,
                                 start_time: float,
                                 request_id: Optional[str]) -> EnhancedRoutingMetrics:
        """Compute enhanced routing metrics"""
        
        # Core metrics
        if self.p_prev is not None:
            delta_w = np.linalg.norm(p_route - self.p_prev)
            gating_sensitivity = delta_w / 0.01  # Normalized by fixed perturbation
        else:
            gating_sensitivity = 0.0
        
        # Winner flip tracking
        current_winner = np.argmax(p_route)
        winner_changed = self.prev_winner is not None and current_winner != self.prev_winner
        
        with self.lock:
            self.flip_history.append(winner_changed)
            winner_flip_rate = np.mean(self.flip_history) if self.flip_history else 0.0
            self.prev_winner = current_winner
        
        # Enhanced metrics
        boundary_distance = 1.0 - np.max(p_route)
        routing_entropy = -np.sum(p_route * np.log(p_route + 1e-9))
        distance_to_vertex = 1.0 - np.max(p_route)  # Same as boundary_distance
        latency_ms = (time.time() - start_time) * 1000
        
        # Control state
        spike_detected = self.detect_spike(gradient_norm)
        control_mode = self.get_control_mode()
        
        # Expected G reduction based on control settings
        # Baseline G ≈ 3.0 for high ambiguity, we target < 2.0
        baseline_G = 3.0 if ambiguity_score > 0.7 else 1.5
        controlled_G = baseline_G * (beta / 1.6) * (lam / 0.8)  # Rough estimate
        expected_G_reduction = max(0.0, baseline_G - controlled_G)
        
        # Mediation ratio (if we have recent partial correlation data)
        mediation_ratio = self._get_recent_mediation_ratio()
        
        return EnhancedRoutingMetrics(
            gating_sensitivity=gating_sensitivity,
            winner_flip_rate=winner_flip_rate,
            boundary_distance=boundary_distance,
            routing_entropy=routing_entropy,
            latency_ms=latency_ms,
            timestamp=time.time(),
            clarity_score=clarity,
            beta=beta,
            lambda_val=lam,
            distance_to_vertex=distance_to_vertex,
            gradient_norm=gradient_norm,
            spike_detected=spike_detected,
            control_state=control_mode,
            expected_G_reduction=expected_G_reduction,
            mediation_ratio=mediation_ratio
        )
    
    def _get_recent_mediation_ratio(self) -> Optional[float]:
        """Get recent mediation ratio from validation results if available"""
        try:
            results_path = Path("validation/results/enhanced_mediator_results.json")
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                    return data.get('summary', {}).get('mediation_ratio')
        except:
            pass
        return None
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x) + 1e-9)
    
    def get_enhanced_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        with self.lock:
            if self.call_count == 0:
                return {'status': 'no_requests'}
            
            avg_latency = self.total_latency / self.call_count
            error_rate = self.error_count / self.call_count
            intervention_rate = self.control_interventions / self.call_count
            
            # Recent metrics analysis
            if self.metrics_buffer:
                recent_metrics = list(self.metrics_buffer)[-100:]
                avg_G = np.mean([m.gating_sensitivity for m in recent_metrics])
                avg_D = np.mean([m.distance_to_vertex for m in recent_metrics])
                avg_gradient = np.mean([m.gradient_norm for m in recent_metrics])
                spike_rate = np.mean([m.spike_detected for m in recent_metrics])
                avg_expected_reduction = np.mean([m.expected_G_reduction for m in recent_metrics])
            else:
                avg_G = avg_D = avg_gradient = spike_rate = avg_expected_reduction = 0.0
            
            return {
                'call_count': self.call_count,
                'avg_latency_ms': avg_latency,
                'error_rate': error_rate,
                'intervention_rate': intervention_rate,
                'control_stats': {
                    'current_tick': self.control_state.tick,
                    'hold_until_tick': self.control_state.hold_until_tick,
                    'spike_count': self.control_state.spike_count,
                    's_matrix_frozen': self.control_state.s_matrix_frozen,
                    'current_mode': self.get_control_mode()
                },
                'performance_metrics': {
                    'avg_gating_sensitivity': avg_G,
                    'avg_distance_to_vertex': avg_D,
                    'avg_gradient_norm': avg_gradient,
                    'spike_rate': spike_rate,
                    'avg_expected_G_reduction': avg_expected_reduction
                },
                'target_performance': {
                    'target_G_high_ambiguity': 2.0,
                    'current_G': avg_G,
                    'G_reduction_achieved': avg_expected_reduction > 0.5
                }
            }
    
    def log_control_event(self, metrics: EnhancedRoutingMetrics, request_id: Optional[str] = None):
        """Log enhanced control event"""
        if not self.logger_enabled:
            return
        
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request_id,
            'tick': self.control_state.tick,
            'A_star': 1.0 - metrics.clarity_score,
            'clarity': metrics.clarity_score,
            'beta': metrics.beta,
            'lambda': metrics.lambda_val,
            'eta_s': self.get_eta_s(metrics.clarity_score, self.control_state.s_matrix_frozen),
            'G': metrics.gating_sensitivity,
            'L_boundary': metrics.boundary_distance,
            'D_vertex': metrics.distance_to_vertex,
            'gradient_norm': metrics.gradient_norm,
            'spike_detected': metrics.spike_detected,
            'control_state': metrics.control_state,
            'expected_G_reduction': metrics.expected_G_reduction,
            'mediation_ratio': metrics.mediation_ratio,
            's_matrix_frozen': self.control_state.s_matrix_frozen
        }
        
        try:
            with open(self.telemetry_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            print(f"Telemetry logging failed: {e}")
    
    def reset_state(self):
        """Reset all controller state"""
        with self.lock:
            self.p_prev = None
            self.prev_winner = None
            self.flip_history.clear()
            self.metrics_buffer.clear()
            
            self.control_state = ControlState(
                gradient_history=deque(maxlen=self.spike_memory),
                spike_count=0,
                hold_until_tick=0,
                s_matrix_frozen=False,
                last_judge_score=0.0,
                tick=0
            )
            
            self.call_count = 0
            self.total_latency = 0.0
            self.error_count = 0
            self.control_interventions = 0
        
        print("🔄 ClarityControllerV2 state reset")
    
    def validate_control_effectiveness(self, test_logits: np.ndarray, 
                                     high_ambiguity: float = 0.8) -> Dict:
        """
        Validate that the enhanced controller reduces G under high ambiguity
        """
        print("🎯 Validating enhanced controller effectiveness...")
        
        # Test with baseline (no control)
        baseline_weights = self._softmax(test_logits)
        perturbed_logits = test_logits + np.random.randn(*test_logits.shape) * 0.01
        baseline_perturbed = self._softmax(perturbed_logits)
        baseline_G = np.linalg.norm(baseline_perturbed - baseline_weights) / 0.01
        
        # Test with enhanced control
        controlled_weights, metrics = self.route_with_enhanced_control(test_logits, high_ambiguity)
        controlled_perturbed, _ = self.route_with_enhanced_control(perturbed_logits, high_ambiguity)
        controlled_G = np.linalg.norm(controlled_perturbed - controlled_weights) / 0.01
        
        # Analysis
        improvement_factor = baseline_G / (controlled_G + 1e-9)
        G_reduction = baseline_G - controlled_G
        
        validation_results = {
            'baseline_G': float(baseline_G),
            'controlled_G': float(controlled_G),
            'improvement_factor': float(improvement_factor),
            'G_reduction': float(G_reduction),
            'target_achieved': controlled_G < 2.0,  # Target for high ambiguity
            'control_settings': {
                'beta': metrics.beta,
                'lambda': metrics.lambda_val,
                'clarity': metrics.clarity_score,
                'control_state': metrics.control_state
            },
            'expected_L_perp_reduction': float(G_reduction * 0.6)  # Estimated based on ρ(G,L_⊥)
        }
        
        print(f"  Baseline G: {baseline_G:.3f}")
        print(f"  Controlled G: {controlled_G:.3f}")
        print(f"  Improvement: {improvement_factor:.2f}x")
        print(f"  Target achieved: {validation_results['target_achieved']}")
        
        return validation_results


def demonstration():
    """Demonstrate enhanced clarity controller"""
    print("=" * 70)
    print("CLARITY CONTROLLER V2 - ENHANCED WITH SPIKE GUARDS")
    print("Mediation-Informed Control: A* → G → L_⊥")
    print("=" * 70)
    
    controller = ClarityControllerV2()
    
    # Test scenarios
    scenarios = [
        (0.05, "Very low ambiguity (computational)"),
        (0.25, "Low ambiguity (factual)"),
        (0.50, "Medium ambiguity"), 
        (0.75, "High ambiguity (philosophical)"),
        (0.95, "Very high ambiguity (abstract)")
    ]
    
    print("\n🎛️  TESTING CONTROL SCHEDULES:")
    for A_star, description in scenarios:
        clarity = 1.0 - A_star
        beta = controller.get_beta(clarity)
        lam = controller.get_lambda(clarity)
        eta_s = controller.get_eta_s(clarity)
        
        print(f"\n{description}: A*={A_star:.2f}, C={clarity:.2f}")
        print(f"  → β={beta:.2f} (sharpness: {'high' if beta > 1.2 else 'low'})")
        print(f"  → λ={lam:.2f} (inertia: {'low' if lam > 0.6 else 'high'})")
        print(f"  → η_s={eta_s:.4f} (S-matrix LR)")
    
    # Test spike detection
    print(f"\n⚡ TESTING SPIKE GUARD:")
    np.random.seed(42)
    base_logits = np.array([0.5, -0.2, 0.8, -0.5, 0.3])
    
    for i in range(8):
        # Create increasingly noisy logits to trigger spikes
        noise_scale = 0.1 * (i + 1) if i < 4 else 0.05
        logits = base_logits + np.random.randn(5) * noise_scale
        
        weights, metrics = controller.route_with_enhanced_control(logits, 0.8)
        
        spike_indicator = "🔥" if metrics.spike_detected else "  "
        hold_indicator = "⏸️ " if metrics.control_state == "holding" else "  "
        
        print(f"  Tick {i+1}: {spike_indicator} G={metrics.gating_sensitivity:.3f}, "
              f"∇w={metrics.gradient_norm:.3f}, {hold_indicator}{metrics.control_state}")
    
    # Validation test
    print(f"\n🎯 EFFECTIVENESS VALIDATION:")
    validation = controller.validate_control_effectiveness(base_logits, 0.8)
    
    print(f"\nFinal Stats:")
    stats = controller.get_enhanced_performance_stats()
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    demonstration()