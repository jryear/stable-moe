#!/usr/bin/env python3
"""
Enhanced Telemetry Schema for Production MoE Routing
Comprehensive event schema with mediation analysis and production metrics

Key additions:
- Mediation fields: rho_AG, rho_GL, rho_AL_given_GDlenTpl 
- Gradient metrics: grad_w_norm, grad2_w_norm
- PI controller telemetry
- Spike guard and safety metrics
- Measurement hygiene: delta_norm, orthogonal_basis_id
"""

import json
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Task type classification"""
    REASONING = "reasoning"
    CODE = "code" 
    GENERATION = "generation"
    ANALYSIS = "analysis"
    CHAT = "chat"
    UNKNOWN = "unknown"

class PolicyType(Enum):
    """Controller policy types"""
    BASE = "base"
    CTRL_V1 = "ctrl_v1"  
    CTRL_V2 = "ctrl_v2"
    CTRL_V3_PI = "ctrl_v3_pi"
    SHADOW = "shadow"

@dataclass
class ControllerState:
    """Controller configuration and state"""
    policy_id: str
    beta: float
    lambda_val: float  # lambda is reserved keyword
    lr_S: Optional[float] = None
    
    # V3 additions
    pi_controller_active: bool = False
    pi_target_G: Optional[float] = None
    pi_error: Optional[float] = None
    pi_integral: Optional[float] = None
    spike_guard_active: bool = False
    spike_count: int = 0
    s_matrix_frozen: bool = False
    auto_revert_triggered: bool = False

@dataclass 
class CoreMetrics:
    """Core routing stability metrics"""
    # Primary ambiguity measure (composite)
    A_star: float  # [0, 1] ambiguity score
    
    # Routing characteristics
    H_route: float  # routing entropy
    d_w_norm: float  # ||∇w|| - gradient norm for spike detection
    G: float  # gating sensitivity (key mediation variable)
    
    # Lipschitz measures
    L_perp: float  # orthogonal Lipschitz (expansion metric)
    L_para: float  # parallel Lipschitz (for comparison)
    
    # Stability proxies
    self_sim: float  # self-similarity across reruns
    distance_to_vertex: float  # D = 1 - max_i w_i (often better than entropy)
    
    # Effective rank measures  
    R_PR: float  # participation ratio
    R_H: float   # spectral-entropy rank

@dataclass
class MediationMetrics:
    """Enhanced mediation analysis metrics"""
    # Core mediation correlations
    rho_A_G: Optional[float] = None  # A* → G pathway (should be positive)
    p_A_G: Optional[float] = None
    rho_G_L: Optional[float] = None  # G → L_⊥ pathway (should be positive)
    p_G_L: Optional[float] = None
    rho_A_L: Optional[float] = None  # A* → L_⊥ direct (should be positive)
    p_A_L: Optional[float] = None
    
    # Partial correlation (critical mediation test)
    rho_A_L_given_G: Optional[float] = None  # Should → 0 if G mediates
    p_A_L_given_G: Optional[float] = None
    
    # Extended conditioning (for robustness)
    rho_A_L_given_GDlenTpl: Optional[float] = None  # Condition on G, D, length, template
    p_A_L_given_GDlenTpl: Optional[float] = None
    
    # Mediation strength
    mediation_ratio: Optional[float] = None  # |rho_A_L_given_G| / |rho_A_L|
    mediation_confirmed: bool = False
    
    # Bootstrap confidence intervals
    rho_A_G_ci_lower: Optional[float] = None
    rho_A_G_ci_upper: Optional[float] = None
    bootstrap_n: int = 0

@dataclass
class MeasurementHygiene:
    """Measurement hygiene and methodology tracking"""
    # Perturbation control
    delta_norm: float  # ||Δx|| - fixed perturbation magnitude
    delta_x_method: str = "unit_norm_embedding"  # How perturbation was generated
    
    # Orthogonal direction basis
    orthogonal_basis_id: Optional[str] = None  # ID of K orthogonal directions used
    K_orthogonal_dirs: int = 4  # Number of orthogonal directions tested
    
    # Winsorization and robust statistics  
    L_para_winsorized: bool = False  # Whether L_∥ was winsorized per template
    winsorize_percentile: float = 95.0
    
    # Window parameters for rank computation
    rank_window_size: int = 64  # Window size for effective rank
    rank_mean_centered: bool = True
    rank_unit_normed: bool = True

@dataclass
class IOMetrics:
    """Input/output and performance metrics"""
    prompt_len: int
    tokens_out: int
    latency_ms: float
    
    # Enhanced performance metrics
    processing_time_ms: float = 0.0
    queue_wait_ms: float = 0.0
    model_inference_ms: float = 0.0
    routing_overhead_ms: float = 0.0

@dataclass
class QualityLabels:
    """Ground truth quality labels and judgments"""
    judge_score: Optional[float] = None  # [0, 1] quality score
    success: Optional[bool] = None  # Binary success indicator
    human_rating: Optional[float] = None  # Human quality rating if available
    
    # Task-specific quality measures
    code_correctness: Optional[float] = None
    reasoning_validity: Optional[float] = None
    factual_accuracy: Optional[float] = None

@dataclass
class VersionInfo:
    """System version and deployment info"""
    router_git: str
    s_matrix_git: Optional[str] = None
    controller_version: str = "v3_pi_spike_guard"
    deployment_env: str = "production"
    feature_flags: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.feature_flags is None:
            self.feature_flags = {}

@dataclass  
class EnhancedTelemetryEvent:
    """
    Complete telemetry event for production MoE routing
    
    This schema captures all metrics needed for:
    - Real-time monitoring and alerting
    - Mediation analysis and validation
    - Performance optimization
    - Debugging and incident response
    """
    # Core identifiers
    ts: float  # Unix timestamp
    request_id: str
    user_id: Optional[str] = None
    template_id: Optional[str] = None  
    task_type: str = TaskType.UNKNOWN.value
    model_id: str = "unknown"
    
    # Controller state
    controller: ControllerState = None
    
    # Core routing metrics  
    metrics: CoreMetrics = None
    
    # Mediation analysis (updated periodically)
    mediation: Optional[MediationMetrics] = None
    
    # Measurement methodology
    measurement: MeasurementHygiene = None
    
    # Performance and I/O
    io: IOMetrics = None
    
    # Quality labels  
    labels: Optional[QualityLabels] = None
    
    # System versions
    versions: VersionInfo = None
    
    # Additional context
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedTelemetryEvent':
        """Create from dictionary"""
        # Handle nested dataclasses
        if 'controller' in data and data['controller']:
            data['controller'] = ControllerState(**data['controller'])
        if 'metrics' in data and data['metrics']:
            data['metrics'] = CoreMetrics(**data['metrics'])
        if 'mediation' in data and data['mediation']:
            data['mediation'] = MediationMetrics(**data['mediation'])
        if 'measurement' in data and data['measurement']:
            data['measurement'] = MeasurementHygiene(**data['measurement'])
        if 'io' in data and data['io']:
            data['io'] = IOMetrics(**data['io'])
        if 'labels' in data and data['labels']:
            data['labels'] = QualityLabels(**data['labels'])
        if 'versions' in data and data['versions']:
            data['versions'] = VersionInfo(**data['versions'])
            
        return cls(**data)

class TelemetryWriter:
    """Enhanced telemetry writer with multiple output formats"""
    
    def __init__(self, 
                 output_dir: Union[str, Path] = "telemetry/events",
                 formats: List[str] = ["jsonl", "parquet"],
                 buffer_size: int = 100,
                 flush_interval: int = 60):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.formats = formats
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        self.event_buffer: List[EnhancedTelemetryEvent] = []
        self.last_flush = time.time()
        
        logger.info(f"TelemetryWriter initialized: {output_dir}, formats: {formats}")
    
    def write_event(self, event: EnhancedTelemetryEvent):
        """Write telemetry event to buffer"""
        self.event_buffer.append(event)
        
        # Auto-flush if buffer full or interval exceeded
        if (len(self.event_buffer) >= self.buffer_size or 
            time.time() - self.last_flush > self.flush_interval):
            self.flush()
    
    def flush(self):
        """Flush buffered events to storage"""
        if not self.event_buffer:
            return
        
        timestamp_str = time.strftime("%Y%m%d_%H", time.gmtime())
        
        try:
            for format_type in self.formats:
                if format_type == "jsonl":
                    self._write_jsonl(timestamp_str)
                elif format_type == "parquet":
                    self._write_parquet(timestamp_str)
                elif format_type == "json":
                    self._write_json(timestamp_str)
            
            logger.debug(f"Flushed {len(self.event_buffer)} events")
            self.event_buffer.clear()
            self.last_flush = time.time()
            
        except Exception as e:
            logger.error(f"Telemetry flush failed: {e}")
    
    def _write_jsonl(self, timestamp_str: str):
        """Write events as JSONL"""
        output_file = self.output_dir / f"events_{timestamp_str}.jsonl"
        
        with open(output_file, 'a') as f:
            for event in self.event_buffer:
                f.write(event.to_json() + '\n')
    
    def _write_parquet(self, timestamp_str: str):
        """Write events as Parquet (requires pandas)"""
        try:
            import pandas as pd
            
            # Convert to flat dictionary format for Parquet
            records = []
            for event in self.event_buffer:
                record = self._flatten_event(event)
                records.append(record)
            
            df = pd.DataFrame(records)
            output_file = self.output_dir / f"events_{timestamp_str}.parquet"
            
            if output_file.exists():
                # Append to existing file
                existing_df = pd.read_parquet(output_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_parquet(output_file, index=False)
            
        except ImportError:
            logger.warning("Pandas not available, skipping Parquet output")
        except Exception as e:
            logger.error(f"Parquet write failed: {e}")
    
    def _write_json(self, timestamp_str: str):
        """Write events as single JSON file"""
        output_file = self.output_dir / f"events_{timestamp_str}.json"
        
        events_data = [event.to_dict() for event in self.event_buffer]
        
        if output_file.exists():
            # Append to existing file
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            events_data = existing_data + events_data
        
        with open(output_file, 'w') as f:
            json.dump(events_data, f, indent=2)
    
    def _flatten_event(self, event: EnhancedTelemetryEvent) -> Dict[str, Any]:
        """Flatten event for tabular storage"""
        flat = {
            'ts': event.ts,
            'request_id': event.request_id,
            'user_id': event.user_id,
            'template_id': event.template_id,
            'task_type': event.task_type,
            'model_id': event.model_id
        }
        
        # Controller fields
        if event.controller:
            flat.update({
                'controller_policy_id': event.controller.policy_id,
                'controller_beta': event.controller.beta,
                'controller_lambda': event.controller.lambda_val,
                'controller_lr_S': event.controller.lr_S,
                'controller_pi_active': event.controller.pi_controller_active,
                'controller_pi_target_G': event.controller.pi_target_G,
                'controller_pi_error': event.controller.pi_error,
                'controller_spike_guard_active': event.controller.spike_guard_active,
                'controller_auto_revert': event.controller.auto_revert_triggered
            })
        
        # Core metrics
        if event.metrics:
            flat.update({
                'A_star': event.metrics.A_star,
                'H_route': event.metrics.H_route,
                'd_w_norm': event.metrics.d_w_norm,
                'G': event.metrics.G,
                'L_perp': event.metrics.L_perp,
                'L_para': event.metrics.L_para,
                'self_sim': event.metrics.self_sim,
                'distance_to_vertex': event.metrics.distance_to_vertex,
                'R_PR': event.metrics.R_PR,
                'R_H': event.metrics.R_H
            })
        
        # Mediation metrics (if available)
        if event.mediation:
            flat.update({
                'mediation_rho_A_G': event.mediation.rho_A_G,
                'mediation_p_A_G': event.mediation.p_A_G,
                'mediation_rho_G_L': event.mediation.rho_G_L,
                'mediation_p_G_L': event.mediation.p_G_L,
                'mediation_rho_A_L_given_G': event.mediation.rho_A_L_given_G,
                'mediation_confirmed': event.mediation.mediation_confirmed,
                'mediation_ratio': event.mediation.mediation_ratio
            })
        
        # I/O metrics
        if event.io:
            flat.update({
                'prompt_len': event.io.prompt_len,
                'tokens_out': event.io.tokens_out,
                'latency_ms': event.io.latency_ms,
                'processing_time_ms': event.io.processing_time_ms,
                'routing_overhead_ms': event.io.routing_overhead_ms
            })
        
        # Quality labels
        if event.labels:
            flat.update({
                'judge_score': event.labels.judge_score,
                'success': event.labels.success,
                'human_rating': event.labels.human_rating
            })
        
        # Measurement hygiene
        if event.measurement:
            flat.update({
                'delta_norm': event.measurement.delta_norm,
                'orthogonal_basis_id': event.measurement.orthogonal_basis_id,
                'K_orthogonal_dirs': event.measurement.K_orthogonal_dirs
            })
        
        return flat

def create_sample_event() -> EnhancedTelemetryEvent:
    """Create a sample telemetry event for testing"""
    return EnhancedTelemetryEvent(
        ts=time.time(),
        request_id=f"req_{uuid.uuid4().hex[:8]}",
        user_id="user_123",
        template_id="tpl_reasoning",
        task_type=TaskType.REASONING.value,
        model_id="qwen2.5:7b",
        
        controller=ControllerState(
            policy_id="ctrl_v3_pi",
            beta=1.20,
            lambda_val=0.55,
            pi_controller_active=True,
            pi_target_G=0.60,
            pi_error=-0.15,
            spike_guard_active=False
        ),
        
        metrics=CoreMetrics(
            A_star=0.31,
            H_route=0.42,
            d_w_norm=0.18,
            G=0.62,
            L_perp=0.93,
            L_para=1.47,
            self_sim=0.78,
            distance_to_vertex=0.41,
            R_PR=142.3,
            R_H=128.6
        ),
        
        mediation=MediationMetrics(
            rho_A_G=0.546,
            p_A_G=0.035,
            rho_G_L=0.334,
            p_G_L=0.082,
            rho_A_L_given_G=0.15,
            mediation_confirmed=True,
            mediation_ratio=0.73
        ),
        
        measurement=MeasurementHygiene(
            delta_norm=0.01,
            orthogonal_basis_id="basis_k4_gram_schmidt",
            K_orthogonal_dirs=4,
            L_para_winsorized=True
        ),
        
        io=IOMetrics(
            prompt_len=182,
            tokens_out=236,
            latency_ms=812,
            processing_time_ms=28,
            routing_overhead_ms=3
        ),
        
        labels=QualityLabels(
            judge_score=0.86,
            success=True,
            reasoning_validity=0.92
        ),
        
        versions=VersionInfo(
            router_git="abc123def",
            controller_version="v3_pi_spike_guard",
            deployment_env="production"
        )
    )

def main():
    """Demo enhanced telemetry system"""
    print("🚀 Enhanced Telemetry Schema Demo")
    
    # Create writer
    writer = TelemetryWriter(
        output_dir="telemetry/demo",
        formats=["jsonl", "json"],
        buffer_size=5
    )
    
    # Generate sample events
    print("📊 Generating sample events...")
    for i in range(10):
        event = create_sample_event()
        event.request_id = f"demo_req_{i}"
        event.metrics.A_star = 0.2 + (i / 20.0)  # Vary ambiguity
        event.metrics.G = 0.3 + (event.metrics.A_star * 0.5)  # Correlated G
        
        writer.write_event(event)
        print(f"  Event {i}: A*={event.metrics.A_star:.2f}, G={event.metrics.G:.3f}")
    
    # Flush remaining events
    writer.flush()
    
    print(f"✅ Events written to telemetry/demo/")

if __name__ == "__main__":
    main()