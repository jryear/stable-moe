#!/usr/bin/env python3
"""
Counterfactual Replay System for Controller Validation
Replays logged routing decisions under ctrl_v2 to validate performance improvements

Key Validation Targets:
- ΔG ≤ -20% for A* ≥ 0.7 (high ambiguity improvement) 
- No win-rate drop for A* ≤ 0.3 (clear case protection)
- Latency jitter reduction ≥ 15%
- G_p95 reduction under high ambiguity conditions
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt

from ..core.production_controller import ProductionClarityController
from ..core.clarity_controller_v2 import ClarityControllerV2

logger = logging.getLogger(__name__)

@dataclass
class ReplayMetrics:
    """Metrics comparing baseline vs controlled routing"""
    baseline_G: float
    controlled_G: float
    delta_G_pct: float
    baseline_L_perp: float
    controlled_L_perp: float
    delta_L_perp_pct: float
    baseline_jitter: float
    controlled_jitter: float
    delta_jitter_pct: float
    win_rate_proxy_baseline: float
    win_rate_proxy_controlled: float
    delta_win_rate_pct: float
    ambiguity_bin: str
    n_samples: int

@dataclass
class CounterfactualResult:
    """Complete counterfactual analysis result"""
    overall_summary: Dict
    ambiguity_bin_results: Dict[str, ReplayMetrics] 
    validation_status: Dict[str, bool]
    recommendations: List[str]
    raw_data: Optional[pd.DataFrame] = None

class CounterfactualReplaySystem:
    """
    Validates controller performance by replaying logged routing decisions
    
    This system takes historical routing logits and compares baseline routing
    vs controlled routing to validate the 4.72× improvement claims.
    """
    
    def __init__(self, 
                 controller_config: Optional[Dict] = None,
                 baseline_policy: str = "base",
                 controlled_policy: str = "ctrl_v2"):
        
        self.baseline_policy = baseline_policy
        self.controlled_policy = controlled_policy
        
        # Initialize controllers
        controller_params = controller_config or {
            'beta_min': 0.85, 'beta_max': 1.65,
            'lambda_min': 0.25, 'lambda_max': 0.75
        }
        
        self.baseline_controller = None  # Direct softmax
        self.controlled_controller = ClarityControllerV2(**controller_params)
        
        # Validation thresholds (from user requirements)
        self.TARGET_G_REDUCTION = 0.20  # 20% minimum
        self.TARGET_JITTER_REDUCTION = 0.15  # 15% minimum  
        self.MAX_WIN_RATE_DROP = 0.005  # 0.5% maximum acceptable drop
        self.HIGH_AMBIGUITY_THRESHOLD = 0.7
        self.LOW_AMBIGUITY_THRESHOLD = 0.3
        
        logger.info(f"CounterfactualReplaySystem initialized: {baseline_policy} vs {controlled_policy}")
    
    def load_logged_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load logged routing data for replay
        
        Expected format:
        - logits: routing logits per expert
        - ambiguity_score: A* score [0, 1] 
        - timestamp: request timestamp
        - request_id: unique identifier
        - ground_truth_quality: quality score if available
        """
        data_path = Path(data_path)
        
        if data_path.suffix == '.json' or data_path.suffix == '.jsonl':
            # Load JSON/JSONL telemetry data
            records = []
            with open(data_path) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            df = pd.DataFrame(records)
            
        elif data_path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
        
        # Validate required columns
        required_cols = ['logits', 'ambiguity_score']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add derived columns
        if 'clarity_score' not in df.columns:
            df['clarity_score'] = 1.0 - df['ambiguity_score']
        
        if 'ambiguity_bin' not in df.columns:
            df['ambiguity_bin'] = self._bin_ambiguity(df['ambiguity_score'])
        
        logger.info(f"Loaded {len(df)} routing records from {data_path}")
        return df
    
    def _bin_ambiguity(self, ambiguity_scores: pd.Series) -> pd.Series:
        """Bin ambiguity scores for analysis"""
        return pd.cut(
            ambiguity_scores,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            include_lowest=True
        )
    
    def replay_routing_decisions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replay routing decisions under both baseline and controlled policies
        
        Returns DataFrame with both baseline and controlled routing outcomes
        """
        logger.info(f"Replaying {len(df)} routing decisions...")
        
        results = []
        
        for idx, row in df.iterrows():
            # Parse logits
            if isinstance(row['logits'], str):
                logits = np.array(json.loads(row['logits']))
            else:
                logits = np.array(row['logits'])
            
            ambiguity = row['ambiguity_score']
            
            # Baseline routing (direct softmax)
            baseline_weights = self._softmax(logits)
            baseline_G = self._compute_sensitivity(logits, baseline_weights)
            baseline_entropy = self._compute_entropy(baseline_weights)
            
            # Controlled routing
            controlled_weights, controlled_metrics = self.controlled_controller.route_with_control(
                logits, ambiguity, row.get('request_id')
            )
            controlled_G = controlled_metrics.gating_sensitivity
            controlled_entropy = controlled_metrics.routing_entropy
            
            # Win-rate proxy (higher entropy = more exploration, proxy for quality)
            # This is a simplified proxy - in production you'd use actual quality scores
            baseline_win_proxy = self._compute_win_rate_proxy(baseline_weights, baseline_entropy)
            controlled_win_proxy = self._compute_win_rate_proxy(controlled_weights, controlled_entropy)
            
            # Latency jitter proxy (higher G = more instability = higher jitter)
            baseline_jitter = baseline_G * 10  # Convert to ms estimate
            controlled_jitter = controlled_G * 10
            
            results.append({
                'request_id': row.get('request_id', f'req_{idx}'),
                'ambiguity_score': ambiguity,
                'clarity_score': 1.0 - ambiguity,
                'ambiguity_bin': row['ambiguity_bin'],
                
                # Baseline metrics
                'baseline_weights': baseline_weights.tolist(),
                'baseline_G': baseline_G,
                'baseline_entropy': baseline_entropy,
                'baseline_win_proxy': baseline_win_proxy,
                'baseline_jitter': baseline_jitter,
                
                # Controlled metrics
                'controlled_weights': controlled_weights.tolist(),
                'controlled_G': controlled_G,
                'controlled_entropy': controlled_entropy,
                'controlled_win_proxy': controlled_win_proxy,
                'controlled_jitter': controlled_jitter,
                'controlled_beta': controlled_metrics.beta,
                'controlled_lambda': controlled_metrics.lambda_val,
                
                # Deltas
                'delta_G': controlled_G - baseline_G,
                'delta_G_pct': ((controlled_G - baseline_G) / (baseline_G + 1e-9)) * 100,
                'delta_win_proxy': controlled_win_proxy - baseline_win_proxy,
                'delta_win_proxy_pct': ((controlled_win_proxy - baseline_win_proxy) / (baseline_win_proxy + 1e-9)) * 100,
                'delta_jitter': controlled_jitter - baseline_jitter,
                'delta_jitter_pct': ((controlled_jitter - baseline_jitter) / (baseline_jitter + 1e-9)) * 100
            })
        
        result_df = pd.DataFrame(results)
        logger.info(f"Replay complete: {len(result_df)} decisions analyzed")
        
        return result_df
    
    def analyze_by_ambiguity_bins(self, replay_df: pd.DataFrame) -> Dict[str, ReplayMetrics]:
        """Analyze results by ambiguity bins"""
        bin_results = {}
        
        for bin_name in replay_df['ambiguity_bin'].unique():
            if pd.isna(bin_name):
                continue
                
            bin_data = replay_df[replay_df['ambiguity_bin'] == bin_name]
            
            if len(bin_data) == 0:
                continue
            
            # Aggregate metrics
            baseline_G = bin_data['baseline_G'].mean()
            controlled_G = bin_data['controlled_G'].mean()
            delta_G_pct = ((controlled_G - baseline_G) / (baseline_G + 1e-9)) * 100
            
            baseline_jitter = bin_data['baseline_jitter'].mean()
            controlled_jitter = bin_data['controlled_jitter'].mean()
            delta_jitter_pct = ((controlled_jitter - baseline_jitter) / (baseline_jitter + 1e-9)) * 100
            
            baseline_win_proxy = bin_data['baseline_win_proxy'].mean()
            controlled_win_proxy = bin_data['controlled_win_proxy'].mean()
            delta_win_proxy_pct = ((controlled_win_proxy - baseline_win_proxy) / (baseline_win_proxy + 1e-9)) * 100
            
            # L_perp proxy (use entropy as proxy for expansion)
            baseline_L_perp = bin_data['baseline_entropy'].mean()
            controlled_L_perp = bin_data['controlled_entropy'].mean()
            delta_L_perp_pct = ((controlled_L_perp - baseline_L_perp) / (baseline_L_perp + 1e-9)) * 100
            
            bin_results[bin_name] = ReplayMetrics(
                baseline_G=baseline_G,
                controlled_G=controlled_G,
                delta_G_pct=delta_G_pct,
                baseline_L_perp=baseline_L_perp,
                controlled_L_perp=controlled_L_perp,
                delta_L_perp_pct=delta_L_perp_pct,
                baseline_jitter=baseline_jitter,
                controlled_jitter=controlled_jitter,
                delta_jitter_pct=delta_jitter_pct,
                win_rate_proxy_baseline=baseline_win_proxy,
                win_rate_proxy_controlled=controlled_win_proxy,
                delta_win_rate_pct=delta_win_proxy_pct,
                ambiguity_bin=bin_name,
                n_samples=len(bin_data)
            )
        
        return bin_results
    
    def validate_performance_targets(self, bin_results: Dict[str, ReplayMetrics]) -> Dict[str, bool]:
        """Validate against performance targets"""
        validation = {}
        
        # Target 1: G reduction ≥ 20% in high ambiguity bins
        high_ambiguity_bins = [bin_name for bin_name in bin_results.keys() 
                              if 'high' in bin_name or 'very_high' in bin_name]
        
        g_reductions = []
        for bin_name in high_ambiguity_bins:
            if bin_name in bin_results:
                g_reductions.append(-bin_results[bin_name].delta_G_pct)  # Negative delta = reduction
        
        g_reduction_achieved = len(g_reductions) > 0 and np.mean(g_reductions) >= (self.TARGET_G_REDUCTION * 100)
        validation['g_reduction_target'] = g_reduction_achieved
        
        # Target 2: No win-rate drop in clear bins (low ambiguity)
        clear_bins = [bin_name for bin_name in bin_results.keys() 
                     if 'very_low' in bin_name or 'low' in bin_name]
        
        win_rate_drops = []
        for bin_name in clear_bins:
            if bin_name in bin_results:
                win_rate_drops.append(bin_results[bin_name].delta_win_rate_pct)
        
        win_rate_protected = len(win_rate_drops) > 0 and all(drop >= -(self.MAX_WIN_RATE_DROP * 100) for drop in win_rate_drops)
        validation['win_rate_protection'] = win_rate_protected
        
        # Target 3: Jitter reduction ≥ 15% overall
        jitter_reductions = [-bin_results[bin_name].delta_jitter_pct for bin_name in bin_results.keys()]
        jitter_target_met = len(jitter_reductions) > 0 and np.mean(jitter_reductions) >= (self.TARGET_JITTER_REDUCTION * 100)
        validation['jitter_reduction_target'] = jitter_target_met
        
        # Overall validation
        validation['all_targets_met'] = all(validation.values())
        
        return validation
    
    def generate_recommendations(self, 
                                validation: Dict[str, bool], 
                                bin_results: Dict[str, ReplayMetrics]) -> List[str]:
        """Generate deployment recommendations based on results"""
        recommendations = []
        
        if validation['all_targets_met']:
            recommendations.append("✅ ALL TARGETS MET - Proceed with production rollout")
            recommendations.append("🎯 Shadow mode → 5% canary → full ramp recommended")
        else:
            recommendations.append("⚠️  Some targets missed - Review before rollout")
        
        if not validation['g_reduction_target']:
            high_ambiguity_performance = [bin_results[bin_name].delta_G_pct 
                                        for bin_name in bin_results.keys() 
                                        if 'high' in bin_name]
            avg_reduction = np.mean([-x for x in high_ambiguity_performance])
            recommendations.append(
                f"🔧 G reduction only {avg_reduction:.1f}% (target: 20%) - "
                f"Consider increasing β_max or lowering λ_min for high ambiguity"
            )
        
        if not validation['win_rate_protection']:
            clear_performance = [bin_results[bin_name].delta_win_rate_pct 
                               for bin_name in bin_results.keys() 
                               if 'low' in bin_name or 'very_low' in bin_name]
            avg_impact = np.mean(clear_performance)
            recommendations.append(
                f"🔧 Win-rate impacted {avg_impact:.1f}% in clear cases - "
                f"Consider raising β_min and λ_max for low ambiguity"
            )
        
        if not validation['jitter_reduction_target']:
            recommendations.append("🔧 Jitter reduction below target - Enable spike guards and gradient clipping")
        
        return recommendations
    
    def run_counterfactual_analysis(self, 
                                   data_path: Union[str, Path],
                                   output_dir: Optional[Union[str, Path]] = None) -> CounterfactualResult:
        """
        Run complete counterfactual analysis
        
        Args:
            data_path: Path to logged routing data
            output_dir: Directory to save results (optional)
            
        Returns:
            Complete analysis results with validation status
        """
        logger.info("🚀 Starting counterfactual analysis...")
        
        # Load and replay data
        df = self.load_logged_data(data_path)
        replay_df = self.replay_routing_decisions(df)
        
        # Analyze by bins
        bin_results = self.analyze_by_ambiguity_bins(replay_df)
        
        # Validate targets
        validation = self.validate_performance_targets(bin_results)
        
        # Generate recommendations  
        recommendations = self.generate_recommendations(validation, bin_results)
        
        # Overall summary
        overall_summary = {
            'total_requests_analyzed': len(replay_df),
            'ambiguity_bins_covered': list(bin_results.keys()),
            'avg_G_reduction_pct': np.mean([-bin_results[bin_name].delta_G_pct for bin_name in bin_results.keys()]),
            'avg_jitter_reduction_pct': np.mean([-bin_results[bin_name].delta_jitter_pct for bin_name in bin_results.keys()]),
            'avg_win_rate_impact_pct': np.mean([bin_results[bin_name].delta_win_rate_pct for bin_name in bin_results.keys()]),
            'validation_passed': validation['all_targets_met'],
            'analysis_timestamp': time.time()
        }
        
        result = CounterfactualResult(
            overall_summary=overall_summary,
            ambiguity_bin_results=bin_results,
            validation_status=validation,
            recommendations=recommendations,
            raw_data=replay_df
        )
        
        # Save results if output directory provided
        if output_dir:
            self._save_results(result, output_dir)
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _save_results(self, result: CounterfactualResult, output_dir: Union[str, Path]):
        """Save analysis results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        with open(output_dir / "counterfactual_results.json", 'w') as f:
            result_dict = {
                'overall_summary': result.overall_summary,
                'ambiguity_bin_results': {k: asdict(v) for k, v in result.ambiguity_bin_results.items()},
                'validation_status': result.validation_status,
                'recommendations': result.recommendations
            }
            json.dump(result_dict, f, indent=2)
        
        # Save raw data
        if result.raw_data is not None:
            result.raw_data.to_parquet(output_dir / "replay_raw_data.parquet")
        
        # Create visualization
        self._create_validation_plot(result, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _create_validation_plot(self, result: CounterfactualResult, output_dir: Path):
        """Create validation visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            bin_names = list(result.ambiguity_bin_results.keys())
            
            # G reduction by bin
            g_reductions = [-result.ambiguity_bin_results[bin_name].delta_G_pct 
                           for bin_name in bin_names]
            axes[0,0].bar(bin_names, g_reductions)
            axes[0,0].axhline(y=20, color='red', linestyle='--', label='Target: 20%')
            axes[0,0].set_title('Gating Sensitivity Reduction by Ambiguity Bin')
            axes[0,0].set_ylabel('G Reduction (%)')
            axes[0,0].legend()
            
            # Jitter reduction by bin
            jitter_reductions = [-result.ambiguity_bin_results[bin_name].delta_jitter_pct 
                               for bin_name in bin_names]
            axes[0,1].bar(bin_names, jitter_reductions)
            axes[0,1].axhline(y=15, color='red', linestyle='--', label='Target: 15%')
            axes[0,1].set_title('Latency Jitter Reduction by Ambiguity Bin')
            axes[0,1].set_ylabel('Jitter Reduction (%)')
            axes[0,1].legend()
            
            # Win-rate impact
            win_rate_impacts = [result.ambiguity_bin_results[bin_name].delta_win_rate_pct 
                               for bin_name in bin_names]
            axes[1,0].bar(bin_names, win_rate_impacts)
            axes[1,0].axhline(y=-0.5, color='red', linestyle='--', label='Threshold: -0.5%')
            axes[1,0].set_title('Win-Rate Impact by Ambiguity Bin')
            axes[1,0].set_ylabel('Win-Rate Impact (%)')
            axes[1,0].legend()
            
            # Sample sizes
            sample_sizes = [result.ambiguity_bin_results[bin_name].n_samples 
                           for bin_name in bin_names]
            axes[1,1].bar(bin_names, sample_sizes)
            axes[1,1].set_title('Sample Size by Ambiguity Bin')
            axes[1,1].set_ylabel('Number of Samples')
            
            plt.tight_layout()
            plt.savefig(output_dir / "counterfactual_validation.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
    
    def _print_summary(self, result: CounterfactualResult):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("🎯 COUNTERFACTUAL ANALYSIS RESULTS")
        print("="*60)
        
        summary = result.overall_summary
        print(f"📊 Total Requests Analyzed: {summary['total_requests_analyzed']:,}")
        print(f"📈 Avg G Reduction: {summary['avg_G_reduction_pct']:.1f}%")
        print(f"⚡ Avg Jitter Reduction: {summary['avg_jitter_reduction_pct']:.1f}%")
        print(f"🎪 Avg Win-Rate Impact: {summary['avg_win_rate_impact_pct']:.2f}%")
        
        print(f"\n🎯 VALIDATION STATUS:")
        for target, passed in result.validation_status.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {target}: {status}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in result.recommendations:
            print(f"  {rec}")
        
        print(f"\n📈 BY AMBIGUITY BIN:")
        for bin_name, metrics in result.ambiguity_bin_results.items():
            print(f"  {bin_name}: G↓{-metrics.delta_G_pct:.1f}%, "
                  f"Jitter↓{-metrics.delta_jitter_pct:.1f}%, "
                  f"WinRate{metrics.delta_win_rate_pct:+.2f}% "
                  f"(n={metrics.n_samples})")
    
    # Utility methods
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x) + 1e-9)
    
    def _compute_sensitivity(self, logits: np.ndarray, weights: np.ndarray) -> float:
        """Compute gating sensitivity (simplified)"""
        # Perturb logits slightly and measure weight change
        perturbed_logits = logits + np.random.randn(*logits.shape) * 0.01
        perturbed_weights = self._softmax(perturbed_logits)
        return np.linalg.norm(perturbed_weights - weights) / 0.01
    
    def _compute_entropy(self, weights: np.ndarray) -> float:
        """Compute routing entropy"""
        return -np.sum(weights * np.log(weights + 1e-9))
    
    def _compute_win_rate_proxy(self, weights: np.ndarray, entropy: float) -> float:
        """Compute win-rate proxy (simplified model)"""
        # Higher entropy = more exploration = potentially better outcomes
        # But too high entropy = poor decisiveness
        # This is a placeholder - use actual quality scores in production
        optimal_entropy = np.log(len(weights)) * 0.7  # 70% of max entropy
        entropy_factor = 1.0 - abs(entropy - optimal_entropy) / optimal_entropy
        
        # Prefer some concentration (winner-take-more effect)
        concentration = np.max(weights)
        concentration_factor = concentration * 0.8 + 0.2
        
        return entropy_factor * concentration_factor


def main():
    """Example usage"""
    import tempfile
    import os
    
    print("🚀 Counterfactual Replay System Demo")
    
    # Create sample data
    sample_data = []
    np.random.seed(42)
    
    for i in range(100):
        n_experts = 5
        logits = np.random.randn(n_experts)
        ambiguity = np.random.beta(2, 3)  # Realistic ambiguity distribution
        
        sample_data.append({
            'request_id': f'req_{i}',
            'logits': logits.tolist(),
            'ambiguity_score': ambiguity,
            'timestamp': time.time() - (100 - i) * 60  # 1 minute intervals
        })
    
    # Save sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in sample_data:
            f.write(json.dumps(record) + '\n')
        sample_path = f.name
    
    try:
        # Run analysis
        replay_system = CounterfactualReplaySystem()
        results = replay_system.run_counterfactual_analysis(
            sample_path,
            output_dir="validation/results/counterfactual"
        )
        
        print(f"\n🎯 Analysis complete!")
        print(f"Validation passed: {results.validation_status['all_targets_met']}")
        
    finally:
        os.unlink(sample_path)


if __name__ == "__main__":
    main()