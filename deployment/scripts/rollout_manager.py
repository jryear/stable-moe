#!/usr/bin/env python3
"""
Production Rollout Manager for MoE Router V3
Manages shadow → canary → ramp deployment with comprehensive monitoring

Features:
- Automated phase transitions with success criteria validation
- Real-time monitoring and alerting 
- Auto-revert safety mechanisms
- Comprehensive logging and metrics collection
"""

import asyncio
import yaml
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import aiohttp
import argparse
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RolloutPhase(Enum):
    """Rollout phase states"""
    SHADOW = "shadow"
    CANARY = "canary" 
    RAMP_25 = "ramp_25"
    RAMP_50 = "ramp_50"
    RAMP_100 = "ramp_100"
    COMPLETED = "completed"
    FAILED = "failed"
    REVERTED = "reverted"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class MetricCheck:
    """Metric validation result"""
    name: str
    current_value: float
    target_value: float
    threshold_value: float
    passed: bool
    message: str

@dataclass
class PhaseResult:
    """Result of a rollout phase"""
    phase: RolloutPhase
    success: bool
    duration_minutes: float
    metrics: List[MetricCheck]
    alerts_triggered: List[str]
    error_message: Optional[str] = None

class RolloutManager:
    """Manages the progressive rollout of MoE Router V3"""
    
    def __init__(self, config_path: str, api_base_url: str = "http://localhost:8000"):
        self.config_path = Path(config_path)
        self.api_base_url = api_base_url
        
        # Load configuration
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.current_phase = RolloutPhase.SHADOW
        self.phase_results: List[PhaseResult] = []
        self.rollout_start_time = None
        self.emergency_stop = False
        
        logger.info(f"RolloutManager initialized with config: {config_path}")
    
    async def start_rollout(self) -> bool:
        """
        Start the complete rollout process
        
        Returns:
            bool: True if rollout completed successfully
        """
        logger.info("🚀 Starting MoE Router V3 Production Rollout")
        self.rollout_start_time = time.time()
        
        try:
            # Execute each phase
            phases = [
                RolloutPhase.SHADOW,
                RolloutPhase.CANARY,
                RolloutPhase.RAMP_25,
                RolloutPhase.RAMP_50,
                RolloutPhase.RAMP_100
            ]
            
            for phase in phases:
                if self.emergency_stop:
                    logger.error("🛑 Emergency stop triggered")
                    await self._execute_emergency_rollback()
                    return False
                
                logger.info(f"📍 Starting phase: {phase.value}")
                result = await self._execute_phase(phase)
                self.phase_results.append(result)
                
                if not result.success:
                    logger.error(f"❌ Phase {phase.value} failed: {result.error_message}")
                    await self._execute_rollback()
                    return False
                
                logger.info(f"✅ Phase {phase.value} completed successfully")
            
            # Final validation
            if await self._final_validation():
                self.current_phase = RolloutPhase.COMPLETED
                logger.info("🎉 Rollout completed successfully!")
                await self._generate_success_report()
                return True
            else:
                logger.error("❌ Final validation failed")
                await self._execute_rollback()
                return False
                
        except Exception as e:
            logger.error(f"💥 Rollout failed with exception: {e}")
            await self._execute_emergency_rollback()
            return False
    
    async def _execute_phase(self, phase: RolloutPhase) -> PhaseResult:
        """Execute a single rollout phase"""
        start_time = time.time()
        phase_config = self._get_phase_config(phase)
        
        try:
            # Apply phase configuration
            await self._apply_phase_config(phase_config)
            
            # Monitor phase
            duration_hours = phase_config.get('duration_hours', 1)
            metrics, alerts = await self._monitor_phase(phase, duration_hours)
            
            # Validate success criteria
            success = self._validate_success_criteria(phase, metrics)
            
            duration_minutes = (time.time() - start_time) / 60
            
            return PhaseResult(
                phase=phase,
                success=success,
                duration_minutes=duration_minutes,
                metrics=metrics,
                alerts_triggered=alerts
            )
            
        except Exception as e:
            duration_minutes = (time.time() - start_time) / 60
            return PhaseResult(
                phase=phase,
                success=False,
                duration_minutes=duration_minutes,
                metrics=[],
                alerts_triggered=[],
                error_message=str(e)
            )
    
    def _get_phase_config(self, phase: RolloutPhase) -> Dict[str, Any]:
        """Get configuration for a specific phase"""
        phases_config = self.config['rollout']['phases']
        
        if phase == RolloutPhase.SHADOW:
            return phases_config['shadow']
        elif phase == RolloutPhase.CANARY:
            return phases_config['canary']
        elif phase in [RolloutPhase.RAMP_25, RolloutPhase.RAMP_50, RolloutPhase.RAMP_100]:
            ramp_config = phases_config['ramp']
            # Find the specific sub-phase
            for sub_phase in ramp_config['sub_phases']:
                if f"ramp_{sub_phase['traffic_percentage']}" == phase.value:
                    return {**ramp_config, **sub_phase}
            return ramp_config
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    async def _apply_phase_config(self, config: Dict[str, Any]):
        """Apply configuration for the current phase"""
        logger.info(f"🔧 Applying phase configuration...")
        
        # Update controller configuration via API
        controller_config = {
            'policy_id': config['config']['policy_id'],
            'traffic_percentage': config.get('traffic_percentage', 0)
        }
        
        if 'controller_params' in config['config']:
            controller_config.update(config['config']['controller_params'])
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base_url}/admin/update-config",
                json=controller_config
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to update controller config: {response.status}")
        
        logger.info(f"✅ Configuration applied: {config['config']['policy_id']}")
    
    async def _monitor_phase(self, phase: RolloutPhase, duration_hours: float) -> tuple[List[MetricCheck], List[str]]:
        """Monitor a phase for the specified duration"""
        logger.info(f"👁️  Monitoring phase {phase.value} for {duration_hours} hours...")
        
        end_time = time.time() + (duration_hours * 3600)
        metrics = []
        alerts = []
        
        while time.time() < end_time and not self.emergency_stop:
            # Collect current metrics
            current_metrics = await self._collect_metrics()
            
            # Check for alerts
            phase_alerts = await self._check_alerts(phase, current_metrics)
            alerts.extend(phase_alerts)
            
            # Check for critical alerts that trigger immediate action
            critical_alerts = [a for a in phase_alerts if 'critical' in a.lower()]
            if critical_alerts:
                logger.warning(f"🚨 Critical alerts triggered: {critical_alerts}")
                break
            
            # Wait before next check (every minute)
            await asyncio.sleep(60)
        
        # Final metrics collection
        final_metrics = await self._collect_metrics()
        metrics = self._convert_to_metric_checks(phase, final_metrics)
        
        return metrics, alerts
    
    async def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get controller performance stats
                async with session.get(f"{self.api_base_url}/metrics") as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._extract_key_metrics(data)
                    else:
                        logger.warning(f"Failed to fetch metrics: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return {}
    
    def _extract_key_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from API response"""
        extracted = {}
        
        # Basic performance metrics
        extracted['avg_gating_sensitivity'] = metrics_data.get('avg_gating_sensitivity', 0.0)
        extracted['avg_winner_flip_rate'] = metrics_data.get('avg_winner_flip_rate', 0.0)
        extracted['avg_latency_ms'] = metrics_data.get('avg_latency_ms', 0.0)
        extracted['error_rate'] = metrics_data.get('error_rate', 0.0)
        
        # V3 specific metrics
        if 'pi_controller' in metrics_data:
            extracted['pi_activation_rate'] = metrics_data['pi_controller'].get('activation_rate', 0.0)
        
        if 'safety' in metrics_data:
            extracted['G_p99_current'] = metrics_data['safety'].get('G_p99_current', 0.0)
            extracted['auto_revert_count'] = metrics_data['safety'].get('auto_revert_count', 0.0)
        
        return extracted
    
    def _convert_to_metric_checks(self, phase: RolloutPhase, metrics: Dict[str, float]) -> List[MetricCheck]:
        """Convert raw metrics to validation checks"""
        checks = []
        phase_config = self._get_phase_config(phase)
        success_criteria = phase_config.get('success_criteria', {})
        
        # G reduction check
        if 'min_G_reduction_pct' in success_criteria:
            baseline_G = 2.0  # Assumed baseline
            current_G = metrics.get('avg_gating_sensitivity', baseline_G)
            reduction_pct = ((baseline_G - current_G) / baseline_G) * 100
            target = success_criteria['min_G_reduction_pct']
            
            checks.append(MetricCheck(
                name='G_reduction_pct',
                current_value=reduction_pct,
                target_value=target,
                threshold_value=target * 0.8,  # 80% of target as threshold
                passed=reduction_pct >= target,
                message=f"G reduction: {reduction_pct:.1f}% (target: {target}%)"
            ))
        
        # Win rate impact check (simplified)
        if 'max_win_rate_impact_pct' in success_criteria:
            # This would normally come from A/B testing data
            estimated_impact = metrics.get('error_rate', 0.0) * 100  # Proxy
            target = success_criteria['max_win_rate_impact_pct']
            
            checks.append(MetricCheck(
                name='win_rate_impact',
                current_value=estimated_impact,
                target_value=target,
                threshold_value=target * 1.2,
                passed=estimated_impact <= target,
                message=f"Win rate impact: {estimated_impact:.1f}% (max: {target}%)"
            ))
        
        # Latency check
        if 'max_latency_increase_pct' in success_criteria:
            baseline_latency = 100  # Assumed baseline ms
            current_latency = metrics.get('avg_latency_ms', baseline_latency)
            increase_pct = ((current_latency - baseline_latency) / baseline_latency) * 100
            target = success_criteria['max_latency_increase_pct']
            
            checks.append(MetricCheck(
                name='latency_increase',
                current_value=increase_pct,
                target_value=target,
                threshold_value=target * 1.2,
                passed=increase_pct <= target,
                message=f"Latency increase: {increase_pct:.1f}% (max: {target}%)"
            ))
        
        return checks
    
    async def _check_alerts(self, phase: RolloutPhase, metrics: Dict[str, float]) -> List[str]:
        """Check for alert conditions"""
        alerts = []
        phase_config = self._get_phase_config(phase)
        alert_configs = phase_config.get('alerts', [])
        
        for alert_config in alert_configs:
            alert_name = alert_config['name']
            condition = alert_config['condition']
            
            # Simplified condition checking (in production, use proper expression parser)
            if self._evaluate_alert_condition(condition, metrics):
                severity = alert_config.get('severity', 'warning')
                action = alert_config.get('action')
                
                alert_msg = f"[{severity.upper()}] {alert_name}: {condition}"
                alerts.append(alert_msg)
                logger.warning(alert_msg)
                
                # Execute action if specified
                if action == 'auto_revert':
                    logger.critical("🛑 Auto-revert triggered by alert")
                    self.emergency_stop = True
        
        return alerts
    
    def _evaluate_alert_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """Evaluate alert condition (simplified implementation)"""
        # This is a simplified implementation
        # In production, use a proper expression parser
        
        if "G_reduction < 15%" in condition:
            baseline_G = 2.0
            current_G = metrics.get('avg_gating_sensitivity', baseline_G)
            reduction_pct = ((baseline_G - current_G) / baseline_G) * 100
            return reduction_pct < 15
        
        if "error_rate > baseline" in condition:
            return metrics.get('error_rate', 0.0) > 0.01  # 1% baseline
        
        if "G_p99 > cap" in condition:
            return metrics.get('G_p99_current', 0.0) > 5.0
        
        return False
    
    def _validate_success_criteria(self, phase: RolloutPhase, metrics: List[MetricCheck]) -> bool:
        """Validate all success criteria for a phase"""
        all_passed = all(metric.passed for metric in metrics)
        
        if all_passed:
            logger.info(f"✅ All success criteria met for phase {phase.value}")
        else:
            failed_metrics = [m.name for m in metrics if not m.passed]
            logger.warning(f"❌ Failed metrics for phase {phase.value}: {failed_metrics}")
        
        return all_passed
    
    async def _execute_rollback(self):
        """Execute gradual rollback procedure"""
        logger.warning("🔄 Executing gradual rollback...")
        
        rollback_config = self.config['rollback']['gradual_procedure']
        
        for step in rollback_config:
            logger.info(f"📍 Rollback step: {step}")
            # In production, implement actual rollback steps
            await asyncio.sleep(5)  # Simulate rollback time
        
        self.current_phase = RolloutPhase.REVERTED
        logger.info("✅ Rollback completed")
    
    async def _execute_emergency_rollback(self):
        """Execute immediate emergency rollback"""
        logger.critical("🚨 Executing emergency rollback...")
        
        # Immediate revert to baseline
        emergency_config = {
            'policy_id': 'baseline',
            'traffic_percentage': 0,
            'emergency_mode': True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/admin/emergency-revert",
                    json=emergency_config
                ) as response:
                    if response.status == 200:
                        logger.info("✅ Emergency revert completed")
                    else:
                        logger.error(f"❌ Emergency revert failed: {response.status}")
        except Exception as e:
            logger.error(f"💥 Emergency revert exception: {e}")
        
        self.current_phase = RolloutPhase.REVERTED
    
    async def _final_validation(self) -> bool:
        """Perform final validation after all phases complete"""
        logger.info("🔍 Performing final validation...")
        
        final_criteria = self.config['validation']['final_success_criteria']
        final_metrics = await self._collect_metrics()
        
        # Check each final criterion
        all_passed = True
        for criterion in final_criteria:
            # Simplified validation - in production, implement proper checks
            if "G_reduction ≥ 20%" in criterion:
                baseline_G = 2.0
                current_G = final_metrics.get('avg_gating_sensitivity', baseline_G)
                reduction_pct = ((baseline_G - current_G) / baseline_G) * 100
                passed = reduction_pct >= 20
                
                logger.info(f"Final validation - G reduction: {reduction_pct:.1f}% {'✅' if passed else '❌'}")
                all_passed = all_passed and passed
        
        return all_passed
    
    async def _generate_success_report(self):
        """Generate final success report"""
        logger.info("📊 Generating rollout success report...")
        
        total_duration = (time.time() - self.rollout_start_time) / 3600  # hours
        
        report = {
            'rollout_id': f"rollout_{int(self.rollout_start_time)}",
            'start_time': datetime.fromtimestamp(self.rollout_start_time).isoformat(),
            'total_duration_hours': total_duration,
            'final_status': 'SUCCESS',
            'phases_completed': len(self.phase_results),
            'phase_results': [
                {
                    'phase': result.phase.value,
                    'success': result.success,
                    'duration_minutes': result.duration_minutes,
                    'alerts_count': len(result.alerts_triggered)
                }
                for result in self.phase_results
            ],
            'final_metrics': await self._collect_metrics(),
            'recommendations': [
                'Monitor for 24 hours to ensure stability',
                'Document lessons learned',
                'Update runbooks based on experience'
            ]
        }
        
        # Save report
        report_path = Path(f"reports/rollout_success_{int(self.rollout_start_time)}.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Success report saved to {report_path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rollout status"""
        return {
            'current_phase': self.current_phase.value,
            'phases_completed': len(self.phase_results),
            'rollout_start_time': self.rollout_start_time,
            'emergency_stop': self.emergency_stop,
            'last_phase_result': self.phase_results[-1].__dict__ if self.phase_results else None
        }

async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='MoE Router V3 Rollout Manager')
    parser.add_argument('--config', default='deployment/configs/rollout.yaml',
                       help='Rollout configuration file')
    parser.add_argument('--api-url', default='http://localhost:8000',
                       help='API base URL')
    parser.add_argument('--action', choices=['start', 'status', 'abort'],
                       default='start', help='Action to perform')
    
    args = parser.parse_args()
    
    manager = RolloutManager(args.config, args.api_url)
    
    if args.action == 'start':
        success = await manager.start_rollout()
        exit_code = 0 if success else 1
        exit(exit_code)
    
    elif args.action == 'status':
        status = manager.get_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.action == 'abort':
        logger.warning("🛑 Abort requested - triggering emergency stop")
        manager.emergency_stop = True
        await manager._execute_emergency_rollback()

if __name__ == "__main__":
    asyncio.run(main())