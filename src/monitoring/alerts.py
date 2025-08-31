#!/usr/bin/env python3
"""
Comprehensive Alerting System for MoE Router V3
Production-ready alerting with auto-revert and escalation policies

Key Features:
- Real-time metric monitoring with configurable thresholds
- Auto-revert triggers for critical failures
- Multi-channel notifications (Slack, PagerDuty, Email)
- Alert correlation and deduplication
- Mediation-aware alerting for routing stability
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import hashlib
from collections import defaultdict, deque
import yaml

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertState(Enum):
    """Alert lifecycle states"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"

@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    description: str
    condition: str  # Metric condition expression
    severity: AlertSeverity
    threshold: float
    duration_seconds: int = 60  # How long condition must be true
    auto_revert: bool = False
    channels: List[str] = field(default_factory=list)
    runbook_url: Optional[str] = None
    
    # Mediation-aware alerting
    ambiguity_conditional: bool = False  # Only alert in certain A* ranges
    min_ambiguity: float = 0.0
    max_ambiguity: float = 1.0

@dataclass
class Alert:
    """Active alert instance"""
    rule_name: str
    message: str
    severity: AlertSeverity
    first_seen: float
    last_seen: float
    state: AlertState
    metric_value: float
    threshold: float
    count: int = 1
    correlation_id: Optional[str] = None
    auto_revert_triggered: bool = False
    
    def get_duration(self) -> float:
        """Get alert duration in seconds"""
        return self.last_seen - self.first_seen

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    name: str
    type: str  # slack, pagerduty, email, webhook
    config: Dict[str, Any]
    enabled: bool = True

class AlertManager:
    """
    Comprehensive alert manager for MoE routing production
    
    Monitors key routing metrics and triggers alerts based on configurable rules.
    Supports auto-revert for critical failures and multi-channel notifications.
    """
    
    def __init__(self, 
                 config_path: str,
                 api_base_url: str = "http://localhost:8000",
                 check_interval: int = 30):
        
        self.config_path = Path(config_path)
        self.api_base_url = api_base_url
        self.check_interval = check_interval
        
        # Load configuration
        self.reload_config()
        
        # Alert state
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.metrics_history: deque = deque(maxlen=100)
        self.auto_revert_count = 0
        
        # Notification channels
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self._init_notification_channels()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info("AlertManager initialized with comprehensive monitoring")
    
    def reload_config(self):
        """Reload alert configuration from file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Alert config not found: {self.config_path}")
        
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Parse alert rules
        self.alert_rules = {}
        for rule_config in self.config.get('alert_rules', []):
            rule = AlertRule(
                name=rule_config['name'],
                description=rule_config['description'],
                condition=rule_config['condition'],
                severity=AlertSeverity(rule_config['severity']),
                threshold=rule_config['threshold'],
                duration_seconds=rule_config.get('duration_seconds', 60),
                auto_revert=rule_config.get('auto_revert', False),
                channels=rule_config.get('channels', ['default']),
                runbook_url=rule_config.get('runbook_url'),
                ambiguity_conditional=rule_config.get('ambiguity_conditional', False),
                min_ambiguity=rule_config.get('min_ambiguity', 0.0),
                max_ambiguity=rule_config.get('max_ambiguity', 1.0)
            )
            self.alert_rules[rule.name] = rule
        
        logger.info(f"Loaded {len(self.alert_rules)} alert rules")
    
    def _init_notification_channels(self):
        """Initialize notification channels"""
        channels_config = self.config.get('notification_channels', [])
        
        for channel_config in channels_config:
            channel = NotificationChannel(
                name=channel_config['name'],
                type=channel_config['type'],
                config=channel_config.get('config', {}),
                enabled=channel_config.get('enabled', True)
            )
            self.notification_channels[channel.name] = channel
        
        logger.info(f"Initialized {len(self.notification_channels)} notification channels")
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("🔍 Alert monitoring started")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("⏹️  Alert monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info(f"Starting monitoring loop (interval: {self.check_interval}s)")
        
        while self.running:
            try:
                # Collect current metrics
                metrics = await self._collect_metrics()
                if metrics:
                    self.metrics_history.append({
                        'timestamp': time.time(),
                        'metrics': metrics
                    })
                    
                    # Evaluate all alert rules
                    await self._evaluate_alert_rules(metrics)
                    
                    # Clean up resolved alerts
                    self._cleanup_resolved_alerts()
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _collect_metrics(self) -> Optional[Dict[str, float]]:
        """Collect metrics from the API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/metrics", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._extract_alerting_metrics(data)
                    else:
                        logger.warning(f"Metrics collection failed: HTTP {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return None
    
    def _extract_alerting_metrics(self, api_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics for alerting"""
        metrics = {}
        
        # Basic performance metrics
        metrics['avg_gating_sensitivity'] = api_data.get('avg_gating_sensitivity', 0.0)
        metrics['avg_winner_flip_rate'] = api_data.get('avg_winner_flip_rate', 0.0)
        metrics['avg_latency_ms'] = api_data.get('avg_latency_ms', 0.0)
        metrics['error_rate'] = api_data.get('error_rate', 0.0)
        metrics['total_requests'] = api_data.get('total_requests', 0)
        
        # V3 Controller metrics
        if 'pi_controller' in api_data:
            pi_data = api_data['pi_controller']
            metrics['pi_activation_rate'] = pi_data.get('activation_rate', 0.0)
            metrics['pi_integral_error'] = abs(pi_data.get('integral_error', 0.0))
            metrics['pi_last_error'] = abs(pi_data.get('last_error', 0.0))
        
        if 'spike_guard' in api_data:
            sg_data = api_data['spike_guard']
            metrics['spike_guard_activation_rate'] = sg_data.get('activation_rate', 0.0)
            metrics['spike_guard_active'] = 1.0 if sg_data.get('hold_active', False) else 0.0
        
        if 'safety' in api_data:
            safety_data = api_data['safety']
            metrics['G_p99'] = safety_data.get('G_p99_current', 0.0)
            metrics['G_mean'] = safety_data.get('G_mean', 0.0)
            metrics['auto_revert_count'] = safety_data.get('auto_revert_count', 0.0)
            metrics['auto_revert_active'] = 1.0 if safety_data.get('auto_revert_active', False) else 0.0
        
        # Computed metrics
        if len(self.metrics_history) >= 2:
            prev_metrics = self.metrics_history[-2]['metrics']
            
            # Rate of change metrics
            prev_G = prev_metrics.get('avg_gating_sensitivity', 0.0)
            current_G = metrics['avg_gating_sensitivity']
            metrics['G_rate_of_change'] = abs(current_G - prev_G)
            
            # Latency trend
            prev_latency = prev_metrics.get('avg_latency_ms', 0.0)
            current_latency = metrics['avg_latency_ms']
            metrics['latency_trend'] = (current_latency - prev_latency) / max(prev_latency, 1.0)
        
        return metrics
    
    async def _evaluate_alert_rules(self, metrics: Dict[str, float]):
        """Evaluate all alert rules against current metrics"""
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Evaluate rule condition
                triggered = self._evaluate_rule_condition(rule, metrics)
                
                if triggered:
                    await self._handle_alert_triggered(rule, metrics, current_time)
                else:
                    self._handle_alert_resolved(rule_name, current_time)
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    def _evaluate_rule_condition(self, rule: AlertRule, metrics: Dict[str, float]) -> bool:
        """Evaluate if a rule condition is met"""
        # Get metric value for condition
        metric_value = self._extract_metric_from_condition(rule.condition, metrics)
        if metric_value is None:
            return False
        
        # Check ambiguity constraints
        if rule.ambiguity_conditional:
            current_ambiguity = self._get_current_ambiguity()
            if not (rule.min_ambiguity <= current_ambiguity <= rule.max_ambiguity):
                return False
        
        # Evaluate threshold condition
        return self._check_threshold_condition(rule.condition, metric_value, rule.threshold)
    
    def _extract_metric_from_condition(self, condition: str, metrics: Dict[str, float]) -> Optional[float]:
        """Extract metric value from condition string"""
        # Simplified condition parser
        # In production, use proper expression parser
        
        condition_lower = condition.lower()
        
        if 'avg_gating_sensitivity' in condition_lower or 'G' in condition:
            return metrics.get('avg_gating_sensitivity', 0.0)
        elif 'winner_flip_rate' in condition_lower:
            return metrics.get('avg_winner_flip_rate', 0.0)
        elif 'latency' in condition_lower:
            return metrics.get('avg_latency_ms', 0.0)
        elif 'error_rate' in condition_lower:
            return metrics.get('error_rate', 0.0)
        elif 'G_p99' in condition:
            return metrics.get('G_p99', 0.0)
        elif 'pi_integral_error' in condition_lower:
            return metrics.get('pi_integral_error', 0.0)
        elif 'spike_guard_activation' in condition_lower:
            return metrics.get('spike_guard_activation_rate', 0.0)
        elif 'auto_revert_count' in condition_lower:
            return metrics.get('auto_revert_count', 0.0)
        
        return None
    
    def _check_threshold_condition(self, condition: str, value: float, threshold: float) -> bool:
        """Check if threshold condition is met"""
        condition_lower = condition.lower()
        
        if ' > ' in condition_lower or 'greater than' in condition_lower:
            return value > threshold
        elif ' < ' in condition_lower or 'less than' in condition_lower:
            return value < threshold
        elif ' >= ' in condition_lower:
            return value >= threshold
        elif ' <= ' in condition_lower:
            return value <= threshold
        elif ' == ' in condition_lower or 'equals' in condition_lower:
            return abs(value - threshold) < 0.001
        else:
            # Default to greater than
            return value > threshold
    
    def _get_current_ambiguity(self) -> float:
        """Get current average ambiguity score"""
        # This would normally come from recent request metrics
        # For now, return a reasonable default
        return 0.5
    
    async def _handle_alert_triggered(self, rule: AlertRule, metrics: Dict[str, float], current_time: float):
        """Handle when an alert rule is triggered"""
        rule_name = rule.name
        metric_value = self._extract_metric_from_condition(rule.condition, metrics)
        
        if rule_name in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[rule_name]
            alert.last_seen = current_time
            alert.count += 1
            alert.metric_value = metric_value or 0.0
            
            # Check if duration threshold is met for actions
            if alert.get_duration() >= rule.duration_seconds:
                if rule.auto_revert and not alert.auto_revert_triggered:
                    await self._trigger_auto_revert(alert, rule)
                
                # Send escalated notification if needed
                await self._maybe_escalate_alert(alert, rule)
        else:
            # Create new alert
            alert = Alert(
                rule_name=rule_name,
                message=f"{rule.description}: {rule.condition}",
                severity=rule.severity,
                first_seen=current_time,
                last_seen=current_time,
                state=AlertState.ACTIVE,
                metric_value=metric_value or 0.0,
                threshold=rule.threshold
            )
            
            self.active_alerts[rule_name] = alert
            
            # Send initial notification
            await self._send_alert_notification(alert, rule, is_new=True)
            
            logger.warning(f"🚨 Alert triggered: {alert.message}")
    
    def _handle_alert_resolved(self, rule_name: str, current_time: float):
        """Handle when an alert condition is no longer met"""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            
            # Move to resolved state
            alert.state = AlertState.RESOLVED
            alert.last_seen = current_time
            
            # Send resolution notification
            rule = self.alert_rules[rule_name]
            asyncio.create_task(self._send_resolution_notification(alert, rule))
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[rule_name]
            
            logger.info(f"✅ Alert resolved: {alert.message}")
    
    def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts"""
        current_time = time.time()
        cleanup_threshold = 300  # 5 minutes
        
        to_remove = []
        for rule_name, alert in self.active_alerts.items():
            if alert.state == AlertState.RESOLVED:
                if current_time - alert.last_seen > cleanup_threshold:
                    to_remove.append(rule_name)
        
        for rule_name in to_remove:
            del self.active_alerts[rule_name]
    
    async def _trigger_auto_revert(self, alert: Alert, rule: AlertRule):
        """Trigger auto-revert for critical alert"""
        logger.critical(f"🛑 Auto-revert triggered by alert: {alert.message}")
        
        alert.auto_revert_triggered = True
        self.auto_revert_count += 1
        
        try:
            # Call emergency revert API
            async with aiohttp.ClientSession() as session:
                revert_payload = {
                    'reason': f'Auto-revert triggered by alert: {rule.name}',
                    'alert_rule': rule.name,
                    'metric_value': alert.metric_value,
                    'threshold': alert.threshold
                }
                
                async with session.post(
                    f"{self.api_base_url}/admin/emergency-revert",
                    json=revert_payload,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        logger.info("✅ Auto-revert executed successfully")
                        
                        # Send critical notification
                        await self._send_auto_revert_notification(alert, rule)
                    else:
                        logger.error(f"❌ Auto-revert failed: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"💥 Auto-revert failed with exception: {e}")
    
    async def _maybe_escalate_alert(self, alert: Alert, rule: AlertRule):
        """Escalate alert if it has been active too long"""
        escalation_threshold = rule.duration_seconds * 3  # 3x the rule duration
        
        if (alert.get_duration() >= escalation_threshold and 
            alert.state != AlertState.ESCALATED):
            
            alert.state = AlertState.ESCALATED
            await self._send_escalation_notification(alert, rule)
            
            logger.warning(f"📢 Alert escalated: {alert.message}")
    
    async def _send_alert_notification(self, alert: Alert, rule: AlertRule, is_new: bool = False):
        """Send alert notification to configured channels"""
        message = self._format_alert_message(alert, rule, is_new)
        
        for channel_name in rule.channels:
            if channel_name in self.notification_channels:
                channel = self.notification_channels[channel_name]
                if channel.enabled:
                    try:
                        await self._send_to_channel(channel, message, alert.severity)
                    except Exception as e:
                        logger.error(f"Failed to send to channel {channel_name}: {e}")
    
    async def _send_resolution_notification(self, alert: Alert, rule: AlertRule):
        """Send alert resolution notification"""
        message = f"✅ RESOLVED: {alert.message}\nDuration: {alert.get_duration():.0f}s"
        
        for channel_name in rule.channels:
            if channel_name in self.notification_channels:
                channel = self.notification_channels[channel_name]
                if channel.enabled:
                    try:
                        await self._send_to_channel(channel, message, AlertSeverity.INFO)
                    except Exception as e:
                        logger.error(f"Failed to send resolution to channel {channel_name}: {e}")
    
    async def _send_auto_revert_notification(self, alert: Alert, rule: AlertRule):
        """Send critical auto-revert notification"""
        message = f"🛑 CRITICAL: Auto-revert triggered!\n{alert.message}\nRevert count: {self.auto_revert_count}"
        
        # Send to all critical channels
        for channel in self.notification_channels.values():
            if channel.enabled and channel.type in ['pagerduty', 'slack']:
                try:
                    await self._send_to_channel(channel, message, AlertSeverity.CRITICAL)
                except Exception as e:
                    logger.error(f"Failed to send auto-revert notification: {e}")
    
    async def _send_escalation_notification(self, alert: Alert, rule: AlertRule):
        """Send alert escalation notification"""
        message = f"📢 ESCALATED: {alert.message}\nDuration: {alert.get_duration():.0f}s\nCount: {alert.count}"
        
        # Escalate to higher-priority channels
        escalation_channels = ['pagerduty', 'oncall']
        
        for channel in self.notification_channels.values():
            if channel.enabled and channel.type in escalation_channels:
                try:
                    await self._send_to_channel(channel, message, AlertSeverity.CRITICAL)
                except Exception as e:
                    logger.error(f"Failed to send escalation notification: {e}")
    
    def _format_alert_message(self, alert: Alert, rule: AlertRule, is_new: bool) -> str:
        """Format alert message for notification"""
        status = "🚨 NEW" if is_new else "⚠️  ONGOING"
        
        message = f"{status} [{alert.severity.value.upper()}] {alert.message}\n"
        message += f"Value: {alert.metric_value:.3f} (threshold: {alert.threshold:.3f})\n"
        message += f"Duration: {alert.get_duration():.0f}s\n"
        
        if rule.runbook_url:
            message += f"Runbook: {rule.runbook_url}\n"
        
        if rule.auto_revert:
            message += "⚡ Auto-revert enabled\n"
        
        return message
    
    async def _send_to_channel(self, channel: NotificationChannel, message: str, severity: AlertSeverity):
        """Send message to specific notification channel"""
        if channel.type == 'slack':
            await self._send_slack(channel, message, severity)
        elif channel.type == 'webhook':
            await self._send_webhook(channel, message, severity)
        elif channel.type == 'email':
            await self._send_email(channel, message, severity)
        elif channel.type == 'pagerduty':
            await self._send_pagerduty(channel, message, severity)
        else:
            logger.warning(f"Unknown channel type: {channel.type}")
    
    async def _send_slack(self, channel: NotificationChannel, message: str, severity: AlertSeverity):
        """Send Slack notification"""
        webhook_url = channel.config.get('webhook_url')
        if not webhook_url:
            logger.error("Slack webhook URL not configured")
            return
        
        color_map = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning", 
            AlertSeverity.CRITICAL: "danger"
        }
        
        payload = {
            "text": "MoE Router Alert",
            "attachments": [{
                "color": color_map.get(severity, "warning"),
                "text": message,
                "footer": "MoE Router Alert Manager",
                "ts": int(time.time())
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Slack notification failed: {response.status}")
    
    async def _send_webhook(self, channel: NotificationChannel, message: str, severity: AlertSeverity):
        """Send generic webhook notification"""
        webhook_url = channel.config.get('url')
        if not webhook_url:
            logger.error("Webhook URL not configured")
            return
        
        payload = {
            "message": message,
            "severity": severity.value,
            "timestamp": time.time(),
            "source": "moe-router-alerts"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status not in [200, 201, 202]:
                    logger.error(f"Webhook notification failed: {response.status}")
    
    async def _send_email(self, channel: NotificationChannel, message: str, severity: AlertSeverity):
        """Send email notification (placeholder)"""
        # In production, implement SMTP email sending
        logger.info(f"Email notification: {message[:100]}...")
    
    async def _send_pagerduty(self, channel: NotificationChannel, message: str, severity: AlertSeverity):
        """Send PagerDuty notification (placeholder)"""
        # In production, implement PagerDuty Events API
        logger.info(f"PagerDuty notification: {message[:100]}...")
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert system status"""
        return {
            'active_alerts': len(self.active_alerts),
            'total_rules': len(self.alert_rules),
            'auto_revert_count': self.auto_revert_count,
            'monitoring_active': self.running,
            'alert_summary': {
                rule_name: {
                    'severity': alert.severity.value,
                    'duration': alert.get_duration(),
                    'count': alert.count,
                    'auto_revert_triggered': alert.auto_revert_triggered
                }
                for rule_name, alert in self.active_alerts.items()
            }
        }

async def main():
    """Demo alert manager"""
    print("🚨 MoE Router Alert Manager Demo")
    
    # Create sample alert config
    config = {
        'alert_rules': [
            {
                'name': 'high_gating_sensitivity',
                'description': 'Gating sensitivity too high',
                'condition': 'avg_gating_sensitivity > threshold',
                'severity': 'warning',
                'threshold': 2.0,
                'duration_seconds': 120,
                'channels': ['default']
            },
            {
                'name': 'G_p99_critical',
                'description': 'G_p99 exceeds critical threshold',
                'condition': 'G_p99 > threshold',
                'severity': 'critical',
                'threshold': 5.0,
                'duration_seconds': 60,
                'auto_revert': True,
                'channels': ['slack', 'pagerduty']
            }
        ],
        'notification_channels': [
            {
                'name': 'default',
                'type': 'webhook',
                'config': {'url': 'https://example.com/webhook'}
            },
            {
                'name': 'slack',
                'type': 'slack', 
                'config': {'webhook_url': 'https://hooks.slack.com/...'}
            }
        ]
    }
    
    # Save config
    config_path = Path("alert_config_demo.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Initialize alert manager
        alert_manager = AlertManager(str(config_path))
        
        # Start monitoring
        await alert_manager.start_monitoring()
        
        # Run for a short demo period
        await asyncio.sleep(10)
        
        # Check status
        status = alert_manager.get_alert_status()
        print(f"Alert Status: {json.dumps(status, indent=2)}")
        
        # Stop monitoring
        await alert_manager.stop_monitoring()
        
    finally:
        # Cleanup
        if config_path.exists():
            config_path.unlink()

if __name__ == "__main__":
    asyncio.run(main())