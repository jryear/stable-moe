# MoE Router V3 Production On-Call Runbook

## 🚨 Emergency Response Guide

> **First Rule**: If unsure, revert to baseline routing immediately using the emergency commands below.

### Emergency Stop Commands

```bash
# Immediate revert to baseline routing (SAFE)
curl -X POST http://localhost:8000/admin/emergency-revert \
  -H "Content-Type: application/json" \
  -d '{"reason": "Manual emergency stop"}'

# Check current status
curl http://localhost:8000/health

# Force restart controller (if API responsive)
curl -X POST http://localhost:8000/reset
```

---

## 🎯 Quick Incident Response

### 1. G_p99 Spike (Most Critical)

**Symptoms**: G_p99 > 5.0, routing instability, potential auto-revert

**Immediate Actions**:
```bash
# 1. Check current status
curl http://localhost:8000/metrics | jq '.safety.G_p99_current'

# 2. If G_p99 > 6.0, emergency revert
curl -X POST http://localhost:8000/admin/emergency-revert

# 3. Otherwise, tune down aggressiveness
curl -X POST http://localhost:8000/admin/tune-parameters \
  -d '{"beta_max": 1.4, "lambda_min": 0.35}'  # Less aggressive

# 4. Enable S-matrix freeze for high ambiguity
curl -X POST http://localhost:8000/admin/freeze-s-matrix \
  -d '{"threshold": 0.6, "duration": 600}'
```

**Root Causes**:
- High-ambiguity traffic surge
- PI controller oscillation  
- Spike guard disabled/misconfigured
- Model behavior change

### 2. Win-Rate Drop in Clear Cases

**Symptoms**: Quality degradation for A* < 0.3, user complaints

**Immediate Actions**:
```bash
# 1. Check clarity distribution
curl http://localhost:8000/metrics | jq '.ambiguity_distribution'

# 2. Raise decisiveness for clear cases
curl -X POST http://localhost:8000/admin/tune-parameters \
  -d '{"beta_min": 0.95, "lambda_max": 0.8}'

# 3. Monitor improvement
watch -n 10 'curl -s http://localhost:8000/metrics | jq .avg_clarity_score'
```

**Validation**:
- Win-rate should improve within 10 minutes
- No increase in G_p99
- Latency remains stable

### 3. Jitter/Latency Spike

**Symptoms**: Latency P95 > 200ms, user experience degraded

**Immediate Actions**:
```bash
# 1. Check if spike guard is working
curl http://localhost:8000/metrics | jq '.spike_guard'

# 2. Lower spike threshold temporarily
curl -X POST http://localhost:8000/admin/tune-spike-guard \
  -d '{"threshold": 2.0, "hold_ticks": 7}'

# 3. Cap high-latency experts if needed
curl -X POST http://localhost:8000/admin/expert-controls \
  -d '{"action": "cap_latency", "threshold_ms": 300}'
```

### 4. Auto-Revert Triggered

**Symptoms**: System reverted to baseline, alerts firing

**Response Checklist**:
1. ✅ **DO NOT** immediately re-enable V3 controller
2. ✅ Investigate root cause using logs
3. ✅ Check for external factors (model updates, traffic changes)
4. ✅ Run counterfactual analysis on recent data
5. ✅ Only re-enable after validation

```bash
# Investigation commands
tail -f /var/log/moe-router/app.log | grep "auto_revert"
curl http://localhost:8000/admin/auto-revert-analysis
curl http://localhost:8000/admin/validate-controller
```

---

## 📊 Diagnostic Procedures

### Health Check Command Reference

```bash
# Basic health
curl http://localhost:8000/health

# Detailed metrics
curl http://localhost:8000/metrics | jq

# Controller status
curl http://localhost:8000/metrics | jq '.pi_controller, .spike_guard, .safety'

# Recent alerts
curl http://localhost:8000/alerts/recent

# Mediation status
curl http://localhost:8000/mediation-status
```

### Log Analysis

```bash
# Check for errors
tail -f /var/log/moe-router/app.log | grep ERROR

# Routing decisions
tail -f /var/log/moe-router/app.log | grep ROUTING_EVENT

# Auto-revert events  
grep "auto_revert" /var/log/moe-router/app.log | tail -20

# Controller state changes
grep "pi_controller\|spike_guard" /var/log/moe-router/app.log | tail -20
```

### Performance Analysis

```bash
# Latency percentiles
curl http://localhost:8000/metrics | jq '.latency_percentiles'

# G reduction effectiveness
curl http://localhost:8000/metrics | jq '.improvement_metrics'

# Expert utilization
curl http://localhost:8000/expert-stats
```

---

## ⚙️ Parameter Tuning Guide

### Safe Parameter Ranges

| Parameter | Safe Range | Default | Conservative | Aggressive |
|-----------|------------|---------|--------------|------------|
| `beta_min` | 0.7 - 1.0 | 0.85 | 0.90 | 0.80 |
| `beta_max` | 1.2 - 2.0 | 1.65 | 1.55 | 1.75 |
| `lambda_min` | 0.2 - 0.4 | 0.25 | 0.30 | 0.20 |
| `lambda_max` | 0.6 - 0.9 | 0.75 | 0.70 | 0.80 |
| `G_target` | 0.4 - 0.8 | 0.60 | 0.65 | 0.55 |

### Common Tuning Scenarios

#### High G, Good Win-Rate
**Problem**: G reduction target missed, but quality is fine  
**Action**: More aggressive control
```bash
curl -X POST http://localhost:8000/admin/tune-parameters \
  -d '{"beta_max": 1.75, "lambda_min": 0.20, "G_target": 0.55}'
```

#### Good G, Poor Win-Rate
**Problem**: G target hit, but quality degraded  
**Action**: More conservative control
```bash
curl -X POST http://localhost:8000/admin/tune-parameters \
  -d '{"beta_min": 0.90, "lambda_max": 0.80, "G_target": 0.65}'
```

#### Oscillations/Instability
**Problem**: PI controller oscillating  
**Action**: Lower gains, increase damping
```bash
curl -X POST http://localhost:8000/admin/tune-pi \
  -d '{"k_p": 0.08, "k_i": 0.015, "integral_decay": 0.9}'
```

---

## 🔍 Troubleshooting Common Issues

### Issue: "Mediation Pathway Weak"

**Alert**: `ρ(A*, G) < 0.25`

**Diagnosis**:
```bash
# Check current correlation
curl http://localhost:8000/mediation-status | jq '.summary.key_correlations'

# Check A* range compression
curl http://localhost:8000/metrics | jq '.ambiguity_distribution'
```

**Solutions**:
1. **Narrow A* range**: Wait for more diverse traffic or run validation with synthetic data
2. **Model behavior change**: Re-run mediation validation
3. **Measurement issues**: Check embedding service health

### Issue: "Spike Guard Overactive"

**Alert**: Spike guard activating > 15% of requests

**Diagnosis**:
```bash
# Check activation rate and threshold
curl http://localhost:8000/metrics | jq '.spike_guard'

# Check gradient distribution
curl http://localhost:8000/admin/gradient-analysis
```

**Solutions**:
```bash
# Raise threshold
curl -X POST http://localhost:8000/admin/tune-spike-guard \
  -d '{"threshold": 3.0}'

# Or adjust detection sensitivity
curl -X POST http://localhost:8000/admin/tune-spike-guard \
  -d '{"trigger_count": 4, "detection_window": 7}'
```

### Issue: "PI Controller Not Activating"

**Symptoms**: PI activation rate < 5% in high-ambiguity traffic

**Diagnosis**:
```bash
# Check activation threshold
curl http://localhost:8000/metrics | jq '.pi_controller.activation_threshold'

# Check recent ambiguity distribution  
curl http://localhost:8000/admin/ambiguity-analysis
```

**Solutions**:
```bash
# Lower activation threshold
curl -X POST http://localhost:8000/admin/tune-pi \
  -d '{"activation_threshold": 0.6}'

# Or check if ambiguity scoring is working
curl http://localhost:8000/admin/validate-ambiguity-scoring
```

### Issue: "Model Responses Changed"

**Symptoms**: Sudden shift in routing behavior, mediation broken

**Investigation**:
```bash
# Check model version/health
curl http://ollama:11434/api/tags

# Compare recent vs historical routing
curl http://localhost:8000/admin/routing-drift-analysis

# Run fresh mediation validation
curl -X POST http://localhost:8000/validate-mediation
```

---

## 📈 Monitoring & Alerting

### Critical Dashboards

1. **Production Overview**: http://localhost:3000
2. **Mediation Analysis**: http://localhost:3000?view=mediation
3. **Controller Health**: http://localhost:3000?view=controller
4. **Expert Utilization**: http://localhost:3000?view=experts

### Key Metrics to Watch

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| `G_p95` (A* > 0.7) | < 2.0 | > 2.5 | > 3.5 |
| `G_p99` | < 4.0 | > 4.5 | > 5.0 |
| Win-rate (A* < 0.3) | > -0.5% | < -1.0% | < -2.0% |
| Latency P95 | < 150ms | > 200ms | > 300ms |
| `ρ(A*, G)` | > 0.30 | < 0.25 | < 0.15 |
| Auto-revert count | 0 | > 1 | > 3 |

### Alert Response Times

- **Critical**: < 5 minutes
- **Warning**: < 15 minutes  
- **Info**: < 1 hour

---

## 🛠️ Advanced Operations

### Counterfactual Analysis

Run after incidents to validate controller decisions:

```bash
# Generate analysis report
curl -X POST http://localhost:8000/admin/counterfactual-analysis \
  -d '{"time_range": "last_hour", "output_dir": "analysis/incident_123"}'

# Check results
curl http://localhost:8000/admin/counterfactual-results/incident_123
```

### Expert Health Management

```bash
# Check expert specialization
curl http://localhost:8000/expert-stats | jq '.specialization'

# Disable problematic expert temporarily
curl -X POST http://localhost:8000/admin/expert-controls \
  -d '{"expert_id": "expert_2", "action": "disable", "duration": 300}'

# Rebalance expert weights
curl -X POST http://localhost:8000/admin/expert-controls \
  -d '{"action": "rebalance", "strategy": "entropy_based"}'
```

### Data Collection for Analysis

```bash
# Export recent routing decisions
curl http://localhost:8000/admin/export-telemetry \
  -d '{"format": "csv", "hours": 24}' > routing_data.csv

# Export mediation analysis
curl http://localhost:8000/admin/export-mediation \
  -d '{"include_raw_data": true}' > mediation_analysis.json
```

---

## 📞 Escalation Procedures

### Level 1: Self-Service
- Use this runbook
- Apply safe parameter tunings
- Monitor for 15 minutes

### Level 2: Engineering On-Call
**Escalate if**:
- Auto-revert triggered multiple times
- G_p99 > 6.0 sustained
- Win-rate drop > 3% in clear cases
- Multiple critical alerts

**Contact**: engineering-oncall@company.com

### Level 3: Engineering Lead
**Escalate if**:
- System-wide outage
- Data corruption suspected  
- Security incident
- Need for emergency deployment

**Contact**: engineering-lead@company.com

---

## 📋 Incident Response Checklist

### During Incident
- [ ] Emergency stop executed if needed
- [ ] Current status documented
- [ ] Key metrics captured
- [ ] Logs collected
- [ ] Stakeholders notified

### Post-Incident
- [ ] Root cause identified
- [ ] Timeline documented
- [ ] Counterfactual analysis complete
- [ ] System validated before re-enabling
- [ ] Post-mortem scheduled
- [ ] Runbook updated

---

## 🔗 Useful Commands Summary

```bash
# Emergency
curl -X POST http://localhost:8000/admin/emergency-revert

# Status checks  
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:8000/mediation-status

# Parameter tuning
curl -X POST http://localhost:8000/admin/tune-parameters -d '{...}'
curl -X POST http://localhost:8000/admin/tune-pi -d '{...}'
curl -X POST http://localhost:8000/admin/tune-spike-guard -d '{...}'

# Analysis
curl -X POST http://localhost:8000/admin/counterfactual-analysis
curl -X POST http://localhost:8000/validate-mediation
curl http://localhost:8000/admin/routing-drift-analysis

# Logs
tail -f /var/log/moe-router/app.log
grep "ERROR\|CRITICAL" /var/log/moe-router/app.log
```

---

## 📚 Additional Resources

- **Architecture Documentation**: docs/ARCHITECTURE.md
- **API Reference**: docs/API.md  
- **Research Paper**: docs/PAPER.md
- **Configuration Guide**: docs/CONFIGURATION.md
- **Deployment Guide**: docs/DEPLOYMENT.md

**Last Updated**: $(date)  
**Version**: V3 PI Spike Guard  
**Maintained By**: Platform Engineering Team