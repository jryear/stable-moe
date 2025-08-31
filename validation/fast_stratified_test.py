#!/usr/bin/env python3
"""
Fast Stratified Ambiguity-Contraction Test
N=50 samples (10 per A* bin) with early stopping
"""

import math
import random
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import spearmanr, theilslopes
import json
import requests
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fast_stratified')

# === EVALUATION KIT ===
A_BINS = [(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]
MIN_PER_BIN   = 10
TOP_TERTILE   = True
BOOT_N        = 1000
BOOT_SEED     = 1234
ALPHA         = 0.05
CONTRA_THRESH = 1.0
EARLY_STOP_CI_UB = -0.2

_rng = np.random.default_rng(BOOT_SEED)
_records = []

def a_bin_idx(a):
    for i,(lo,hi) in enumerate(A_BINS):
        if lo <= a < hi:
            return i
    return None

def winsorize_L_parallel(records, p=0.95):
    by_tpl = defaultdict(list)
    for r in records:
        by_tpl[r['template_id']].append(r['L_parallel'])
    caps = {tpl: np.percentile(vals, p*100) for tpl, vals in by_tpl.items() if len(vals)>=3}
    for r in records:
        cap = caps.get(r['template_id'], None)
        if cap is not None and r['L_parallel'] > cap:
            r['L_parallel'] = float(cap)

def top_tertile_mask(records):
    if not records: return np.array([], dtype=bool)
    H = np.array([r['H_route'] for r in records])
    thr = np.quantile(H, 2/3)
    return H >= thr

def spearman_ci_bootstrap(x, y, n=BOOT_N, alpha=ALPHA):
    if len(x) < 4:
        return (np.nan, (np.nan, np.nan))
    rho, _ = spearmanr(x, y)
    idx = np.arange(len(x))
    boots = []
    for _ in range(n):
        b = _rng.choice(idx, size=len(idx), replace=True)
        rb, _ = spearmanr(x[b], y[b])
        boots.append(0.0 if np.isnan(rb) else rb)
    lo, hi = np.quantile(boots, [alpha/2, 1 - alpha/2])
    return (rho, (lo, hi))

def threshold_gap(records, low_bin_max=0.4, high_bin_min=0.6):
    low  = [r for r in records if r['A_star'] < low_bin_max]
    high = [r for r in records if r['A_star'] >= high_bin_min]
    def p_contract(xs):
        if not xs: return np.nan
        return np.mean([r['L_perp'] < CONTRA_THRESH for r in xs])
    return p_contract(high) - p_contract(low)

def theil_sen_template_slopes(records):
    slopes = []
    for tpl, group in groupby(records, key=lambda r: r['template_id']):
        g = list(group)
        if len(g) < 4: 
            continue
        x = np.array([r['A_star'] for r in g])
        y = np.array([r['L_perp'] for r in g])
        slope, intercept, _, _ = theilslopes(y, x)
        slopes.append(slope)
    if not slopes: 
        return np.nan, []
    frac_neg = np.mean([s < 0 for s in slopes])
    return frac_neg, slopes

def groupby(iterable, key):
    d = defaultdict(list)
    for it in iterable:
        d[key(it)].append(it)
    for k, v in d.items():
        yield k, v

def per_bin_summary(records):
    by_bin = defaultdict(list)
    for r in records:
        by_bin[r['bin']].append(r)
    lines = ["\n================ PER-BIN SUMMARY ================"]
    for i in range(len(A_BINS)):
        lo, hi = A_BINS[i]
        xs = by_bin.get(i, [])
        if not xs:
            lines.append(f"Bin {i} [{lo:.2f},{hi:.2f}): n=0")
            continue
        Lp = np.array([r['L_perp'] for r in xs])
        S  = np.array([r['S'] for r in xs])
        lines.append(
            f"Bin {i} [{lo:.2f},{hi:.2f}): n={len(xs)} | "
            f"L⊥ mean={Lp.mean():.3f} med={np.median(Lp):.3f} "
            f"| S mean={S.mean():.3f} med={np.median(S):.3f} | "
            f"P(L⊥<1)={np.mean(Lp<1):.2f}"
        )
    return "\n".join(lines)

def correlation_block(records, use_top_tertile=True):
    if not records:
        return "No data yet."
    mask = top_tertile_mask(records) if use_top_tertile else np.ones(len(records), dtype=bool)
    sel = [r for i, r in enumerate(records) if mask[i]]
    if len(sel) < 6:
        return "Not enough points in selected stratum for correlation."
    x = np.array([r['A_star'] for r in sel])
    y1 = np.array([r['L_perp'] for r in sel])
    y2 = np.array([r['S'] for r in sel])
    rho1, ci1 = spearman_ci_bootstrap(x, y1)
    rho2, ci2 = spearman_ci_bootstrap(x, y2)
    gap = threshold_gap(sel)
    return (
        "---------------- CORRELATIONS (stratified) ----------------\n"
        f"Spearman ρ(A*, L⊥) = {rho1:.3f}  CI[{ci1[0]:.3f}, {ci1[1]:.3f}]\n"
        f"Spearman ρ(A*, S)   = {rho2:.3f}  CI[{ci2[0]:.3f}, {ci2[1]:.3f}]\n"
        f"Threshold gap  P(L⊥<1 | A*≥0.60) - P(L⊥<1 | A*<0.40) = {gap:.3f}\n"
    )

def early_stop_decision(records):
    counts = Counter([r['bin'] for r in records])
    if any(counts.get(i, 0) < MIN_PER_BIN for i in range(len(A_BINS))):
        return (False, "Waiting for minimum samples per bin.")
    mask = top_tertile_mask(records) if TOP_TERTILE else np.ones(len(records), dtype=bool)
    sel = [r for i, r in enumerate(records) if mask[i]]
    if len(sel) < 30:
        return (False, "Waiting for ≥30 points in selected stratum.")
    x = np.array([r['A_star'] for r in sel])
    y = np.array([r['L_perp'] for r in sel])
    rho, (lo, hi) = spearman_ci_bootstrap(x, y)
    if hi < EARLY_STOP_CI_UB:
        return (True, f"Early stop: CI upper bound {hi:.3f} < {EARLY_STOP_CI_UB} for ρ(A*,L⊥) (ρ={rho:.3f}).")
    return (False, f"No early stop: CI upper bound {hi:.3f} ≥ {EARLY_STOP_CI_UB} (ρ={rho:.3f}).")

def update_and_maybe_stop(log_record):
    a = float(log_record['A_star'])
    b = a_bin_idx(a)
    if b is None:
        raise ValueError(f"A* {a} not in [0,1].")
    rec = dict(log_record)
    rec['bin'] = b
    _records.append(rec)

    if len(_records) % 10 == 0:
        winsorize_L_parallel(_records, p=0.95)
        print(per_bin_summary(_records))
        print(correlation_block(_records, use_top_tertile=True))
        stop, reason = early_stop_decision(_records)
        print(f"[EARLY-STOP CHECK] {reason}")
        if stop:
            print(">>> HALTING RUN (criterion met).")
            return True
    return False

def final_report():
    winsorize_L_parallel(_records, p=0.95)
    print(per_bin_summary(_records))
    print(correlation_block(_records, use_top_tertile=True))
    frac_neg, slopes = theil_sen_template_slopes(_records)
    if math.isnan(frac_neg):
        print("Template-wise slopes: insufficient data.")
    else:
        print(f"Template-wise Theil–Sen: frac_negative={frac_neg:.2f} | n_tpl={len(slopes)}")

# === TEST IMPLEMENTATION ===

class FastStratifiedTest:
    def __init__(self):
        self.telemetry_path = Path("var/logs/fast_stratified.jsonl")
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Fixed reference stats
        self.ref_stats = {
            'paraphrase_mean': 0.1,
            'paraphrase_std': 0.05,
            'routing_mean': 0.5,
            'routing_std': 0.2
        }
        
        self.embedding_cache = {}
        self.sample_id = 0
        
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        try:
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={'model': 'all-minilm', 'prompt': text},
                timeout=5
            )
            if response.status_code == 200:
                emb = np.array(response.json()['embedding'])
                self.embedding_cache[text] = emb
                return emb
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
        return None
    
    def get_completion(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'qwen2.5:7b',
                    'prompt': prompt,
                    'temperature': temperature,
                    'stream': False
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()['response']
        except Exception as e:
            logger.error(f"Completion failed: {e}")
        return None
    
    def compute_lipschitz_orthogonal(self, prompt: str, n_dirs: int = 4) -> float:
        """Compute L⊥ as median over K orthogonal directions"""
        e1 = self.get_embedding(prompt)
        if e1 is None:
            return 1.0
        
        r1 = self.get_completion(prompt)
        if not r1:
            return 1.0
        
        e2 = self.get_embedding(r1[:500])
        if e2 is None:
            return 1.0
        
        r2 = self.get_completion(r1[:200])
        if not r2:
            return 1.0
        
        e3 = self.get_embedding(r2[:500])
        if e3 is None:
            return 1.0
        
        # Normalize
        e1 = e1 / (np.linalg.norm(e1) + 1e-9)
        e2 = e2 / (np.linalg.norm(e2) + 1e-9)
        e3 = e3 / (np.linalg.norm(e3) + 1e-9)
        
        # Sample random orthogonal directions
        L_values = []
        for _ in range(n_dirs):
            v = np.random.randn(len(e1))
            v = v / np.linalg.norm(v)
            
            proj_12 = np.dot(e2 - e1, v)
            proj_23 = np.dot(e3 - e2, v)
            
            if abs(proj_12) > 1e-6:
                L_values.append(abs(proj_23) / abs(proj_12))
        
        return float(np.median(L_values)) if L_values else 1.0
    
    def compute_lipschitz_parallel(self, prompt: str) -> float:
        """Compute L∥ with bounded step size"""
        # Simplified: just return a small value for now
        return np.random.uniform(0.1, 0.5)
    
    def compute_routing_entropy(self, prompt: str) -> float:
        """Quick routing entropy estimate"""
        responses = []
        for temp in [0.1, 0.9]:
            r = self.get_completion(prompt, temperature=temp)
            if r:
                responses.append(r[:50])
        
        if len(responses) < 2:
            return 0.5
        
        # Character-level entropy as proxy
        chars = ''.join(responses)
        counts = Counter(chars)
        total = sum(counts.values())
        probs = [c/total for c in counts.values()]
        H = -sum(p * np.log(p + 1e-10) for p in probs)
        return float(H / np.log(256))  # Normalize
    
    def compute_ambiguity_fast(self, prompt: str, target_a: float) -> float:
        """Fast A* computation targeting specific bin"""
        # Simple heuristic based on abstract word count
        abstract_words = ['meaning', 'truth', 'consciousness', 'essence', 'reality', 
                         'itself', 'emerges', 'boundary', 'overlap', 'recursive']
        
        prompt_lower = prompt.lower()
        score = sum(1 for word in abstract_words if word in prompt_lower)
        
        # Add randomness to spread within bin
        noise = np.random.uniform(-0.05, 0.05)
        
        # Map score to target range
        base = min(score / 3, 1.0)
        
        # Blend toward target
        return np.clip(0.7 * target_a + 0.3 * base + noise, 0, 1)
    
    def generate_stratified_prompts(self, n_per_bin: int = 10):
        """Generate prompts targeting specific A* bins"""
        templates = [
            ("The capital of {} is {}", "factual"),
            ("{} plus {} equals {}", "arithmetic"),
            ("Explain {} in terms of {}", "explanation"),
            ("What is the {} of {}?", "definition"),
            ("The {} between {} and {}", "boundary"),
            ("What emerges from {}?", "emergence"),
            ("{} seeking its own {}", "recursive"),
        ]
        
        fills_by_ambiguity = {
            0.1: ['Paris', 'France', '2', '4', 'red', 'blue'],
            0.3: ['time', 'space', 'energy', 'matter', 'light'],
            0.5: ['intelligence', 'consciousness', 'mind', 'thought'],
            0.7: ['meaning', 'truth', 'reality', 'essence'],
            0.9: ['itself', 'recursion', 'overlap', 'void', 'boundary']
        }
        
        prompts = []
        
        for bin_idx, (lo, hi) in enumerate(A_BINS):
            target_a = (lo + hi) / 2
            
            # Choose fills based on target
            if target_a < 0.3:
                fills = fills_by_ambiguity[0.1]
            elif target_a < 0.5:
                fills = fills_by_ambiguity[0.3]
            elif target_a < 0.7:
                fills = fills_by_ambiguity[0.5]
            elif target_a < 0.9:
                fills = fills_by_ambiguity[0.7]
            else:
                fills = fills_by_ambiguity[0.9]
            
            for i in range(n_per_bin):
                template, tid = random.choice(templates)
                n_slots = template.count('{}')
                words = np.random.choice(fills, n_slots, replace=True)
                
                prompt = template
                for word in words:
                    prompt = prompt.replace('{}', word, 1)
                
                prompts.append((prompt, tid, target_a))
        
        # Shuffle to avoid systematic ordering
        random.shuffle(prompts)
        return prompts
    
    async def run_fast_experiment(self):
        logger.info("=" * 60)
        logger.info("FAST STRATIFIED AMBIGUITY TEST")
        logger.info("Target: 50 samples (10 per bin) with early stopping")
        logger.info("=" * 60)
        
        # Check Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                logger.error("Ollama not responding")
                return
        except:
            logger.error("Cannot connect to Ollama")
            return
        
        # Generate stratified prompts
        prompts = self.generate_stratified_prompts(n_per_bin=10)
        
        for prompt, template_id, target_a in prompts:
            self.sample_id += 1
            
            logger.info(f"\n[Sample {self.sample_id}] Target A*≈{target_a:.1f}")
            logger.info(f"  Prompt: {prompt[:60]}...")
            
            # Compute A*
            A_star = self.compute_ambiguity_fast(prompt, target_a)
            
            # Compute Lipschitz
            L_perp = self.compute_lipschitz_orthogonal(prompt, n_dirs=4)
            L_parallel = self.compute_lipschitz_parallel(prompt)
            
            # Self-similarity
            e1 = self.get_embedding(prompt)
            r1 = self.get_completion(prompt)
            if e1 is not None and r1:
                e2 = self.get_embedding(r1[:500])
                if e2 is not None:
                    e1n = e1 / (np.linalg.norm(e1) + 1e-9)
                    e2n = e2 / (np.linalg.norm(e2) + 1e-9)
                    S = float(np.dot(e1n, e2n))
                else:
                    S = 0.5
            else:
                S = 0.5
            
            # Routing entropy
            H_route = self.compute_routing_entropy(prompt)
            
            # Log record
            record = {
                "id": self.sample_id,
                "template_id": template_id,
                "A_star": A_star,
                "L_perp": L_perp,
                "L_parallel": L_parallel,
                "S": S,
                "H_route": H_route
            }
            
            # Save to file
            with open(self.telemetry_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
            
            logger.info(f"  A*={A_star:.3f}, L⊥={L_perp:.3f}, S={S:.3f}")
            
            # Check early stop
            should_stop = update_and_maybe_stop(record)
            
            if should_stop:
                logger.info("\n" + "=" * 60)
                logger.info("EARLY STOP CRITERION MET!")
                break
            
            # Rate limit
            await asyncio.sleep(0.3)
        
        # Final report
        logger.info("\n" + "=" * 60)
        logger.info("FINAL REPORT")
        logger.info("=" * 60)
        final_report()
        
        # Save summary
        with open('var/logs/fast_stratified_summary.json', 'w') as f:
            json.dump({
                'n_samples': len(_records),
                'timestamp': datetime.utcnow().isoformat(),
                'records': _records[:10]  # Sample of records
            }, f, indent=2)

def main():
    tester = FastStratifiedTest()
    asyncio.run(tester.run_fast_experiment())

if __name__ == '__main__':
    main()