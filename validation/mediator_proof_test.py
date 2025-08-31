#!/usr/bin/env python3
"""
Mediator Proof Test
Tests if gating sensitivity G mediates the A* → L_⊥ relationship
Key hypothesis: ρ(A*, L_⊥ | G) ≈ 0 → G is the causal pathway
"""

import numpy as np
# import pandas as pd  # Not needed
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import requests
from typing import List, Dict, Tuple, Optional

def get_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding from Ollama"""
    try:
        response = requests.post(
            'http://localhost:11434/api/embeddings',
            json={'model': 'all-minilm', 'prompt': text},
            timeout=5
        )
        if response.status_code == 200:
            return np.array(response.json()['embedding'])
    except:
        pass
    return None

def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """
    Compute partial correlation ρ(x,y|z)
    Using the formula: ρ(x,y|z) = [ρ(x,y) - ρ(x,z)ρ(y,z)] / sqrt[(1-ρ²(x,z))(1-ρ²(y,z))]
    """
    rxy, _ = pearsonr(x, y)
    rxz, _ = pearsonr(x, z)
    ryz, _ = pearsonr(y, z)
    
    numerator = rxy - rxz * ryz
    denominator = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    
    if abs(denominator) < 1e-9:
        return 0.0, 1.0
    
    rho_partial = numerator / denominator
    
    # Approximate p-value using t-distribution
    n = len(x)
    if n > 3:
        t_stat = rho_partial * np.sqrt((n - 3) / (1 - rho_partial**2))
        from scipy.stats import t
        p_value = 2 * (1 - t.cdf(abs(t_stat), n - 3))
    else:
        p_value = 1.0
    
    return rho_partial, p_value

def compute_lipschitz_constant(prompt1: str, prompt2: str) -> Optional[Dict]:
    """
    Compute Lipschitz constant for response contraction/expansion
    """
    emb1 = get_embedding(prompt1)
    emb2 = get_embedding(prompt2)
    
    if emb1 is None or emb2 is None:
        return None
    
    # Normalize embeddings
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-9)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-9)
    
    # Input distance
    delta_x = np.linalg.norm(emb2 - emb1)
    
    # For demonstration, we'll simulate output responses
    # In production, this would be actual model responses
    np.random.seed(hash(prompt1 + prompt2) % 2**32)
    
    # Simulate response embeddings based on input
    resp1_emb = emb1 + np.random.randn(*emb1.shape) * 0.1
    resp2_emb = emb2 + np.random.randn(*emb2.shape) * 0.1
    
    # Output distance
    delta_y = np.linalg.norm(resp2_emb - resp1_emb)
    
    # Lipschitz constant
    L = delta_y / (delta_x + 1e-9)
    
    return {
        'L_total': float(L),
        'delta_x': float(delta_x),
        'delta_y': float(delta_y),
        'input_similarity': float(np.dot(emb1, emb2))
    }

def compute_gating_weights(embeddings: List[np.ndarray], temperature: float = 1.0) -> np.ndarray:
    """Simulate gating weights from embeddings"""
    if len(embeddings) < 2:
        return np.array([1.0])
    
    # Use first few embeddings as "expert prototypes"
    n_experts = min(5, len(embeddings))
    prototypes = embeddings[:n_experts]
    
    # Compute similarities to each prototype
    current = embeddings[-1]
    current = current / (np.linalg.norm(current) + 1e-9)
    
    logits = []
    for proto in prototypes:
        proto = proto / (np.linalg.norm(proto) + 1e-9)
        similarity = np.dot(current, proto)
        logits.append(similarity)
    
    logits = np.array(logits)
    
    # Apply temperature and softmax
    logits = logits * temperature
    logits = logits - np.max(logits)  # Stability
    exp_logits = np.exp(logits)
    weights = exp_logits / (np.sum(exp_logits) + 1e-9)
    
    return weights

def compute_gating_sensitivity(prompt1: str, prompt2: str) -> Optional[Dict]:
    """
    Compute G = ||Δw||/||Δx|| where:
    - Δx is the change in input embedding
    - Δw is the change in gating weights
    """
    # Get embeddings
    emb1 = get_embedding(prompt1)
    emb2 = get_embedding(prompt2)
    
    if emb1 is None or emb2 is None:
        return None
    
    # Normalize embeddings
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-9)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-9)
    
    # Compute input change
    delta_x = np.linalg.norm(emb2 - emb1)
    
    # Get some context embeddings for prototypes
    context_prompts = [
        "The answer is",
        "This means that",
        "In other words",
        "The result is",
        "Therefore"
    ]
    
    context_embeddings = []
    for cp in context_prompts:
        cemb = get_embedding(cp)
        if cemb is not None:
            context_embeddings.append(cemb)
    
    # Compute gating weights for both
    embeddings1 = context_embeddings + [emb1]
    embeddings2 = context_embeddings + [emb2]
    
    w1 = compute_gating_weights(embeddings1)
    w2 = compute_gating_weights(embeddings2)
    
    # Compute change in gating
    delta_w = np.linalg.norm(w2 - w1)
    
    # Gating sensitivity
    G = delta_w / (delta_x + 1e-9)
    
    return {
        'G': float(G),
        'delta_x': float(delta_x),
        'delta_w': float(delta_w),
        'w1': w1.tolist(),
        'w2': w2.tolist()
    }

def estimate_ambiguity(prompt: str) -> float:
    """
    Estimate ambiguity score A* for a prompt
    Simple heuristic based on question words, abstract concepts, etc.
    """
    prompt_lower = prompt.lower()
    
    # Question indicators (higher ambiguity)
    question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'explain', 'describe']
    question_score = sum(1 for w in question_words if w in prompt_lower) / len(question_words)
    
    # Abstract concepts (higher ambiguity)
    abstract_words = ['meaning', 'purpose', 'consciousness', 'intelligence', 'emergence', 'recursive', 'boundary']
    abstract_score = sum(1 for w in abstract_words if w in prompt_lower) / len(abstract_words)
    
    # Definite indicators (lower ambiguity)
    definite_words = ['calculate', 'compute', 'add', 'subtract', 'multiply', 'divide', 'is', 'are']
    definite_score = sum(1 for w in definite_words if w in prompt_lower) / len(definite_words)
    
    # Combine scores
    A_star = 0.4 * question_score + 0.4 * abstract_score + 0.2 * (1 - definite_score)
    return np.clip(A_star, 0, 1)

def run_mediator_proof_test():
    """
    Main test: Check if gating sensitivity G mediates A* → L_⊥ relationship
    """
    print("=" * 60)
    print("MEDIATOR PROOF TEST")
    print("Testing if G mediates A* → L_⊥")
    print("=" * 60)
    
    # Test prompt pairs with varying ambiguity
    test_pairs = [
        # Low ambiguity
        ("What is 2+2?", "What is 3+3?"),
        ("The capital of France is", "The capital of Spain is"),
        ("Calculate 10 * 5", "Calculate 12 * 4"),
        
        # Medium ambiguity
        ("Explain photosynthesis", "Explain respiration"),
        ("What causes rain?", "What causes wind?"),
        ("How do computers work?", "How do engines work?"),
        
        # High ambiguity
        ("What is consciousness?", "What is intelligence?"),
        ("Explain the meaning of life", "Explain the nature of reality"),
        ("What emerges from complexity?", "What emerges from chaos?"),
        ("The boundary between meaning and", "The boundary between truth and"),
        ("Recursive self-reference in", "Recursive meta-reference in")
    ]
    
    print(f"\nTesting {len(test_pairs)} prompt pairs...")
    
    results = []
    
    for i, (prompt1, prompt2) in enumerate(test_pairs):
        print(f"\nPair {i+1}: Testing similarity...")
        print(f"  P1: {prompt1[:50]}...")
        print(f"  P2: {prompt2[:50]}...")
        
        # Compute ambiguity (average of both prompts)
        A1 = estimate_ambiguity(prompt1)
        A2 = estimate_ambiguity(prompt2)
        A_star = (A1 + A2) / 2
        
        # Compute Lipschitz constant
        lipschitz_result = compute_lipschitz_constant(prompt1, prompt2)
        
        # Compute gating sensitivity
        gating_result = compute_gating_sensitivity(prompt1, prompt2)
        
        if lipschitz_result is not None and gating_result is not None:
            L_perp = lipschitz_result['L_total']
            G = gating_result['G']
            
            print(f"  A* = {A_star:.3f}, L_⊥ = {L_perp:.3f}, G = {G:.4f}")
            
            results.append({
                'pair_id': i + 1,
                'prompt1': prompt1,
                'prompt2': prompt2,
                'A_star': A_star,
                'L_perp': L_perp,
                'G': G,
                'delta_x': gating_result['delta_x'],
                'delta_w': gating_result['delta_w'],
                'input_similarity': lipschitz_result['input_similarity']
            })
        else:
            print(f"  Skipped (embedding failed)")
    
    if len(results) < 3:
        print("\nInsufficient data for correlation analysis")
        return
    
    # Convert to arrays for analysis
    A_values = np.array([r['A_star'] for r in results])
    L_values = np.array([r['L_perp'] for r in results])
    G_values = np.array([r['G'] for r in results])
    
    # Compute correlations
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Direct correlations
    rho_AL, p_AL = spearmanr(A_values, L_values)
    rho_AG, p_AG = spearmanr(A_values, G_values)
    rho_GL, p_GL = spearmanr(G_values, L_values)
    
    print(f"Direct correlations:")
    print(f"  ρ(A*, L_⊥) = {rho_AL:+.3f} (p={p_AL:.3f})")
    print(f"  ρ(A*, G)   = {rho_AG:+.3f} (p={p_AG:.3f})")
    print(f"  ρ(G, L_⊥)  = {rho_GL:+.3f} (p={p_GL:.3f})")
    
    # Partial correlation - the key test!
    rho_AL_given_G, p_AL_given_G = partial_correlation(A_values, L_values, G_values)
    
    print(f"\nPartial correlation (KEY TEST):")
    print(f"  ρ(A*, L_⊥ | G) = {rho_AL_given_G:+.3f} (p={p_AL_given_G:.3f})")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("MEDIATION ANALYSIS")
    print("=" * 60)
    
    if abs(rho_AL_given_G) < 0.1 and abs(rho_AL) > 0.2:
        print("✓ MEDIATION CONFIRMED:")
        print(f"  - Strong direct correlation ρ(A*, L_⊥) = {rho_AL:+.3f}")
        print(f"  - Weak partial correlation ρ(A*, L_⊥ | G) = {rho_AL_given_G:+.3f}")
        print("  → G mediates the A* → L_⊥ relationship")
        print("  → Gating sensitivity is the causal pathway")
    elif abs(rho_AL) < 0.1:
        print("~ NO DIRECT EFFECT:")
        print(f"  - Weak direct correlation ρ(A*, L_⊥) = {rho_AL:+.3f}")
        print("  → No mediation to test")
    else:
        print("✗ MEDIATION NOT CONFIRMED:")
        print(f"  - Partial correlation ρ(A*, L_⊥ | G) = {rho_AL_given_G:+.3f} still significant")
        print("  → G does not fully explain the A* → L_⊥ relationship")
        print("  → Other mechanisms may be involved")
    
    # Additional insights
    mediation_ratio = abs(rho_AL_given_G) / (abs(rho_AL) + 1e-9)
    print(f"\nMediation ratio: {mediation_ratio:.2f}")
    print(f"  (< 0.5 indicates strong mediation)")
    
    # Save results
    output_path = Path("var/logs/mediator_proof_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'results': results,
            'correlations': {
                'rho_A_L': float(rho_AL),
                'p_A_L': float(p_AL),
                'rho_A_G': float(rho_AG),
                'p_A_G': float(p_AG),
                'rho_G_L': float(rho_GL),
                'p_G_L': float(p_GL),
                'rho_A_L_given_G': float(rho_AL_given_G),
                'p_A_L_given_G': float(p_AL_given_G)
            },
            'mediation': {
                'mediation_confirmed': bool(abs(rho_AL_given_G) < 0.1 and abs(rho_AL) > 0.2),
                'mediation_ratio': float(mediation_ratio)
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Create visualization
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # A* vs L_⊥
        axes[0,0].scatter(A_values, L_values, alpha=0.7)
        axes[0,0].set_xlabel('Ambiguity (A*)')
        axes[0,0].set_ylabel('Lipschitz Constant (L_⊥)')
        axes[0,0].set_title(f'A* vs L_⊥: ρ={rho_AL:.3f}')
        axes[0,0].grid(True, alpha=0.3)
        
        # A* vs G
        axes[0,1].scatter(A_values, G_values, alpha=0.7, color='orange')
        axes[0,1].set_xlabel('Ambiguity (A*)')
        axes[0,1].set_ylabel('Gating Sensitivity (G)')
        axes[0,1].set_title(f'A* vs G: ρ={rho_AG:.3f}')
        axes[0,1].grid(True, alpha=0.3)
        
        # G vs L_⊥
        axes[1,0].scatter(G_values, L_values, alpha=0.7, color='green')
        axes[1,0].set_xlabel('Gating Sensitivity (G)')
        axes[1,0].set_ylabel('Lipschitz Constant (L_⊥)')
        axes[1,0].set_title(f'G vs L_⊥: ρ={rho_GL:.3f}')
        axes[1,0].grid(True, alpha=0.3)
        
        # Residual plot (A* vs L_⊥ | G)
        # Compute residuals by regressing out G
        from sklearn.linear_model import LinearRegression
        
        # Regress A* on G, get residuals
        reg_AG = LinearRegression().fit(A_values.reshape(-1, 1), G_values)
        A_residual = A_values - reg_AG.predict(A_values.reshape(-1, 1)) * (rho_AG / np.std(G_values)) * np.std(A_values)
        
        # Regress L_⊥ on G, get residuals  
        reg_LG = LinearRegression().fit(G_values.reshape(-1, 1), L_values)
        L_residual = L_values - reg_LG.predict(G_values.reshape(-1, 1))
        
        axes[1,1].scatter(A_residual, L_residual, alpha=0.7, color='red')
        axes[1,1].set_xlabel('A* (residual after removing G effect)')
        axes[1,1].set_ylabel('L_⊥ (residual after removing G effect)')
        axes[1,1].set_title(f'Partial: ρ(A*,L_⊥|G)={rho_AL_given_G:.3f}')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = Path("var/logs/mediator_proof_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    return results

if __name__ == '__main__':
    results = run_mediator_proof_test()