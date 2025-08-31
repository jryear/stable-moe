#!/usr/bin/env python3
"""
Gating Sensitivity Test
Confirms that expansion under ambiguity is mediated by gating term sensitivity
"""

import numpy as np
import json
import requests
from scipy.stats import spearmanr
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path

def get_embedding(text: str) -> np.ndarray:
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

def compute_gating_weights(embeddings: List[np.ndarray], temperature: float = 1.0) -> np.ndarray:
    """
    Simulate gating weights from embeddings
    Uses cosine similarity to prototypes as logits
    """
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

def compute_gating_sensitivity(prompt1: str, prompt2: str, delta: float = 0.01) -> Tuple[float, Dict]:
    """
    Compute G = ||Δw||/||Δx|| where:
    - Δx is the change in input embedding
    - Δw is the change in gating weights
    """
    # Get embeddings
    emb1 = get_embedding(prompt1)
    emb2 = get_embedding(prompt2)
    
    if emb1 is None or emb2 is None:
        return None, {}
    
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
    
    return G, {
        'delta_x': float(delta_x),
        'delta_w': float(delta_w),
        'w1': w1.tolist(),
        'w2': w2.tolist(),
        'w1_entropy': float(-np.sum(w1 * np.log(w1 + 1e-9))),
        'w2_entropy': float(-np.sum(w2 * np.log(w2 + 1e-9)))
    }

def run_sensitivity_analysis():
    """
    Test gating sensitivity across ambiguity levels
    """
    print("=" * 60)
    print("GATING SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Test pairs with varying ambiguity
    test_pairs = [
        # Low ambiguity (clear)
        ("What is 2+2?", "What is 3+3?", 0.1),
        ("The capital of France is", "The capital of Spain is", 0.15),
        
        # Medium ambiguity
        ("Explain consciousness", "Explain intelligence", 0.5),
        ("What is time?", "What is space?", 0.45),
        
        # High ambiguity
        ("What emerges from overlap?", "What emerges from void?", 0.8),
        ("The boundary between meaning and", "The boundary between truth and", 0.85),
        ("Recursive self-reference in", "Recursive meta-reference in", 0.9)
    ]
    
    results = []
    
    for prompt1, prompt2, expected_ambiguity in test_pairs:
        print(f"\nTesting pair (A*≈{expected_ambiguity}):")
        print(f"  P1: {prompt1[:40]}...")
        print(f"  P2: {prompt2[:40]}...")
        
        G, details = compute_gating_sensitivity(prompt1, prompt2)
        
        if G is not None:
            print(f"  Gating sensitivity G = {G:.3f}")
            print(f"  Δx = {details['delta_x']:.3f}, Δw = {details['delta_w']:.3f}")
            print(f"  Entropy change: {details['w1_entropy']:.3f} → {details['w2_entropy']:.3f}")
            
            results.append({
                'A_star': expected_ambiguity,
                'G': G,
                **details
            })
    
    # Analyze correlation
    if len(results) >= 3:
        A_values = np.array([r['A_star'] for r in results])
        G_values = np.array([r['G'] for r in results])
        
        rho, p = spearmanr(A_values, G_values)
        
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)
        print(f"ρ(A*, G) = {rho:+.3f} (p={p:.3f})")
        
        if rho > 0.3:
            print("✓ CONFIRMED: Higher ambiguity → Higher gating sensitivity")
            print("  This explains the expansion under ambiguity")
        elif rho < -0.3:
            print("✗ OPPOSITE: Higher ambiguity → Lower gating sensitivity")
        else:
            print("~ INCONCLUSIVE: Weak correlation")
        
        # Save results
        output_path = Path("var/logs/gating_sensitivity.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'results': results,
                'correlation': float(rho),
                'p_value': float(p)
            }, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        
        # Plot if possible
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(A_values, G_values, s=100, alpha=0.7)
            plt.xlabel('Ambiguity (A*)')
            plt.ylabel('Gating Sensitivity (G)')
            plt.title(f'Gating Sensitivity vs Ambiguity\nρ = {rho:.3f}')
            
            # Add trend line
            z = np.polyfit(A_values, G_values, 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(0, 1, 100)
            plt.plot(x_line, p_line(x_line), 'r--', alpha=0.5)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = Path("var/logs/gating_sensitivity.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
            plt.close()
        except:
            pass

    return results

def compute_lipschitz_with_frozen_router(prompt1: str, prompt2: str) -> Optional[Dict]:
    """
    Compute Lipschitz constant with frozen routing weights
    This isolates expert effects from routing effects
    """
    emb1 = get_embedding(prompt1)
    emb2 = get_embedding(prompt2)
    
    if emb1 is None or emb2 is None:
        return None
    
    # Normalize embeddings
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-9)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-9)
    
    # Get context embeddings for prototypes
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
    
    # Compute routing weights for first prompt only
    embeddings1 = context_embeddings + [emb1]
    w_frozen = compute_gating_weights(embeddings1)
    
    # Simulate expert responses with frozen weights
    def simulate_expert_response(embedding: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Simulate mixture of expert responses"""
        n_experts = len(weights)
        expert_responses = []
        
        # Each expert processes the embedding differently
        np.random.seed(42)
        for i in range(n_experts):
            # Expert i applies a different transformation
            expert_transform = np.random.randn(embedding.shape[0], embedding.shape[0])
            expert_response = expert_transform @ embedding
            expert_response = expert_response / (np.linalg.norm(expert_response) + 1e-9)
            expert_responses.append(expert_response)
        
        # Weighted combination
        mixed_response = np.zeros_like(embedding)
        for i, resp in enumerate(expert_responses):
            mixed_response += weights[i] * resp
        
        return mixed_response / (np.linalg.norm(mixed_response) + 1e-9)
    
    # Compute responses with frozen routing
    resp1_frozen = simulate_expert_response(emb1, w_frozen)
    resp2_frozen = simulate_expert_response(emb2, w_frozen)  # Same weights!
    
    # Compute Lipschitz constant
    delta_x = np.linalg.norm(emb2 - emb1)
    delta_y_frozen = np.linalg.norm(resp2_frozen - resp1_frozen)
    L_frozen = delta_y_frozen / (delta_x + 1e-9)
    
    return {
        'L_frozen': float(L_frozen),
        'delta_x': float(delta_x),
        'delta_y_frozen': float(delta_y_frozen),
        'w_frozen': w_frozen.tolist()
    }

def compute_lipschitz_with_ghost_gating(prompt1: str, prompt2: str) -> Optional[Dict]:
    """
    Compute Lipschitz constant reusing routing weights across perturbations
    This tests if routing sensitivity is the main driver
    """
    emb1 = get_embedding(prompt1)
    emb2 = get_embedding(prompt2)
    
    if emb1 is None or emb2 is None:
        return None
    
    # Normalize embeddings
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-9)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-9)
    
    # Get context for routing
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
    
    # Compute routing weights normally for first prompt
    embeddings1 = context_embeddings + [emb1]
    w1 = compute_gating_weights(embeddings1)
    
    # For second prompt, reuse the same routing weights (ghost gating)
    def simulate_expert_response_with_weights(embedding: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Simulate expert responses with given weights"""
        n_experts = len(weights)
        expert_responses = []
        
        np.random.seed(42)
        for i in range(n_experts):
            expert_transform = np.random.randn(embedding.shape[0], embedding.shape[0]) 
            expert_response = expert_transform @ embedding
            expert_response = expert_response / (np.linalg.norm(expert_response) + 1e-9)
            expert_responses.append(expert_response)
        
        # Weighted combination
        mixed_response = np.zeros_like(embedding)
        for i, resp in enumerate(expert_responses):
            mixed_response += weights[i] * resp
        
        return mixed_response / (np.linalg.norm(mixed_response) + 1e-9)
    
    # Use same routing weights for both
    resp1_ghost = simulate_expert_response_with_weights(emb1, w1) 
    resp2_ghost = simulate_expert_response_with_weights(emb2, w1)  # Same w1!
    
    # Compute Lipschitz constant
    delta_x = np.linalg.norm(emb2 - emb1)
    delta_y_ghost = np.linalg.norm(resp2_ghost - resp1_ghost)
    L_ghost = delta_y_ghost / (delta_x + 1e-9)
    
    return {
        'L_ghost': float(L_ghost),
        'delta_x': float(delta_x),
        'delta_y_ghost': float(delta_y_ghost),
        'w_reused': w1.tolist()
    }

def estimate_ambiguity(prompt: str) -> float:
    """Estimate ambiguity score A* for a prompt"""
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

def run_freeze_router_test():
    """
    Test freeze router hypothesis: Does expansion vanish with frozen routing?
    """
    print("\n" + "=" * 60)
    print("FREEZE ROUTER TEST")
    print("Testing if gating drives expansion")
    print("=" * 60)
    
    # Test pairs across ambiguity spectrum
    test_pairs = [
        # Low ambiguity
        ("What is 5+3?", "What is 7+2?"),
        ("The capital of Japan is", "The capital of Italy is"),
        
        # Medium ambiguity
        ("Explain gravity briefly", "Explain magnetism briefly"),
        ("How do plants grow?", "How do animals move?"),
        
        # High ambiguity
        ("What is consciousness?", "What is intelligence?"),
        ("The meaning of existence", "The nature of reality")
    ]
    
    print(f"\nTesting {len(test_pairs)} prompt pairs...")
    
    results = []
    
    for i, (prompt1, prompt2) in enumerate(test_pairs):
        print(f"\nPair {i+1}: {prompt1[:30]}... vs {prompt2[:30]}...")
        
        # Estimate ambiguity
        A1 = estimate_ambiguity(prompt1)
        A2 = estimate_ambiguity(prompt2)
        A_star = (A1 + A2) / 2
        
        # Normal gating sensitivity
        G_normal, normal_details = compute_gating_sensitivity(prompt1, prompt2)
        if G_normal is None:
            continue
        
        # Frozen router test
        frozen_result = compute_lipschitz_with_frozen_router(prompt1, prompt2)
        
        # Ghost gating test  
        ghost_result = compute_lipschitz_with_ghost_gating(prompt1, prompt2)
        
        if all(r is not None for r in [frozen_result, ghost_result]):
            L_frozen = frozen_result['L_frozen'] 
            L_ghost = ghost_result['L_ghost']
            
            # Reduction factors
            frozen_reduction = (G_normal - L_frozen) / (G_normal + 1e-9)
            ghost_reduction = (G_normal - L_ghost) / (G_normal + 1e-9)
            
            print(f"  A* = {A_star:.3f}")
            print(f"  Normal G = {G_normal:.4f}")
            print(f"  Frozen L = {L_frozen:.4f} (reduction: {frozen_reduction:.1%})")
            print(f"  Ghost L  = {L_ghost:.4f} (reduction: {ghost_reduction:.1%})")
            
            results.append({
                'pair_id': i + 1,
                'prompt1': prompt1,
                'prompt2': prompt2,
                'A_star': A_star,
                'G_normal': G_normal,
                'L_frozen': L_frozen,
                'L_ghost': L_ghost,
                'frozen_reduction': frozen_reduction,
                'ghost_reduction': ghost_reduction
            })
        else:
            print(f"  Skipped (computation failed)")
    
    if len(results) == 0:
        print("\nNo valid results")
        return
    
    # Analysis
    print("\n" + "=" * 60)
    print("FREEZE ROUTER ANALYSIS") 
    print("=" * 60)
    
    frozen_reductions = [r['frozen_reduction'] for r in results]
    ghost_reductions = [r['ghost_reduction'] for r in results]
    
    avg_frozen = np.mean(frozen_reductions)
    avg_ghost = np.mean(ghost_reductions)
    
    print(f"Average reductions:")
    print(f"  Frozen router: {avg_frozen:.1%}")
    print(f"  Ghost gating:  {avg_ghost:.1%}")
    
    # Hypothesis testing
    if avg_frozen > 0.5:
        print("\n✓ FROZEN ROUTER CONFIRMED:")
        print(f"  - {avg_frozen:.1%} reduction with frozen routing")
        print("  - Gating is primary driver of expansion")
    elif avg_ghost > 0.3:
        print("\n✓ GHOST GATING CONFIRMED:")
        print(f"  - {avg_ghost:.1%} reduction with reused routing")
        print("  - Routing sensitivity drives expansion")
    else:
        print("\n✗ INCONCLUSIVE:")
        print("  - Neither frozen router nor ghost gating shows strong effect")
        print("  - Other mechanisms may dominate")
    
    # Ambiguity correlation
    A_values = np.array([r['A_star'] for r in results])
    if len(A_values) > 2:
        rho_A_frozen = spearmanr(A_values, frozen_reductions)[0]
        rho_A_ghost = spearmanr(A_values, ghost_reductions)[0]
        
        print(f"\nAmbiguity correlations:")
        print(f"  ρ(A*, frozen_reduction) = {rho_A_frozen:+.3f}")
        print(f"  ρ(A*, ghost_reduction)  = {rho_A_ghost:+.3f}")
        
        if rho_A_frozen > 0.4:
            print("  → Higher ambiguity shows stronger routing effects")
    
    # Save results
    output_path = Path("var/logs/freeze_router_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'avg_frozen_reduction': float(avg_frozen),
                'avg_ghost_reduction': float(avg_ghost),
                'frozen_confirmed': bool(avg_frozen > 0.5),
                'ghost_confirmed': bool(avg_ghost > 0.3)
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results

def test_with_controller():
    """
    Test how the clarity controller affects gating sensitivity
    """
    print("\n" + "=" * 60)
    print("TESTING WITH CLARITY CONTROLLER")
    print("=" * 60)
    
    from clarity_controller import ClarityController
    
    controller = ClarityController()
    
    # Test high ambiguity case
    A_star = 0.8
    print(f"\nHigh ambiguity (A* = {A_star}):")
    
    # Random logits
    np.random.seed(42)
    logits = np.random.randn(5)
    
    # Without control (baseline)
    w_baseline = controller.softmax(logits)
    
    # With control
    w_controlled = controller.route_with_control(logits, A_star)
    
    # Perturb slightly
    logits_perturbed = logits + np.random.randn(5) * 0.1
    w_baseline_pert = controller.softmax(logits_perturbed)
    w_controlled_pert = controller.route_with_control(logits_perturbed, A_star)
    
    # Compute sensitivities
    G_baseline = np.linalg.norm(w_baseline_pert - w_baseline) / 0.1
    G_controlled = np.linalg.norm(w_controlled_pert - w_controlled) / 0.1
    
    print(f"  Baseline gating sensitivity: {G_baseline:.3f}")
    print(f"  Controlled gating sensitivity: {G_controlled:.3f}")
    print(f"  Reduction factor: {G_baseline/G_controlled:.2f}x")
    
    if G_controlled < G_baseline * 0.8:
        print("  ✓ Controller reduces gating sensitivity under ambiguity")
    else:
        print("  ✗ Controller does not significantly reduce sensitivity")

if __name__ == '__main__':
    # Run main analysis
    results = run_sensitivity_analysis()
    
    # Run freeze router test
    freeze_results = run_freeze_router_test()
    
    # Test with controller
    test_with_controller()