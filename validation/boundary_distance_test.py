#!/usr/bin/env python3
"""
Boundary Distance Test
Tests if expansion stems from boundary flips in routing simplex
Key hypothesis: ρ(D, L_⊥) > ρ(H_route, L_⊥) where D = 1 - max_i(w_i)
"""

import numpy as np
# import pandas as pd  # Not needed
import json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
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

def compute_gating_weights(embeddings: List[np.ndarray], temperature: float = 1.0) -> np.ndarray:
    """Simulate gating weights from embeddings using cosine similarity to prototypes"""
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

def compute_boundary_distance(weights: np.ndarray) -> float:
    """
    Compute distance to simplex boundary
    D = 1 - max_i(w_i)
    Higher D means closer to center, lower D means closer to vertex
    """
    return 1.0 - np.max(weights)

def compute_routing_entropy(weights: np.ndarray) -> float:
    """
    Compute routing entropy H = -Σ w_i log(w_i)
    """
    return -np.sum(weights * np.log(weights + 1e-9))

def perturb_embedding(embedding: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
    """Add small perturbation to embedding"""
    noise = np.random.randn(*embedding.shape) * noise_scale
    perturbed = embedding + noise
    return perturbed / (np.linalg.norm(perturbed) + 1e-9)

def compute_expert_winner_changes(embeddings1: List[np.ndarray], 
                                 embeddings2: List[np.ndarray]) -> Dict:
    """
    Track which expert wins (highest weight) and how often it changes
    """
    w1 = compute_gating_weights(embeddings1)
    w2 = compute_gating_weights(embeddings2)
    
    winner1 = np.argmax(w1)
    winner2 = np.argmax(w2)
    
    winner_changed = winner1 != winner2
    
    # Compute confidence gap (difference between top-2 experts)
    sorted_w1 = np.sort(w1)
    sorted_w2 = np.sort(w2)
    
    gap1 = sorted_w1[-1] - sorted_w1[-2]  # Top - second
    gap2 = sorted_w2[-1] - sorted_w2[-2]
    
    return {
        'winner1': int(winner1),
        'winner2': int(winner2),
        'winner_changed': bool(winner_changed),
        'confidence_gap1': float(gap1),
        'confidence_gap2': float(gap2),
        'gap_change': float(abs(gap2 - gap1))
    }

def compute_lipschitz_with_boundary_analysis(prompt1: str, prompt2: str, 
                                           n_perturbations: int = 10) -> Optional[Dict]:
    """
    Compute Lipschitz constant while tracking boundary behavior
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
    
    # Base embeddings
    embeddings1 = context_embeddings + [emb1]
    embeddings2 = context_embeddings + [emb2]
    
    # Compute base routing
    w1_base = compute_gating_weights(embeddings1)
    w2_base = compute_gating_weights(embeddings2)
    
    # Boundary distances
    D1_base = compute_boundary_distance(w1_base)
    D2_base = compute_boundary_distance(w2_base)
    
    # Routing entropies
    H1_base = compute_routing_entropy(w1_base)
    H2_base = compute_routing_entropy(w2_base)
    
    # Winner analysis
    winner_info = compute_expert_winner_changes(embeddings1, embeddings2)
    
    # Perturbation analysis
    lipschitz_values = []
    boundary_distances = []
    routing_entropies = []
    winner_changes = []
    
    np.random.seed(42)
    
    for i in range(n_perturbations):
        # Perturb embeddings slightly
        emb1_pert = perturb_embedding(emb1, noise_scale=0.005)
        emb2_pert = perturb_embedding(emb2, noise_scale=0.005)
        
        # Compute perturbed routing
        embeddings1_pert = context_embeddings + [emb1_pert]
        embeddings2_pert = context_embeddings + [emb2_pert]
        
        w1_pert = compute_gating_weights(embeddings1_pert)
        w2_pert = compute_gating_weights(embeddings2_pert)
        
        # Compute changes
        delta_w = np.linalg.norm(w2_pert - w1_pert)
        delta_x = np.linalg.norm(emb2_pert - emb1_pert)
        L_local = delta_w / (delta_x + 1e-9)
        
        # Boundary distances
        D1_pert = compute_boundary_distance(w1_pert)
        D2_pert = compute_boundary_distance(w2_pert)
        D_avg = (D1_pert + D2_pert) / 2
        
        # Routing entropies
        H1_pert = compute_routing_entropy(w1_pert)
        H2_pert = compute_routing_entropy(w2_pert)
        H_avg = (H1_pert + H2_pert) / 2
        
        # Winner changes
        winner_pert_info = compute_expert_winner_changes(embeddings1_pert, embeddings2_pert)
        
        lipschitz_values.append(L_local)
        boundary_distances.append(D_avg)
        routing_entropies.append(H_avg)
        winner_changes.append(winner_pert_info['winner_changed'])
    
    # Overall Lipschitz constant (main input pair)
    delta_w_main = np.linalg.norm(w2_base - w1_base)
    delta_x_main = np.linalg.norm(emb2 - emb1)
    L_main = delta_w_main / (delta_x_main + 1e-9)
    
    return {
        'L_main': float(L_main),
        'D1': float(D1_base),
        'D2': float(D2_base),
        'D_avg': float((D1_base + D2_base) / 2),
        'H1': float(H1_base),
        'H2': float(H2_base),
        'H_avg': float((H1_base + H2_base) / 2),
        'winner_info': winner_info,
        'perturbation_analysis': {
            'L_values': lipschitz_values,
            'D_values': boundary_distances,
            'H_values': routing_entropies,
            'winner_flip_rate': float(np.mean(winner_changes)),
            'L_std': float(np.std(lipschitz_values)),
            'D_std': float(np.std(boundary_distances)),
            'H_std': float(np.std(routing_entropies))
        },
        'correlations_within_perturbations': {
            'rho_D_L': float(spearmanr(boundary_distances, lipschitz_values)[0]),
            'rho_H_L': float(spearmanr(routing_entropies, lipschitz_values)[0])
        }
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

def run_boundary_distance_test():
    """
    Main test: Compare ρ(D, L_⊥) vs ρ(H_route, L_⊥)
    """
    print("=" * 60)
    print("BOUNDARY DISTANCE TEST")
    print("Testing if boundary flips drive expansion")
    print("=" * 60)
    
    # Test prompt pairs with varying ambiguity and routing patterns
    test_pairs = [
        # Low ambiguity (should have decisive routing, low D)
        ("What is 2+2?", "What is 3+3?"),
        ("The capital of France is", "The capital of Spain is"),
        ("Calculate 15 * 3", "Calculate 18 * 2"),
        ("Name the largest planet", "Name the smallest planet"),
        
        # Medium ambiguity (mixed routing patterns)
        ("Explain photosynthesis briefly", "Explain respiration briefly"),
        ("What causes earthquakes?", "What causes volcanoes?"),
        ("How do birds fly?", "How do fish swim?"),
        ("Describe a forest", "Describe a desert"),
        
        # High ambiguity (should have uncertain routing, high D)
        ("What is the meaning of existence?", "What is the nature of reality?"),
        ("Explain consciousness and awareness", "Explain intelligence and thought"),
        ("What emerges from complex systems?", "What emerges from simple rules?"),
        ("The boundary between order and", "The boundary between chaos and"),
        ("Recursive patterns in nature", "Self-similar structures everywhere"),
        ("Meta-cognitive reflection on", "Higher-order thinking about")
    ]
    
    print(f"\nTesting {len(test_pairs)} prompt pairs...")
    
    results = []
    
    for i, (prompt1, prompt2) in enumerate(test_pairs):
        print(f"\nPair {i+1}: {prompt1[:30]}... vs {prompt2[:30]}...")
        
        # Compute ambiguity
        A1 = estimate_ambiguity(prompt1)
        A2 = estimate_ambiguity(prompt2)
        A_star = (A1 + A2) / 2
        
        # Compute boundary analysis
        boundary_result = compute_lipschitz_with_boundary_analysis(prompt1, prompt2)
        
        if boundary_result is not None:
            print(f"  A* = {A_star:.3f}")
            print(f"  L_main = {boundary_result['L_main']:.4f}")
            print(f"  D_avg = {boundary_result['D_avg']:.3f} (boundary distance)")
            print(f"  H_avg = {boundary_result['H_avg']:.3f} (routing entropy)")
            print(f"  Winner changed: {boundary_result['winner_info']['winner_changed']}")
            print(f"  Winner flip rate: {boundary_result['perturbation_analysis']['winner_flip_rate']:.2f}")
            
            results.append({
                'pair_id': i + 1,
                'prompt1': prompt1,
                'prompt2': prompt2,
                'A_star': A_star,
                **boundary_result
            })
        else:
            print(f"  Skipped (embedding failed)")
    
    if len(results) < 3:
        print("\nInsufficient data for correlation analysis")
        return
    
    # Extract main variables for analysis
    A_values = np.array([r['A_star'] for r in results])
    L_values = np.array([r['L_main'] for r in results])
    D_values = np.array([r['D_avg'] for r in results])
    H_values = np.array([r['H_avg'] for r in results])
    flip_rates = np.array([r['perturbation_analysis']['winner_flip_rate'] for r in results])
    
    # Compute correlations
    print("\n" + "=" * 60)
    print("BOUNDARY VS ENTROPY ANALYSIS")
    print("=" * 60)
    
    # Key comparison: D vs H as predictors of L
    rho_DL, p_DL = spearmanr(D_values, L_values)
    rho_HL, p_HL = spearmanr(H_values, L_values)
    rho_AL, p_AL = spearmanr(A_values, L_values)
    rho_FL, p_FL = spearmanr(flip_rates, L_values)
    
    print(f"Correlations with Lipschitz constant L_⊥:")
    print(f"  ρ(D, L_⊥)        = {rho_DL:+.3f} (p={p_DL:.3f})  [Boundary distance]")
    print(f"  ρ(H, L_⊥)        = {rho_HL:+.3f} (p={p_HL:.3f})  [Routing entropy]")
    print(f"  ρ(A*, L_⊥)       = {rho_AL:+.3f} (p={p_AL:.3f})  [Ambiguity]")
    print(f"  ρ(FlipRate, L_⊥) = {rho_FL:+.3f} (p={p_FL:.3f})  [Winner instability]")
    
    # Cross-correlations
    rho_AD, p_AD = spearmanr(A_values, D_values)
    rho_AH, p_AH = spearmanr(A_values, H_values)
    rho_DH, p_DH = spearmanr(D_values, H_values)
    
    print(f"\nCross-correlations:")
    print(f"  ρ(A*, D) = {rho_AD:+.3f} (p={p_AD:.3f})  [Ambiguity vs boundary distance]")
    print(f"  ρ(A*, H) = {rho_AH:+.3f} (p={p_AH:.3f})  [Ambiguity vs entropy]")
    print(f"  ρ(D, H)  = {rho_DH:+.3f} (p={p_DH:.3f})  [Boundary vs entropy]")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("BOUNDARY FLIP HYPOTHESIS")
    print("=" * 60)
    
    if abs(rho_DL) > abs(rho_HL) and abs(rho_DL) > 0.3:
        print("✓ BOUNDARY FLIPS CONFIRMED:")
        print(f"  - Boundary distance predicts expansion better: |ρ(D,L)| = {abs(rho_DL):.3f} > |ρ(H,L)| = {abs(rho_HL):.3f}")
        print("  - Expansion stems from routing decisions near simplex vertices")
        print("  - Small input changes flip expert winners → large output changes")
    elif abs(rho_HL) > abs(rho_DL) and abs(rho_HL) > 0.3:
        print("~ ENTROPY DOMINATES:")
        print(f"  - Routing entropy predicts expansion better: |ρ(H,L)| = {abs(rho_HL):.3f} > |ρ(D,L)| = {abs(rho_DL):.3f}")
        print("  - Expansion relates to overall routing uncertainty, not boundary effects")
    else:
        print("✗ INCONCLUSIVE:")
        print(f"  - Weak correlations: |ρ(D,L)| = {abs(rho_DL):.3f}, |ρ(H,L)| = {abs(rho_HL):.3f}")
        print("  - Neither boundary distance nor entropy strongly predicts expansion")
    
    # Winner instability analysis
    if abs(rho_FL) > 0.4:
        print(f"\n⚡ WINNER INSTABILITY SIGNIFICANT:")
        print(f"  - Strong correlation ρ(FlipRate, L_⊥) = {rho_FL:+.3f}")
        print("  - Expansion linked to expert winner changes")
    
    # Save results
    output_path = Path("var/logs/boundary_distance_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'results': results,
            'correlations': {
                'rho_D_L': float(rho_DL),
                'p_D_L': float(p_DL),
                'rho_H_L': float(rho_HL),
                'p_H_L': float(p_HL),
                'rho_A_L': float(rho_AL),
                'p_A_L': float(p_AL),
                'rho_FlipRate_L': float(rho_FL),
                'p_FlipRate_L': float(p_FL),
                'rho_A_D': float(rho_AD),
                'rho_A_H': float(rho_AH),
                'rho_D_H': float(rho_DH)
            },
            'hypothesis_test': {
                'boundary_stronger': bool(abs(rho_DL) > abs(rho_HL)),
                'boundary_significant': bool(abs(rho_DL) > 0.3),
                'entropy_significant': bool(abs(rho_HL) > 0.3),
                'winner_instability_significant': bool(abs(rho_FL) > 0.4)
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Create visualization
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # D vs L
        axes[0,0].scatter(D_values, L_values, alpha=0.7, s=60)
        axes[0,0].set_xlabel('Boundary Distance (D)')
        axes[0,0].set_ylabel('Lipschitz Constant (L_⊥)')
        axes[0,0].set_title(f'Boundary vs Expansion\nρ={rho_DL:.3f}')
        axes[0,0].grid(True, alpha=0.3)
        
        # H vs L
        axes[0,1].scatter(H_values, L_values, alpha=0.7, s=60, color='orange')
        axes[0,1].set_xlabel('Routing Entropy (H)')
        axes[0,1].set_ylabel('Lipschitz Constant (L_⊥)')
        axes[0,1].set_title(f'Entropy vs Expansion\nρ={rho_HL:.3f}')
        axes[0,1].grid(True, alpha=0.3)
        
        # FlipRate vs L
        axes[0,2].scatter(flip_rates, L_values, alpha=0.7, s=60, color='red')
        axes[0,2].set_xlabel('Winner Flip Rate')
        axes[0,2].set_ylabel('Lipschitz Constant (L_⊥)')
        axes[0,2].set_title(f'Winner Instability vs Expansion\nρ={rho_FL:.3f}')
        axes[0,2].grid(True, alpha=0.3)
        
        # A* vs D
        axes[1,0].scatter(A_values, D_values, alpha=0.7, s=60, color='green')
        axes[1,0].set_xlabel('Ambiguity (A*)')
        axes[1,0].set_ylabel('Boundary Distance (D)')
        axes[1,0].set_title(f'Ambiguity vs Boundary\nρ={rho_AD:.3f}')
        axes[1,0].grid(True, alpha=0.3)
        
        # A* vs H
        axes[1,1].scatter(A_values, H_values, alpha=0.7, s=60, color='purple')
        axes[1,1].set_xlabel('Ambiguity (A*)')
        axes[1,1].set_ylabel('Routing Entropy (H)')
        axes[1,1].set_title(f'Ambiguity vs Entropy\nρ={rho_AH:.3f}')
        axes[1,1].grid(True, alpha=0.3)
        
        # D vs H
        axes[1,2].scatter(D_values, H_values, alpha=0.7, s=60, color='brown')
        axes[1,2].set_xlabel('Boundary Distance (D)')
        axes[1,2].set_ylabel('Routing Entropy (H)')
        axes[1,2].set_title(f'Boundary vs Entropy\nρ={rho_DH:.3f}')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = Path("var/logs/boundary_distance_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    return results

if __name__ == '__main__':
    results = run_boundary_distance_test()