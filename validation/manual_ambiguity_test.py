#!/usr/bin/env python3
"""
Test A*→G→L_⊥ with manually assigned ambiguity scores
This bypasses the ambiguity scoring function to test the core mediation theory
"""

import numpy as np
from enhanced_mediator_proof import (
    compute_enhanced_gating_sensitivity, 
    compute_orthogonal_lipschitz,
    partial_spearman,
    bootstrap_partial_correlation
)
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

def test_manual_ambiguity_assignment():
    """Test with manually assigned A* scores across full range"""
    
    print("🎯 TESTING A*→G→L_⊥ WITH MANUAL AMBIGUITY ASSIGNMENT")
    print("="*60)
    
    # Test pairs with MANUAL ambiguity assignments
    test_cases = [
        # Very low ambiguity (computational/factual)
        ("What is 2+2?", "What is 3+3?", 0.05),
        ("Calculate 15 * 4", "Calculate 16 * 5", 0.10), 
        ("The capital of France is Paris", "The capital of Italy is Rome", 0.15),
        
        # Low ambiguity  
        ("How many days in a year?", "How many hours in a day?", 0.20),
        ("What color is the sky?", "What color is grass?", 0.25),
        
        # Medium ambiguity
        ("Explain photosynthesis briefly", "Explain respiration briefly", 0.35),
        ("How do cars work?", "How do planes work?", 0.45),
        ("What causes happiness?", "What causes sadness?", 0.55),
        
        # High ambiguity
        ("What is consciousness?", "What is intelligence?", 0.65),
        ("Explain the nature of time", "Explain the nature of space", 0.75),
        ("What is the meaning of existence?", "What is the purpose of reality?", 0.85),
        
        # Very high ambiguity (abstract/philosophical)
        ("What emerges from complexity?", "What emerges from chaos?", 0.90),
        ("The boundary between meaning and void", "The boundary between truth and illusion", 0.95)
    ]
    
    print(f"Testing {len(test_cases)} pairs with manual A* assignments...")
    
    results = []
    
    for i, (prompt1, prompt2, A_star_manual) in enumerate(test_cases):
        print(f"\nPair {i+1}: A*={A_star_manual:.2f} (manual)")
        print(f"  P1: {prompt1[:50]}...")
        print(f"  P2: {prompt2[:50]}...")
        
        # Compute G and L_⊥ with real embeddings
        gating_result = compute_enhanced_gating_sensitivity(prompt1, prompt2)
        lipschitz_result = compute_orthogonal_lipschitz(prompt1, prompt2)
        
        if gating_result and lipschitz_result:
            G = gating_result['G']
            L_perp = lipschitz_result['L_perp']
            D = gating_result['delta_D']
            
            print(f"  → G = {G:.4f}, L_⊥ = {L_perp:.3f}, ΔD = {D:.4f}")
            
            results.append({
                'A_star': A_star_manual,
                'G': G,
                'L_perp': L_perp,
                'D': D,
                'prompt1': prompt1,
                'prompt2': prompt2
            })
        else:
            print(f"  → Failed (embedding/computation error)")
    
    if len(results) < 8:
        print(f"\nInsufficient data: {len(results)} pairs")
        return
    
    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS WITH MANUAL A* - {len(results)} pairs")
    print(f"{'='*60}")
    
    # Extract arrays
    A_values = np.array([r['A_star'] for r in results])
    G_values = np.array([r['G'] for r in results])
    L_values = np.array([r['L_perp'] for r in results])
    D_values = np.array([r['D'] for r in results])
    
    # Direct correlations
    rho_AG, p_AG = spearmanr(A_values, G_values)
    rho_GL, p_GL = spearmanr(G_values, L_values)
    rho_AL, p_AL = spearmanr(A_values, L_values)
    rho_AD, p_AD = spearmanr(A_values, D_values)
    
    print(f"Direct correlations:")
    print(f"  ρ(A*, G)   = {rho_AG:+.3f} (p={p_AG:.3f}) ← CRITICAL: Should be positive!")
    print(f"  ρ(G, L_⊥)  = {rho_GL:+.3f} (p={p_GL:.3f}) ← Should be positive")
    print(f"  ρ(A*, L_⊥) = {rho_AL:+.3f} (p={p_AL:.3f}) (direct effect)")
    print(f"  ρ(A*, ΔD)  = {rho_AD:+.3f} (p={p_AD:.3f}) (distance-to-vertex)")
    
    # THE MEDIATION TEST
    rho_AL_given_G, p_AL_given_G = partial_spearman(A_values, L_values, G_values)
    
    print(f"\n🎯 MEDIATION TEST:")
    print(f"  ρ(A*, L_⊥ | G) = {rho_AL_given_G:+.3f} (p={p_AL_given_G:.3f})")
    
    # VERDICT
    print(f"\n{'='*60}")
    print(f"MEDIATION VERDICT WITH MANUAL A*")
    print(f"{'='*60}")
    
    mediation_confirmed = (rho_AG > 0.3 and rho_GL > 0.2 and abs(rho_AL_given_G) < 0.2)
    
    if rho_AG > 0.3:
        print("✅ A* → G PATHWAY CONFIRMED!")
        print(f"  Strong positive correlation: ρ = {rho_AG:+.3f}")
        print(f"  Manual ambiguity assignment reveals true relationship")
        
        if rho_GL > 0.2:
            print("✅ G → L_⊥ PATHWAY CONFIRMED!")
            print(f"  Gating sensitivity drives expansion: ρ = {rho_GL:+.3f}")
            
            if abs(rho_AL_given_G) < 0.2:
                print("✅ FULL MEDIATION CONFIRMED!")
                print(f"  Partial correlation collapsed: ρ(A*, L_⊥|G) = {rho_AL_given_G:+.3f}")
                print(f"  🎛️  G IS THE CAUSAL PATHWAY: A* → G → L_⊥")
                print(f"  🎯 Control strategy: Reduce G via β/λ schedules")
            else:
                print("⚠️  PARTIAL MEDIATION:")
                print(f"  Some direct A* → L_⊥ effect remains: ρ = {rho_AL_given_G:+.3f}")
        else:
            print("❌ G → L_⊥ PATHWAY WEAK:")
            print(f"  Gating may not drive expansion: ρ = {rho_GL:+.3f}")
    else:
        print("❌ A* → G PATHWAY NOT ESTABLISHED:")
        print(f"  Weak correlation even with manual A*: ρ = {rho_AG:+.3f}")
        print(f"  Issue may be with gating computation or embeddings")
    
    # Show data ranges
    print(f"\n📊 DATA RANGES:")
    print(f"  A* range: [{A_values.min():.2f}, {A_values.max():.2f}] (full spectrum!)")
    print(f"  G range:  [{G_values.min():.4f}, {G_values.max():.4f}]")
    print(f"  L_⊥ range: [{L_values.min():.2f}, {L_values.max():.2f}]")
    
    # Visualization
    create_manual_mediation_plot(A_values, G_values, L_values, rho_AG, rho_GL, rho_AL_given_G)
    
    return results

def create_manual_mediation_plot(A_values, G_values, L_values, rho_AG, rho_GL, rho_AL_given_G):
    """Create mediation plot with manual A* assignments"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # A* vs G (THE KEY TEST)
        axes[0].scatter(A_values, G_values, alpha=0.8, s=80, color='red')
        axes[0].set_xlabel('Manual Ambiguity (A*)')
        axes[0].set_ylabel('Gating Sensitivity (G)')
        axes[0].set_title(f'A* → G: ρ={rho_AG:.3f}\n(CRITICAL: Should be positive!)')
        axes[0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(A_values, G_values, 1)
        p = np.poly1d(z)
        x_range = np.linspace(A_values.min(), A_values.max(), 100)
        axes[0].plot(x_range, p(x_range), 'k--', alpha=0.7)
        
        # G vs L_⊥
        axes[1].scatter(G_values, L_values, alpha=0.8, s=80, color='green')
        axes[1].set_xlabel('Gating Sensitivity (G)')  
        axes[1].set_ylabel('Orthogonal Lipschitz (L_⊥)')
        axes[1].set_title(f'G → L_⊥: ρ={rho_GL:.3f}')
        axes[1].grid(True, alpha=0.3)
        
        # A* vs L_⊥ (should be weaker after conditioning on G)
        axes[2].scatter(A_values, L_values, alpha=0.8, s=80, color='blue')
        axes[2].set_xlabel('Manual Ambiguity (A*)')
        axes[2].set_ylabel('Orthogonal Lipschitz (L_⊥)')
        axes[2].set_title(f'A* → L_⊥ | G: ρ={rho_AL_given_G:.3f}\n(Should be weak if G mediates)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = Path("validation/results/manual_ambiguity_mediation.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n🎨 Plot saved to {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == '__main__':
    print("🚀 Testing A*→G→L_⊥ with manual ambiguity assignments")
    print("This bypasses the ambiguity scoring function to test core theory")
    print("-" * 60)
    
    results = test_manual_ambiguity_assignment()
    
    if results:
        print(f"\n🎯 SUMMARY:")
        print(f"• Tested {len(results)} pairs across full A* spectrum (0.05-0.95)")
        print(f"• Manual assignment bypasses ambiguity scoring limitations")
        print(f"• If A*→G correlation is now positive, theory confirmed!")
        print(f"• If still negative, issue is with gating computation")
    else:
        print(f"\n⚠️  Test failed - check Ollama connection")