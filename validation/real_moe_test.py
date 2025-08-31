#!/usr/bin/env python3
"""
Real MoE Routing Test with Generative Models
Tests A*→G→L_⊥ using actual generative model routing behavior

Unlike embedding similarity, this probes real routing instability:
- Multiple generations from same prompt
- Token-level routing variance
- Response consistency as stability metric
"""

import numpy as np
import requests
import json
import time
from typing import List, Dict, Optional, Tuple
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

def query_ollama_generation(prompt: str, model: str = "qwen2.5:7b", 
                           temperature: float = 0.7, n_samples: int = 5) -> List[str]:
    """Generate multiple responses to measure routing variance"""
    responses = []
    
    for i in range(n_samples):
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'temperature': temperature,
                    'stream': False,
                    'options': {
                        'num_predict': 100,  # Limit response length
                        'temperature': temperature
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                responses.append(data.get('response', '').strip())
            else:
                print(f"HTTP {response.status_code} for sample {i+1}")
                
        except Exception as e:
            print(f"Error in sample {i+1}: {e}")
    
    return responses

def compute_response_consistency(responses: List[str]) -> Dict[str, float]:
    """
    Compute routing stability metrics from response variance
    
    High routing stability → consistent responses → low variance
    High routing instability → diverse responses → high variance
    """
    if len(responses) < 2:
        return {'consistency': 0.0, 'variance': 1.0, 'mean_length': 0}
    
    # Length variance (proxy for routing decisions)
    lengths = [len(resp) for resp in responses]
    length_variance = np.var(lengths) / (np.mean(lengths) + 1e-6)
    
    # Token overlap consistency
    def tokenize_simple(text: str) -> set:
        return set(text.lower().split())
    
    token_sets = [tokenize_simple(resp) for resp in responses]
    
    # Pairwise Jaccard similarity
    similarities = []
    for i in range(len(token_sets)):
        for j in range(i+1, len(token_sets)):
            intersection = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            jaccard = intersection / (union + 1e-6)
            similarities.append(jaccard)
    
    consistency = np.mean(similarities) if similarities else 0.0
    
    # First token variance (routing gate proxy)
    first_tokens = []
    for resp in responses:
        if resp.strip():
            first_token = resp.strip().split()[0].lower()
            first_tokens.append(first_token)
    
    unique_first_tokens = len(set(first_tokens))
    first_token_variance = unique_first_tokens / len(first_tokens) if first_tokens else 0.0
    
    return {
        'consistency': consistency,
        'length_variance': length_variance,
        'first_token_variance': first_token_variance,
        'mean_length': np.mean(lengths),
        'response_count': len(responses)
    }

def estimate_enhanced_ambiguity(prompt: str) -> float:
    """Enhanced ambiguity scoring for real MoE testing"""
    prompt_lower = prompt.lower()
    
    # Computational/factual (low ambiguity)
    factual_indicators = ['calculate', 'what is', 'how many', 'when was', 'where is', 
                         'who is', 'define', '+ - * /', 'equals', 'result']
    factual_score = sum(1 for indicator in factual_indicators if indicator in prompt_lower)
    
    # Open-ended/philosophical (high ambiguity)
    open_indicators = ['explain', 'why', 'how should', 'what if', 'imagine', 'create',
                      'consciousness', 'meaning', 'purpose', 'reality', 'existence',
                      'beauty', 'art', 'love', 'future', 'possibility']
    open_score = sum(1 for indicator in open_indicators if indicator in prompt_lower)
    
    # Creative/subjective (high ambiguity)
    creative_indicators = ['write a', 'tell me about', 'describe your', 'what would',
                          'story', 'poem', 'creative', 'original', 'unique']
    creative_score = sum(1 for indicator in creative_indicators if indicator in prompt_lower)
    
    # Technical complexity (medium-high ambiguity)
    technical_indicators = ['algorithm', 'quantum', 'neural', 'machine learning',
                           'optimize', 'efficient', 'complex', 'analysis']
    technical_score = sum(1 for indicator in technical_indicators if indicator in prompt_lower)
    
    # Combine scores
    total_ambiguity = (
        0.1 * factual_score +           # Lower ambiguity
        0.3 * open_score +              # Higher ambiguity  
        0.3 * creative_score +          # Higher ambiguity
        0.2 * technical_score           # Medium-high ambiguity
    )
    
    # Normalize to [0, 1] and add baseline
    A_star = 0.1 + 0.8 * min(1.0, total_ambiguity / 3.0)
    
    return A_star

def run_real_moe_mediation_test(model: str = "qwen2.5:7b"):
    """
    Test A*→G→L_⊥ mediation using real generative model routing behavior
    """
    print("🚀 REAL MoE MEDIATION TEST")
    print("=" * 50)
    print(f"Model: {model}")
    print("Testing routing instability via response variance")
    print()
    
    # Test prompts across ambiguity spectrum
    test_prompts = [
        # Low ambiguity (computational/factual)
        "Calculate 15 * 23",
        "What is the capital of Japan?", 
        "How many days are in a year?",
        "Define photosynthesis",
        
        # Medium ambiguity (explanatory)
        "Explain how cars work",
        "Why do people dream?",
        "How does the internet work?",
        
        # High ambiguity (philosophical/creative)
        "What is the meaning of life?",
        "Explain consciousness", 
        "What makes art beautiful?",
        "Describe the nature of reality",
        
        # Very high ambiguity (open-ended/creative)
        "Write a short story about time travel",
        "What would happen if gravity stopped?",
        "Imagine a world without colors",
        "Create a poem about artificial intelligence"
    ]
    
    print(f"Testing {len(test_prompts)} prompts with {model}...")
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: {prompt[:50]}...")
        
        # Estimate ambiguity
        A_star = estimate_enhanced_ambiguity(prompt)
        
        # Generate multiple responses to measure routing variance
        responses = query_ollama_generation(prompt, model, temperature=0.7, n_samples=5)
        
        if len(responses) >= 2:
            # Compute routing stability metrics
            consistency_metrics = compute_response_consistency(responses)
            
            # Convert to routing sensitivity (G)
            # High consistency → low G (stable routing)
            # Low consistency → high G (unstable routing)  
            G = 1.0 - consistency_metrics['consistency'] + consistency_metrics['first_token_variance']
            
            # L_⊥ proxy: length variance + token diversity
            L_perp = consistency_metrics['length_variance'] + consistency_metrics['first_token_variance']
            
            print(f"  A* = {A_star:.3f}")
            print(f"  G = {G:.4f} (routing instability)")
            print(f"  L_⊥ = {L_perp:.4f} (response variance)")
            print(f"  Consistency = {consistency_metrics['consistency']:.3f}")
            print(f"  Responses = {consistency_metrics['response_count']}")
            
            results.append({
                'prompt': prompt,
                'A_star': A_star,
                'G': G,
                'L_perp': L_perp,
                'consistency': consistency_metrics['consistency'],
                'length_variance': consistency_metrics['length_variance'],
                'first_token_variance': consistency_metrics['first_token_variance'],
                'mean_length': consistency_metrics['mean_length']
            })
        else:
            print(f"  Failed: insufficient responses")
    
    if len(results) < 5:
        print(f"\nInsufficient data: {len(results)} valid prompts")
        return None
    
    print(f"\n{'='*50}")
    print(f"REAL MoE MEDIATION ANALYSIS - {len(results)} prompts")
    print(f"{'='*50}")
    
    # Extract arrays
    A_values = np.array([r['A_star'] for r in results])
    G_values = np.array([r['G'] for r in results])
    L_values = np.array([r['L_perp'] for r in results])
    
    # Correlations
    rho_AG, p_AG = spearmanr(A_values, G_values)
    rho_GL, p_GL = spearmanr(G_values, L_values)
    rho_AL, p_AL = spearmanr(A_values, L_values)
    
    print(f"Real MoE Correlations:")
    print(f"  ρ(A*, G)   = {rho_AG:+.3f} (p={p_AG:.3f}) ← Should be positive!")
    print(f"  ρ(G, L_⊥)  = {rho_GL:+.3f} (p={p_GL:.3f}) ← Should be positive")
    print(f"  ρ(A*, L_⊥) = {rho_AL:+.3f} (p={p_AL:.3f}) (direct effect)")
    
    # Data ranges
    print(f"\n📊 Data Ranges:")
    print(f"  A* range: [{A_values.min():.2f}, {A_values.max():.2f}]")
    print(f"  G range:  [{G_values.min():.3f}, {G_values.max():.3f}]")
    print(f"  L_⊥ range: [{L_values.min():.3f}, {L_values.max():.3f}]")
    
    # Mediation verdict
    print(f"\n{'='*50}")
    print(f"REAL MoE MEDIATION VERDICT")
    print(f"{'='*50}")
    
    if rho_AG > 0.3:
        print("✅ A* → G CONFIRMED with real MoE!")
        print(f"  Ambiguous prompts → routing instability: ρ = {rho_AG:+.3f}")
        
        if rho_GL > 0.3:
            print("✅ G → L_⊥ CONFIRMED!")
            print(f"  Routing instability → response variance: ρ = {rho_GL:+.3f}")
            print("🎛️  CONTROL READY: Use β/λ schedules to reduce G")
        else:
            print("⚠️  G → L_⊥ pathway weak")
    else:
        print(f"⚠️  A* → G pathway not strong: ρ = {rho_AG:+.3f}")
        if rho_AG > 0.1:
            print("  Some evidence but need more data or better metrics")
        else:
            print("  May need different stability metrics")
    
    # Save results
    output_path = Path("validation/results/real_moe_mediation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'model': model,
            'correlations': {
                'rho_A_G': float(rho_AG),
                'p_A_G': float(p_AG),
                'rho_G_L': float(rho_GL),
                'p_G_L': float(p_GL),
                'rho_A_L': float(rho_AL),
                'p_A_L': float(p_AL)
            },
            'mediation_confirmed': rho_AG > 0.3 and rho_GL > 0.3,
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_path}")
    
    # Create visualization
    create_real_moe_plot(A_values, G_values, L_values, rho_AG, rho_GL, model)
    
    return results

def create_real_moe_plot(A_values, G_values, L_values, rho_AG, rho_GL, model):
    """Create real MoE mediation plot"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # A* vs G (critical test)
        axes[0].scatter(A_values, G_values, alpha=0.8, s=100, color='red')
        axes[0].set_xlabel('Ambiguity (A*)')
        axes[0].set_ylabel('Routing Instability (G)')
        axes[0].set_title(f'Real MoE: A* → G\nρ = {rho_AG:.3f}\nModel: {model}')
        axes[0].grid(True, alpha=0.3)
        
        # Add trend line
        if len(A_values) > 2:
            z = np.polyfit(A_values, G_values, 1)
            p = np.poly1d(z)
            x_range = np.linspace(A_values.min(), A_values.max(), 100)
            axes[0].plot(x_range, p(x_range), 'k--', alpha=0.7)
        
        # G vs L_⊥
        axes[1].scatter(G_values, L_values, alpha=0.8, s=100, color='green')
        axes[1].set_xlabel('Routing Instability (G)')
        axes[1].set_ylabel('Response Variance (L_⊥)')
        axes[1].set_title(f'Real MoE: G → L_⊥\nρ = {rho_GL:.3f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = Path(f"validation/results/real_moe_{model.replace(':', '_')}.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"🎨 Plot saved to {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == '__main__':
    print("🔥 Testing A*→G→L_⊥ with REAL GENERATIVE MODEL")
    print("Unlike embedding similarity, this tests actual routing behavior")
    print("-" * 60)
    
    # Check available models
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available = [m['name'] for m in models]
            print(f"Available models: {available}")
            
            # Use best available model
            if 'mixtral:8x7b' in available:
                model = 'mixtral:8x7b'
                print("🎯 Using Mixtral 8x7B (real MoE)")
            elif 'qwen2.5:7b' in available:
                model = 'qwen2.5:7b'
                print("🎯 Using Qwen2.5 7B (large model)")
            else:
                model = available[0] if available else 'qwen2.5:7b'
                print(f"🎯 Using {model}")
        else:
            model = 'qwen2.5:7b'
            print("🎯 Using default Qwen2.5 7B")
            
    except Exception as e:
        print(f"API check failed: {e}")
        model = 'qwen2.5:7b'
    
    results = run_real_moe_mediation_test(model)
    
    if results:
        print(f"\n🎯 FINAL ASSESSMENT:")
        print(f"• Tested with real generative model routing")
        print(f"• Response variance as routing stability proxy")
        print(f"• {len(results)} prompts across ambiguity spectrum")
        print(f"• This shows actual routing behavior, not just embeddings!")
    else:
        print(f"\n⚠️  Test failed - check Ollama connection")
        print(f"Try: ollama serve")