#!/usr/bin/env python3
"""
Calculate optimal max_workers setting based on system configuration.
Provides recommendations for OLLAMA_NUM_PARALLEL and max_workers settings.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_gpu_info():
    """Get GPU memory information using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                return {
                    'name': parts[0],
                    'total_mb': float(parts[1]),
                    'used_mb': float(parts[2]),
                    'free_mb': float(parts[3])
                }
    except Exception as e:
        logger.debug(f"Could not get GPU info: {e}")
    
    return None


def estimate_model_vram(model_name: str) -> float:
    """
    Estimate VRAM usage for common model sizes (in MB).
    Based on q4_k_m quantization.
    """
    model_name = model_name.lower()
    
    # Model size patterns
    if '7b' in model_name or '8b' in model_name:
        return 4500  # ~4.5GB for 7-8B models
    elif '13b' in model_name:
        return 8000  # ~8GB for 13B models
    elif '3b' in model_name:
        return 2500  # ~2.5GB for 3B models
    elif '1.5b' in model_name or '1b' in model_name:
        return 1500  # ~1.5GB for small models
    else:
        return 5000  # Default conservative estimate


def estimate_context_vram(num_ctx: int = 2048) -> float:
    """
    Estimate VRAM per context buffer (KV cache) in MB.
    Roughly 0.5MB per 1000 tokens for 7B models.
    """
    return (num_ctx / 1000) * 500


def calculate_optimal_workers(gpu_vram_mb: float, model_size_mb: float, 
                              context_size_tokens: int = 2048,
                              safety_margin_mb: float = 1024) -> dict:
    """
    Calculate optimal number of parallel workers.
    
    Args:
        gpu_vram_mb: Total GPU VRAM in MB
        model_size_mb: Estimated model size in MB
        context_size_tokens: Context window size
        safety_margin_mb: Reserved VRAM for system/overhead
        
    Returns:
        dict: Recommendations for different scenarios
    """
    context_vram = estimate_context_vram(context_size_tokens)
    available_vram = gpu_vram_mb - model_size_mb - safety_margin_mb
    
    if available_vram <= 0:
        return {
            'error': 'Insufficient VRAM',
            'message': 'Model size exceeds available VRAM',
            'conservative': 1,
            'recommended': 1,
            'aggressive': 1
        }
    
    # Calculate different scenarios
    max_contexts = int(available_vram / context_vram)
    
    # Conservative: Leave plenty of room
    conservative = max(1, min(4, max_contexts // 2))
    
    # Recommended: Balanced approach
    recommended = max(2, min(6, int(max_contexts * 0.7)))
    
    # Aggressive: Push limits (may cause memory pressure)
    aggressive = max(2, min(8, max_contexts))
    
    return {
        'conservative': conservative,
        'recommended': recommended,
        'aggressive': aggressive,
        'max_theoretical': max_contexts,
        'vram_breakdown': {
            'total_mb': gpu_vram_mb,
            'model_mb': model_size_mb,
            'safety_margin_mb': safety_margin_mb,
            'available_for_contexts_mb': available_vram,
            'per_context_mb': context_vram
        }
    }


def print_recommendations(models: list = None):
    """Print recommendations for the current system."""
    
    print("=" * 70)
    print("  Optimal max_workers Calculator")
    print("=" * 70)
    print()
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    if gpu_info:
        print(f" GPU Information:")
        print(f"   • Name: {gpu_info['name']}")
        print(f"   • Total VRAM: {gpu_info['total_mb'] / 1024:.1f} GB")
        print(f"   • Used: {gpu_info['used_mb'] / 1024:.1f} GB")
        print(f"   • Free: {gpu_info['free_mb'] / 1024:.1f} GB")
        print()
        
        total_vram = gpu_info['total_mb']
    else:
        print("  Could not detect GPU. Using manual input.")
        try:
            total_vram = float(input("Enter your GPU VRAM in GB: ")) * 1024
        except:
            print("Invalid input. Using 16GB default.")
            total_vram = 16 * 1024
        print()
    
    # Default models if not provided
    if models is None:
        models = [
            'deepseek-r1:7b',
            'llama3.1:8b',
            'qwen2.5:7b',
            'bode3.1:8b'
        ]
    
    print(" Recommendations by Model:")
    print()
    
    for model in models:
        model_vram = estimate_model_vram(model)
        calc = calculate_optimal_workers(total_vram, model_vram)
        
        if 'error' in calc:
            print(f" {model}: {calc['message']}")
            continue
        
        print(f" {model}")
        print(f"   Model size: ~{model_vram / 1024:.1f} GB")
        print(f"   Available for parallelism: ~{calc['vram_breakdown']['available_for_contexts_mb'] / 1024:.1f} GB")
        print()
        print(f"   Recommendations:")
        print(f"   • Conservative (safest):  OLLAMA_NUM_PARALLEL={calc['conservative']}")
        print(f"   • Recommended (balanced): OLLAMA_NUM_PARALLEL={calc['recommended']} ⭐")
        print(f"   • Aggressive (maximum):   OLLAMA_NUM_PARALLEL={calc['aggressive']}")
        print(f"   • Theoretical max:        {calc['max_theoretical']} contexts")
        print()
    
    # Overall recommendations
    print("=" * 70)
    print(" Configuration Recommendations:")
    print("=" * 70)
    print()
    
    # Use the recommended value for the most common model
    typical_model_vram = estimate_model_vram('7b')
    calc = calculate_optimal_workers(total_vram, typical_model_vram)
    
    recommended_workers = calc['recommended']
    
    print(f"1  Update Ollama systemd service:")
    print(f"   Run: ./scripts/configure_parallel_ollama.sh")
    print(f"   Or manually set: OLLAMA_NUM_PARALLEL={recommended_workers}")
    print()
    
    print(f"2  Update experiment_settings.yaml:")
    print(f"   performance:")
    print(f"     max_workers_ollama: {recommended_workers}")
    print(f"     max_workers_cloud: 10")
    print()
    
    print(f"3  Monitor and adjust:")
    print(f"   • Run: watch -n 1 nvidia-smi")
    print(f"   • Check VRAM usage during training")
    print(f"   • If VRAM usage > 90%, reduce workers")
    print(f"   • If VRAM usage < 70%, can increase workers")
    print()
    
    print("=" * 70)
    print(" Tips:")
    print("=" * 70)
    print("• Start with recommended value and monitor")
    print("• DeepSeek-R1 uses more VRAM per request (long thinking chains)")
    print("• Multiple models loaded = less room for parallel contexts")
    print("• Python max_workers should match OLLAMA_NUM_PARALLEL")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate optimal max_workers for Ollama parallelism'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Model names to analyze (e.g., llama3.1:8b qwen2.5:7b)'
    )
    parser.add_argument(
        '--vram',
        type=float,
        help='GPU VRAM in GB (auto-detected if not specified)'
    )
    
    args = parser.parse_args()
    
    # Override GPU detection if manually specified
    if args.vram:
        # Monkey patch get_gpu_info
        original_get_gpu_info = get_gpu_info
        def mock_get_gpu_info():
            return {
                'name': 'User Specified',
                'total_mb': args.vram * 1024,
                'used_mb': 0,
                'free_mb': args.vram * 1024
            }
        get_gpu_info = mock_get_gpu_info
    
    print_recommendations(args.models)
