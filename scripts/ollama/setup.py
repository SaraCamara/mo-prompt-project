#!/usr/bin/env python3
"""
Setup utility for mo-prompt-project parallelism and Ollama configuration.

Usage:
    python -m scripts.setup              # Interactive setup
    python -m scripts.setup --check      # Check current configuration
    python -m scripts.setup --auto       # Auto-configure based on GPU
    python -m scripts.setup --workers 6  # Set specific worker count
"""

import os
import sys
import subprocess
import argparse
import logging
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_gpu_info():
    """Get GPU memory information."""
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


def check_ollama_status():
    """Check if Ollama is running and get configuration."""
    try:
        # Check if Ollama process is running
        result = subprocess.run(
            ['pgrep', '-f', 'ollama serve'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {'running': False}
        
        pid = result.stdout.strip().split('\n')[0]
        
        # Get environment variables from systemd service (more reliable than /proc)
        env_vars = {}
        try:
            # Method 1: Use systemctl show
            service_result = subprocess.run(
                ['systemctl', 'show', 'ollama', '--property=Environment'],
                capture_output=True,
                text=True
            )
            if service_result.returncode == 0 and service_result.stdout:
                env_line = service_result.stdout.strip()
                if env_line.startswith('Environment='):
                    env_line = env_line[12:]  # Remove 'Environment=' prefix
                    # Parse space-separated KEY=value pairs
                    import shlex
                    try:
                        for pair in shlex.split(env_line):
                            if '=' in pair:
                                key, _, value = pair.partition('=')
                                if key.startswith('OLLAMA_'):
                                    env_vars[key] = value
                    except:
                        pass
            
            # Method 2: Fallback to reading service file directly
            if not env_vars:
                try:
                    with open('/etc/systemd/system/ollama.service', 'r') as f:
                        for line in f:
                            if line.strip().startswith('Environment="OLLAMA_'):
                                # Parse: Environment="OLLAMA_NUM_PARALLEL=6"
                                line = line.strip().replace('Environment="', '').rstrip('"')
                                if '=' in line:
                                    key, _, value = line.partition('=')
                                    env_vars[key] = value
                except:
                    pass
        except:
            pass
        
        # Test API connectivity
        api_accessible = False
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            api_accessible = response.status_code == 200
        except:
            pass
        
        return {
            'running': True,
            'pid': pid,
            'env_vars': env_vars,
            'api_accessible': api_accessible
        }
    except Exception as e:
        logger.debug(f"Error checking Ollama: {e}")
        return {'running': False}


def calculate_recommended_workers(gpu_vram_mb, model_size_mb=4500):
    """Calculate recommended worker count based on available VRAM."""
    context_vram = 1000  # ~1GB per context (conservative)
    safety_margin = 1024
    
    available_vram = gpu_vram_mb - model_size_mb - safety_margin
    
    if available_vram <= 0:
        return 1
    
    max_contexts = int(available_vram / context_vram)
    return max(2, min(6, int(max_contexts * 0.7)))


def update_experiment_settings(max_workers_ollama, max_workers_cloud=10):
    """Update experiment_settings.yaml with new worker counts."""
    project_root = get_project_root()
    settings_path = project_root / 'config' / 'experiment_settings.yaml'
    
    if not settings_path.exists():
        logger.error(f"Settings file not found: {settings_path}")
        return False
    
    try:
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
        
        # Update or create performance section
        if 'performance' not in settings:
            settings['performance'] = {}
        
        settings['performance']['max_workers_ollama'] = max_workers_ollama
        settings['performance']['max_workers_cloud'] = max_workers_cloud
        
        # Ensure other performance keys exist with defaults
        settings['performance'].setdefault('max_workers_limit', 20)
        settings['performance'].setdefault('warn_above_workers', 8)
        
        with open(settings_path, 'w') as f:
            yaml.safe_dump(settings, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f" Updated {settings_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        return False


def display_current_config():
    """Display current configuration."""
    print("\n" + "="*70)
    print("  Current Configuration")
    print("="*70 + "\n")
    
    # GPU Info
    gpu_info = get_gpu_info()
    if gpu_info:
        print(" GPU:")
        print(f"   • {gpu_info['name']}")
        print(f"   • VRAM: {gpu_info['total_mb']/1024:.1f}GB total, "
              f"{gpu_info['free_mb']/1024:.1f}GB free")
    else:
        print("  GPU: Could not detect")
    
    print()
    
    # Ollama Status
    ollama = check_ollama_status()
    if ollama['running']:
        print(" Ollama:")
        print(f"   • Status: Running (PID {ollama['pid']})")
        print(f"   • API: {'Accessible' if ollama['api_accessible'] else 'Not accessible'}")
        
        if ollama['env_vars']:
            print("   • Configuration:")
            for key, value in sorted(ollama['env_vars'].items()):
                print(f"     - {key}={value}")
        else:
            print("     No OLLAMA_* environment variables detected")
            print("      (Parallelism may not be configured)")
    else:
        print(" Ollama: Not running")
    
    print()
    
    # Python Settings
    project_root = get_project_root()
    settings_path = project_root / 'config' / 'experiment_settings.yaml'
    
    if settings_path.exists():
        try:
            # Read without resolving placeholders
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f)
            
            perf = settings.get('performance', {})
            print("  Python Configuration:")
            print(f"   • max_workers_ollama: {perf.get('max_workers_ollama', 'NOT SET')}")
            print(f"   • max_workers_cloud: {perf.get('max_workers_cloud', 'NOT SET')}")
            print(f"   • max_workers_limit: {perf.get('max_workers_limit', 'NOT SET')}")
        except Exception as e:
            print(f"  Python Configuration: Could not read ({e})")
    else:
        print(" Python Configuration: experiment_settings.yaml not found")
    
    print()


def setup_interactive():
    """Interactive setup wizard."""
    print("\n" + "="*70)
    print("   Ollama Parallelism Setup Wizard")
    print("="*70 + "\n")
    
    # Check GPU
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("  Could not detect GPU automatically.")
        vram_gb = float(input("Enter your GPU VRAM in GB (e.g., 16): ") or "16")
        gpu_vram_mb = vram_gb * 1024
    else:
        print(f" Detected: {gpu_info['name']}")
        print(f"   VRAM: {gpu_info['total_mb']/1024:.1f}GB total, "
              f"{gpu_info['free_mb']/1024:.1f}GB free\n")
        gpu_vram_mb = gpu_info['total_mb']
    
    # Calculate recommendation
    recommended = calculate_recommended_workers(gpu_vram_mb)
    
    print(f" Recommended max_workers_ollama: {recommended}")
    print(f"   (Based on ~4.5GB model size + context buffers)\n")
    
    # Get user choice
    choice = input(f"Enter max_workers_ollama [{recommended}]: ").strip()
    max_workers = int(choice) if choice else recommended
    
    if max_workers > 8:
        print(f"\n  Warning: {max_workers} workers is quite high.")
        print("   This may cause memory pressure on 16GB GPUs.")
        confirm = input("   Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return False
    
    # Update Python config
    print(f"\n Updating experiment_settings.yaml...")
    if not update_experiment_settings(max_workers):
        return False
    
    # Check Ollama
    print("\n Checking Ollama configuration...")
    ollama = check_ollama_status()
    
    if not ollama['running']:
        print(" Ollama is not running.")
        print("\nTo start Ollama with parallelism, run:")
        print("   ./scripts/configure_parallel_ollama.sh")
        return True
    
    ollama_parallel = ollama['env_vars'].get('OLLAMA_NUM_PARALLEL')
    
    if ollama_parallel:
        print(f" Ollama is configured with OLLAMA_NUM_PARALLEL={ollama_parallel}")
        
        if int(ollama_parallel) < max_workers:
            print(f"\n  Note: Your Python max_workers ({max_workers}) > "
                  f"OLLAMA_NUM_PARALLEL ({ollama_parallel})")
            print("   Requests will queue on the Ollama side.")
            print("   Consider running: ./scripts/configure_parallel_ollama.sh")
    else:
        print("  Ollama is running but OLLAMA_NUM_PARALLEL is not set.")
        print("   Running in sequential mode (1 request at a time).")
        print("\n To enable parallelism, run:")
        print("   ./scripts/configure_parallel_ollama.sh")
    
    print("\n" + "="*70)
    print("   Setup Complete!")
    print("="*70)
    print("\nYou can now run experiments:")
    print("  python scripts/main.py")
    print("\nOr check status anytime:")
    print("  python -m scripts.setup --check")
    print()
    
    return True


def setup_auto():
    """Automatic setup based on GPU detection."""
    print("\n Auto-configuring based on GPU detection...\n")
    
    gpu_info = get_gpu_info()
    if not gpu_info:
        print(" Could not detect GPU. Use --workers to set manually.")
        return False
    
    gpu_vram_mb = gpu_info['total_mb']
    recommended = calculate_recommended_workers(gpu_vram_mb)
    
    print(f"Detected: {gpu_info['name']} ({gpu_vram_mb/1024:.1f}GB)")
    print(f"Recommended workers: {recommended}\n")
    
    return update_experiment_settings(recommended)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Setup utility for Ollama parallelism configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.setup              # Interactive setup
  python -m scripts.setup --check      # Check current configuration
  python -m scripts.setup --auto       # Auto-configure
  python -m scripts.setup --workers 6  # Set specific worker count
        """
    )
    
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check current configuration without making changes'
    )
    parser.add_argument(
        '--auto', '-a',
        action='store_true',
        help='Automatically configure based on GPU detection'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        help='Set specific max_workers_ollama value'
    )
    parser.add_argument(
        '--cloud-workers',
        type=int,
        default=10,
        help='Set max_workers_cloud value (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Check mode
    if args.check:
        display_current_config()
        return 0
    
    # Auto mode
    if args.auto:
        success = setup_auto()
        return 0 if success else 1
    
    # Manual worker count
    if args.workers:
        print(f"\n Setting max_workers_ollama={args.workers}...\n")
        success = update_experiment_settings(args.workers, args.cloud_workers)
        if success:
            print("\n Configuration updated!")
            print("   Check with: python -m scripts.setup --check")
        return 0 if success else 1
    
    # Interactive mode (default)
    try:
        success = setup_interactive()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
