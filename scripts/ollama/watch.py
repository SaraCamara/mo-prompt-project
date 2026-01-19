#!/usr/bin/env python3
"""
Live monitoring dashboard for Ollama models and resource usage.

Usage:
    python -m scripts.watch           # Default 5s refresh
    python -m scripts.watch --fast    # 2s refresh
    python -m scripts.watch --slow    # 10s refresh
"""

import subprocess
import time
import sys
import os
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)


class OllamaMonitor:
    """Monitor Ollama processes and resource usage."""
    
    def __init__(self):
        self.ollama_api = "http://localhost:11434"
        
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def get_gpu_info(self):
        """Get GPU information from nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            gpus.append({
                                'index': parts[0],
                                'name': parts[1],
                                'mem_used_mb': float(parts[2]),
                                'mem_total_mb': float(parts[3]),
                                'gpu_util': float(parts[4]),
                                'temp': float(parts[5])
                            })
                return gpus
        except Exception:
            pass
        return None
    
    def get_ollama_processes(self):
        """Get Ollama processes with CPU and memory usage."""
        try:
            # Get all ollama processes
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            processes = []
            for line in result.stdout.split('\n'):
                if 'ollama' in line.lower() and 'grep' not in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        # Extract model info from command line if it's a runner
                        model_info = "main server"
                        if 'runner' in line:
                            # Try to extract model name from blob path
                            if '--model' in line:
                                try:
                                    idx = parts.index('--model')
                                    if idx + 1 < len(parts):
                                        blob_path = parts[idx + 1]
                                        model_info = f"runner ({blob_path[-12:]}...)"
                                except:
                                    model_info = "runner"
                        
                        processes.append({
                            'user': parts[0],
                            'pid': parts[1],
                            'cpu': float(parts[2]),
                            'mem': float(parts[3]),
                            'vsz': parts[4],
                            'rss': parts[5],
                            'stat': parts[7],
                            'start': parts[8],
                            'time': parts[9],
                            'command': ' '.join(parts[10:12]),
                            'model_info': model_info
                        })
            
            return processes
        except Exception as e:
            return []
    
    def get_ollama_models(self):
        """Get list of loaded models from Ollama API."""
        try:
            response = requests.get(f"{self.ollama_api}/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
        except:
            pass
        return []
    
    def get_ollama_status(self):
        """Get Ollama server status."""
        try:
            # Check main process
            result = subprocess.run(
                ['pgrep', '-f', 'ollama serve'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {'running': False}
            
            pid = result.stdout.strip().split('\n')[0]
            
            # Get environment variables from systemd service file (more reliable)
            env_vars = {}
            try:
                # Try reading systemd service file first
                service_result = subprocess.run(
                    ['systemctl', 'show', 'ollama', '--property=Environment'],
                    capture_output=True,
                    text=True
                )
                if service_result.returncode == 0 and service_result.stdout:
                    # Parse Environment=VAR1=val1 VAR2=val2 format
                    env_line = service_result.stdout.strip()
                    if env_line.startswith('Environment='):
                        env_line = env_line[12:]  # Remove 'Environment=' prefix
                        # Split by space but handle quoted values
                        import shlex
                        try:
                            for pair in shlex.split(env_line):
                                if '=' in pair:
                                    key, _, value = pair.partition('=')
                                    if key.startswith('OLLAMA_'):
                                        env_vars[key] = value
                        except:
                            pass
                
                # Fallback: try reading from service file directly
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
            
            return {
                'running': True,
                'pid': pid,
                'env_vars': env_vars
            }
        except:
            return {'running': False}
    
    def format_bytes(self, mb):
        """Format bytes to human readable."""
        if mb < 1024:
            return f"{mb:.0f}M"
        else:
            return f"{mb/1024:.1f}G"
    
    def format_percent_bar(self, percent, width=20):
        """Create a visual percentage bar."""
        filled = int(width * percent / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"{bar} {percent:5.1f}%"
    
    def display_dashboard(self):
        """Display the monitoring dashboard."""
        self.clear_screen()
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 80)
        print(f"   Ollama Resource Monitor - {now}")
        print("=" * 80)
        print()
        
        # GPU Information
        gpus = self.get_gpu_info()
        if gpus:
            print(" GPU Status:")
            for gpu in gpus:
                mem_percent = (gpu['mem_used_mb'] / gpu['mem_total_mb']) * 100
                print(f"   GPU {gpu['index']}: {gpu['name']}")
                print(f"   │ Memory:  {self.format_percent_bar(mem_percent, 25)} "
                      f"({self.format_bytes(gpu['mem_used_mb'])} / {self.format_bytes(gpu['mem_total_mb'])})")
                print(f"   │ GPU:     {self.format_percent_bar(gpu['gpu_util'], 25)}")
                print(f"   └ Temp:    {gpu['temp']:.0f}°C")
        else:
            print("  GPU: Could not detect")
        
        print()
        
        # Ollama Status
        ollama_status = self.get_ollama_status()
        if ollama_status['running']:
            print(f" Ollama Server: Running (PID {ollama_status['pid']})")
            
            env_vars = ollama_status.get('env_vars', {})
            if env_vars:
                parallel = env_vars.get('OLLAMA_NUM_PARALLEL', 'Not set')
                keep_alive = env_vars.get('OLLAMA_KEEP_ALIVE', 'Not set')
                max_models = env_vars.get('OLLAMA_MAX_LOADED_MODELS', 'Not set')
                
                print(f"   │ Parallel:     {parallel}")
                print(f"   │ Keep-Alive:   {keep_alive}")
                print(f"   └ Max Models:  {max_models}")
            else:
                print("   └   No parallel configuration detected")
        else:
            print(" Ollama Server: Not running")
        
        print()
        
        # Loaded Models
        models = self.get_ollama_models()
        if models:
            print(f" Loaded Models: ({len(models)})")
            for model in models[:5]:  # Show first 5
                name = model.get('name', 'unknown')
                size = model.get('size', 0)
                size_gb = size / (1024**3)
                modified = model.get('modified_at', '')[:10] if model.get('modified_at') else 'unknown'
                print(f"   • {name:<30} {size_gb:>6.1f}GB  (modified: {modified})")
            if len(models) > 5:
                print(f"   ... and {len(models) - 5} more")
        else:
            print(" Loaded Models: None detected")
        
        print()
        
        # Process Information
        processes = self.get_ollama_processes()
        if processes:
            print(f"  Ollama Processes: ({len(processes)})")
            print(f"   {'PID':<8} {'CPU%':<8} {'MEM%':<8} {'TIME':<10} {'TYPE':<35}")
            print(f"   {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*35}")
            
            total_cpu = 0
            total_mem = 0
            
            for proc in processes:
                cpu_bar = '█' * int(proc['cpu'] / 10) + '░' * (10 - int(proc['cpu'] / 10))
                mem_bar = '█' * int(proc['mem'] / 10) + '░' * (10 - int(proc['mem'] / 10))
                
                print(f"   {proc['pid']:<8} "
                      f"{proc['cpu']:>5.1f}%  "
                      f"{proc['mem']:>5.1f}%  "
                      f"{proc['time']:<10} "
                      f"{proc['model_info']:<35}")
                
                total_cpu += proc['cpu']
                total_mem += proc['mem']
            
            print(f"   {'-'*8} {'-'*8} {'-'*8}")
            print(f"   {'TOTAL':<8} {total_cpu:>5.1f}%  {total_mem:>5.1f}%")
        else:
            print("  Ollama Processes: None found")
        
        print()
        print("=" * 80)
        print("Press Ctrl+C to exit")
    
    def run(self, refresh_interval=5):
        """Run the monitoring loop."""
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            self.clear_screen()
            print("\n Monitoring stopped")
            sys.exit(0)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Live monitoring dashboard for Ollama models and resources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.watch              # Default 5s refresh
  python -m scripts.watch --fast       # 2s refresh
  python -m scripts.watch --slow       # 10s refresh
  python -m scripts.watch --interval 3 # Custom 3s refresh
        """
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast refresh (2 seconds)'
    )
    parser.add_argument(
        '--slow',
        action='store_true',
        help='Slow refresh (10 seconds)'
    )
    parser.add_argument(
        '--interval', '-i',
        type=int,
        help='Custom refresh interval in seconds'
    )
    
    args = parser.parse_args()
    
    # Determine refresh interval
    if args.interval:
        interval = args.interval
    elif args.fast:
        interval = 2
    elif args.slow:
        interval = 10
    else:
        interval = 5
    
    print(f"Starting Ollama monitor (refresh every {interval}s)...")
    print("Press Ctrl+C to exit\n")
    time.sleep(1)
    
    monitor = OllamaMonitor()
    monitor.run(interval)


if __name__ == '__main__':
    main()
