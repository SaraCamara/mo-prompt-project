"""
Utilities for mo-prompt-project.
Can be run as:
  python -m scripts.main       - Main entry point for evolution
  python -m scripts.ollama.setup  - Setup parallelism and configuration
  python -m scripts.ollama.watch  - Monitor Ollama resources in real-time

Handles:
- GPU detection and validation
- Ollama configuration verification
- Parallelism setup recommendations
- One-time environment configuration
- Real-time resource monitoring
"""

import sys

def main():
    """Route to appropriate utility based on first argument."""
    from .main import main as main_main
    main_main()

if __name__ == "__main__":
    main()
