"""Ollama local setup, configuration, and monitoring tools."""

from .setup import (
    get_gpu_info,
    check_ollama_status,
    calculate_recommended_workers,
    display_current_config,
)
from .watch import OllamaMonitor

__all__ = [
    "get_gpu_info",
    "check_ollama_status",
    "calculate_recommended_workers",
    "display_current_config",
    "OllamaMonitor",
]
