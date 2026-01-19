"""Execution state tracking for resumable evolution runs."""
import os
import json
import csv
import logging
import datetime
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ExecutionTracker:
    """Tracks execution state and timing for resumable evolution runs.
    
    Maintains persistent state across runs, enabling automatic resume
    and accurate timing even after interruptions.
    """
    
    def __init__(self, base_output_dir: str):
        """Initialize tracker with output directory.
        
        Args:
            base_output_dir: Directory where tracking files will be stored
        """
        self.base_output_dir = base_output_dir
        self.metadata_path = os.path.join(base_output_dir, "execution_metadata.json")
        self.timing_log_path = os.path.join(base_output_dir, "timing_log.csv")
        self.metadata = {}
        self.generation_times = []
        
    def initialize(self, config: Dict[str, Any], start_time: float, 
                   start_datetime: str, resuming: bool = False) -> None:
        """Initialize or load execution metadata.
        
        Args:
            config: Experiment configuration dictionary
            start_time: Unix timestamp of execution start
            start_datetime: Formatted datetime string
            resuming: Whether this is resuming a previous run
        """
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        if resuming and os.path.exists(self.metadata_path):
            # Load existing metadata
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded existing metadata from {self.metadata_path}")
                
                # Restore generation times
                self.generation_times = self.metadata.get("generation_times", [])
                
            except Exception as e:
                logger.error(f"Error loading metadata: {e}. Starting fresh.")
                self.metadata = {}
        
        # Create or update metadata
        if not resuming or not self.metadata:
            # New run
            evaluator = config.get("evaluators", [{}])[0]
            strategy = config.get("strategies", [{}])[0]
            
            self.metadata = {
                "task": config.get("task", "unknown"),
                "model_name": evaluator.get("name", "unknown"),
                "model_id": evaluator.get("model", "unknown"),
                "strategy": strategy.get("name", "unknown"),
                "objective": config.get("objective", "unknown"),
                "start_time": start_datetime,
                "start_timestamp": start_time,
                "last_update_time": start_datetime,
                "last_update_timestamp": start_time,
                "last_completed_generation": -1,
                "total_generations_planned": config.get("evolution_params", {}).get("max_generations", 10),
                "status": "running",
                "stagnation_counter": 0,
                "last_front_hash": None,
                "evolution_params": config.get("evolution_params", {}),
                "accumulated_time_seconds": 0.0,
                "generation_times": [],
                "base_output_dir": self.base_output_dir
            }
            self._save_metadata()
            self._initialize_timing_log()
        else:
            # Resuming - update start time for this session
            self.metadata["last_resume_time"] = start_datetime
            self.metadata["last_resume_timestamp"] = start_time
            self.metadata["status"] = "running"
            self._save_metadata()
            
    def update_generation(self, generation: int, duration: float, 
                         metrics: Optional[Dict[str, Any]] = None,
                         stagnation_counter: int = 0,
                         front_hash: Optional[int] = None) -> None:
        """Update tracking after a generation completes.
        
        Args:
            generation: Generation number that just completed
            duration: Duration of this generation in seconds
            metrics: Optional metrics dictionary for this generation
            stagnation_counter: Current stagnation counter value
            front_hash: Hash of Pareto front (multi-objective only)
        """
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        
        # Update timing
        self.generation_times.append(duration)
        self.metadata["generation_times"] = self.generation_times
        self.metadata["accumulated_time_seconds"] += duration
        
        # Update generation info
        self.metadata["last_completed_generation"] = generation
        self.metadata["last_update_time"] = now_str
        self.metadata["last_update_timestamp"] = time.time()
        self.metadata["stagnation_counter"] = stagnation_counter
        
        if front_hash is not None:
            self.metadata["last_front_hash"] = front_hash
        
        # Add generation-specific metrics if provided
        if metrics:
            gen_key = f"generation_{generation}_metrics"
            self.metadata[gen_key] = metrics
        
        # Save to files
        self._save_metadata()
        self._append_timing_log(generation, duration, now_str)
        
        logger.info(f"Generation {generation} tracked: {duration:.2f}s")
        
    def mark_completed(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Mark execution as successfully completed.
        
        Args:
            final_metrics: Optional final metrics to store
        """
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metadata["status"] = "completed"
        self.metadata["end_time"] = now_str
        self.metadata["end_timestamp"] = time.time()
        
        if final_metrics:
            self.metadata["final_metrics"] = final_metrics
        
        self._save_metadata()
        logger.info("Execution marked as completed")
        
    def mark_stopped(self, reason: str = "interrupted") -> None:
        """Mark execution as stopped/interrupted.
        
        Args:
            reason: Reason for stopping (e.g., "error", "interrupted", "stagnation")
        """
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metadata["status"] = "stopped"
        self.metadata["stop_time"] = now_str
        self.metadata["stop_timestamp"] = time.time()
        self.metadata["stop_reason"] = reason
        
        self._save_metadata()
        logger.info(f"Execution marked as stopped: {reason}")
        
    def get_resumable_state(self) -> Optional[Dict[str, Any]]:
        """Check if execution can be resumed and return state.
        
        Returns:
            Dictionary with resumable state info, or None if can't resume
        """
        if not os.path.exists(self.metadata_path):
            return None
        
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            status = metadata.get("status")
            last_gen = metadata.get("last_completed_generation", -1)
            
            # Can resume if running/stopped and at least one generation completed
            if status in ["running", "stopped"] and last_gen >= 0:
                return {
                    "last_completed_generation": last_gen,
                    "next_generation": last_gen + 1,
                    "accumulated_time": metadata.get("accumulated_time_seconds", 0),
                    "generation_times": metadata.get("generation_times", []),
                    "stagnation_counter": metadata.get("stagnation_counter", 0),
                    "last_front_hash": metadata.get("last_front_hash"),
                    "start_time": metadata.get("start_time"),
                    "stop_reason": metadata.get("stop_reason", "unknown")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error reading metadata for resume: {e}")
            return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get current metadata dictionary.
        
        Returns:
            Current metadata
        """
        return self.metadata.copy()
    
    def get_generation_times(self) -> List[float]:
        """Get list of generation times.
        
        Returns:
            List of generation durations in seconds
        """
        return self.generation_times.copy()
    
    def _save_metadata(self) -> None:
        """Save metadata to JSON file atomically."""
        tmp_path = self.metadata_path + ".tmp"
        try:
            with open(tmp_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Atomic rename
            os.replace(tmp_path, self.metadata_path)
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _initialize_timing_log(self) -> None:
        """Initialize timing log CSV file."""
        if not os.path.exists(self.timing_log_path):
            try:
                with open(self.timing_log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "generation", "start_time", "end_time", 
                        "duration_seconds", "cumulative_time_seconds"
                    ])
            except Exception as e:
                logger.error(f"Error initializing timing log: {e}")
    
    def _append_timing_log(self, generation: int, duration: float, end_time: str) -> None:
        """Append timing entry to CSV log.
        
        Args:
            generation: Generation number
            duration: Duration in seconds
            end_time: End time formatted string
        """
        try:
            # Calculate start time for this generation
            cumulative = sum(self.generation_times)
            
            start_timestamp = self.metadata["start_timestamp"] + (cumulative - duration)
            start_time = datetime.datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            with open(self.timing_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    generation, start_time, end_time, 
                    f"{duration:.2f}", f"{cumulative:.2f}"
                ])
                
        except Exception as e:
            logger.error(f"Error appending to timing log: {e}")


def detect_resumable_run(base_output_dir: str) -> Optional[Dict[str, Any]]:
    """Detect if there's a resumable run in the output directory.
    
    Args:
        base_output_dir: Directory to check for resumable state
    
    Returns:
        Dictionary with resumable state, or None if no resumable run found
    """
    tracker = ExecutionTracker(base_output_dir)
    return tracker.get_resumable_state()
