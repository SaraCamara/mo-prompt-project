# MAX_WORKERS Configuration Update

## Summary

Made `MAX_WORKERS` safer and more configurable by:
1.  Moving configuration to `experiment_settings.yaml`
2.  Adding intelligent validation and warnings
3.  Creating a calculator to determine optimal settings
4.  Adding VRAM-based safety limits

## Changes Made

### 1. Configuration File (`config/experiment_settings.yaml`)

Added new `performance` section:

```yaml
# Performance and Parallelism Configuration
performance:
  # Maximum concurrent workers for parallel evaluation
  max_workers_ollama: 6      # For local Ollama instances
  max_workers_cloud: 10      # For cloud APIs (OpenAI, Maritalk, etc.)
  
  # Safety limits - automatic validation
  max_workers_limit: 20      # Hard limit to prevent resource exhaustion
  warn_above_workers: 8      # Warn if workers exceed this for Ollama
```

### 2. Code Changes (`scripts/prompt_evaluator.py`)

**Added `get_safe_max_workers()` function:**
- Reads configuration from `experiment_settings.yaml`
- Validates against hard limits
- Issues warnings for potentially problematic configurations
- Logs decisions for debugging

**Updated both evaluation functions:**
- `evaluate_prompt_squad()` - Line 138
- `evaluate_prompt_imdb()` - Line 186

**Before:**
```python
evaluator_type = evaluator_config.get("tipo", "").lower()
MAX_WORKERS = 4 if evaluator_type == "ollama" else 10
```

**After:**
```python
# Get safe max_workers from configuration with validation
MAX_WORKERS = get_safe_max_workers(executor_config, experiment_settings)
```

### 3. New Tools

**`scripts/calculate_optimal_workers.py`**
- Detects GPU VRAM automatically
- Calculates optimal parallelism based on model sizes
- Provides conservative, recommended, and aggressive settings
- Generates personalized configuration recommendations

**Usage:**
```bash
source .venv/bin/activate
python scripts/calculate_optimal_workers.py

# Or with custom VRAM:
python scripts/calculate_optimal_workers.py --vram 16

# Or with specific models:
python scripts/calculate_optimal_workers.py --models llama3.1:8b qwen2.5:7b
```

### 4. Documentation Updates

- Updated [OLLAMA_PARALLEL_QUICKSTART.md](../OLLAMA_PARALLEL_QUICKSTART.md) with venv usage
- Updated [docs/PARALLEL_OLLAMA_OPTIMIZATION.md](../docs/PARALLEL_OLLAMA_OPTIMIZATION.md) with calculator instructions

## Current Recommendations (Based on Your RTX 5060 Ti)

Your GPU analysis shows:
- **Total VRAM**: 15.9 GB
- **Currently used**: 6.7 GB  
- **Free**: 8.9 GB

### Optimal Configuration

```yaml
# config/experiment_settings.yaml
performance:
  max_workers_ollama: 6      # ⭐ Recommended for your hardware
  max_workers_cloud: 10
  max_workers_limit: 20
  warn_above_workers: 8
```

This gives you **~3x speedup** compared to sequential processing!

## Validation Features

### Automatic Checks
1. **Hard Limit**: Prevents setting > 20 workers (resource exhaustion protection)
2. **Warning Threshold**: Warns if Ollama workers > 8 (memory pressure)
3. **Configuration Mismatch**: Warns if Python workers > OLLAMA_NUM_PARALLEL
4. **Type Detection**: Automatically uses different limits for Ollama vs cloud APIs

### Example Warnings

If you set `max_workers_ollama: 12`, you'll see:
```
WARNING: High max_workers (12) configured for Ollama. 
This may cause memory issues or timeouts. 
Recommended: 8 or lower for typical 7B models on 16GB VRAM. 
Ensure OLLAMA_NUM_PARALLEL is set to at least 12 on the server.
```

## How to Use

1. **Calculate optimal settings:**
   ```bash
   source .venv/bin/activate
   python scripts/calculate_optimal_workers.py
   ```

2. **Update configuration:**
   Edit `config/experiment_settings.yaml` with recommended values

3. **Configure Ollama server:**
   ```bash
   ./scripts/configure_parallel_ollama.sh
   ```

4. **Run experiments:**
   ```bash
   source .venv/bin/activate
   python scripts/main.py
   ```

## Benefits

 **Safer**: Hard limits prevent OOM crashes
 **Configurable**: Easy to adjust per environment
 **Intelligent**: Automatic warnings for risky configurations
 **Documented**: Clear logging of decisions
 **Validated**: Calculator provides evidence-based recommendations

## Migration Guide

No manual migration needed! The changes are backward compatible:
- If `performance` section is missing, defaults to old behavior (4 for Ollama, 10 for cloud)
- Existing experiments will continue to work
- Can gradually adopt new configuration

## Testing

The calculator confirmed optimal settings for your hardware:
- **Conservative**: 4 workers (safest, ~2.5x speedup)
- **Recommended**: 6 workers (balanced, ~3.3x speedup) ⭐
- **Aggressive**: 8 workers (maximum, ~4x speedup, may cause memory pressure)

Your current [config/experiment_settings.yaml](../config/experiment_settings.yaml) is already set to `max_workers_ollama: 6` - the optimal value!
