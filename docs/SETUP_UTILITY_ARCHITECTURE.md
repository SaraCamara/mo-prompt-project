# Setup Utility - Architecture Improvements

## Problem Solved

Previously, parallelism configuration was mixed with experiment execution. This meant:
-  Had to reconfigure when switching models
-  Hard to validate if setup was correct
-  No clear separation between "setup" and "run"
-  Difficult to run multiple experiments in sequence

## New Architecture

```
┌─────────────────────────────────────────────────────┐
│  SETUP PHASE (Once)                                 │
│  python -m scripts.setup                            │
│                                                      │
│  • Detect GPU capacity                              │
│  • Calculate optimal parallelism                    │
│  • Update experiment_settings.yaml                  │
│  • Validate Ollama configuration                    │
│  • Provide recommendations                          │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  EXPERIMENT PHASE (Many times)                      │
│  python scripts/main.py                             │
│                                                      │
│  • Uses configured parallelism                      │
│  • Switch models freely                             │
│  • Change strategies (zero/few-shot)                │
│  • Run multiple experiments                         │
│  • No reconfiguration needed                        │
└─────────────────────────────────────────────────────┘
```

## What Was Created

### 1. Setup Module (`scripts/setup.py`)

A comprehensive setup utility that can be run as:
```bash
python -m scripts.setup
```

**Features:**
-  **GPU Detection**: Auto-detects VRAM and calculates optimal workers
-  **Configuration**: Updates `experiment_settings.yaml` with optimal values
-  **Validation**: Checks Ollama status and environment variables
-  **Status Display**: Shows current configuration at any time
-  **Auto Mode**: Fully automatic configuration based on GPU
-  **Manual Mode**: Set specific worker counts if needed

**Modes:**

```bash
# Interactive (asks for confirmation)
python -m scripts.setup

# Check current configuration
python -m scripts.setup --check

# Auto-configure based on GPU
python -m scripts.setup --auto

# Set specific worker count
python -m scripts.setup --workers 6

# Set both Ollama and cloud workers
python -m scripts.setup --workers 6 --cloud-workers 10
```

### 2. Module Entry Point (`scripts/__main__.py`)

Allows running the setup as a Python module:
```bash
python -m scripts.setup
```

### 3. Documentation

**[docs/RUNNING_MULTIPLE_EXPERIMENTS.md](../docs/RUNNING_MULTIPLE_EXPERIMENTS.md)**
- Complete guide for the new workflow
- Example scenarios (same model/different strategies, different models/same strategy)
- When to re-run setup vs when to just run experiments
- Troubleshooting tips

**Updated [README.md](../README.md)**
- Updated installation instructions
- Added setup steps before running experiments
- Clear separation between setup and execution

## Benefits

###  Setup Once, Run Many

```bash
# Day 1 - Setup (once)
python -m scripts.setup --auto

# Days 1-N - Run experiments (many times)
python scripts/main.py  # Model A, zero-shot
python scripts/main.py  # Model A, few-shot
python scripts/main.py  # Model B, few-shot
python scripts/main.py  # Model B, zero-shot
# ... no reconfiguration needed!
```

###  Easy Validation

```bash
# Check if configuration is still valid
python -m scripts.setup --check
```

Output:
```
 GPU:
   • NVIDIA GeForce RTX 5060 Ti
   • VRAM: 15.9GB total, 14.1GB free

 Ollama:
   • Status: Running (PID 205)
   • API: Accessible
   • Configuration:
     - OLLAMA_NUM_PARALLEL=6
     - OLLAMA_KEEP_ALIVE=15m

  Python Configuration:
   • max_workers_ollama: 6
   • max_workers_cloud: 10
```

###  Intelligent Recommendations

The setup utility analyzes your GPU and provides tailored recommendations:

```
 Recommended max_workers_ollama: 6
   (Based on ~4.5GB model size + context buffers)
```

###  Safety Checks

-  Warns if Ollama is not running
-  Warns if OLLAMA_NUM_PARALLEL is not set
-  Warns if Python workers > Ollama parallel slots
-  Warns if worker count is too high for GPU

###  Workflow Clarity

```
Setup:       python -m scripts.setup --auto
Validate:    python -m scripts.setup --check
Run:         python scripts/main.py
```

No confusion about what to run when!

## Usage Examples

### Scenario 1: First Time User

```bash
# 1. Setup environment
python -m scripts.setup

# Follow interactive prompts...
#  Configuration complete!

# 2. Configure Ollama
./scripts/configure_parallel_ollama.sh

# 3. Run first experiment
python scripts/main.py
```

### Scenario 2: Quick Auto-Setup

```bash
# One command to configure everything
python -m scripts.setup --auto && ./scripts/configure_parallel_ollama.sh

# Ready to run!
python scripts/main.py
```

### Scenario 3: Running Multiple Experiments

```bash
# Setup once
python -m scripts.setup --auto

# Run many experiments - just change config between runs
for model in llama3.1:8b qwen2.5:7b deepseek-r1:7b; do
    # Update experiment_settings.yaml with $model
    python scripts/main.py
done
```

### Scenario 4: Check Configuration After Restart

```bash
# After rebooting or restarting Ollama
python -m scripts.setup --check

# If Ollama lost configuration:
./scripts/configure_parallel_ollama.sh
```

### Scenario 5: Adjust Worker Count

```bash
# Increase parallelism for smaller models
python -m scripts.setup --workers 8

# Update Ollama to match
OLLAMA_NUM_PARALLEL=8 ./scripts/configure_parallel_ollama.sh

# Check it worked
python -m scripts.setup --check
```

## Integration with Existing Code

The setup utility **doesn't change** the experiment execution code. It only:

1. Updates `config/experiment_settings.yaml`:
   ```yaml
   performance:
     max_workers_ollama: 6
     max_workers_cloud: 10
   ```

2. The existing `get_safe_max_workers()` function in `prompt_evaluator.py` reads these values

3. Experiments run with the configured parallelism automatically

No code changes needed in `main.py`, `multi_evolution.py`, etc!

## Files Created/Modified

### Created:
-  `scripts/setup.py` - Main setup utility (340 lines)
-  `scripts/__main__.py` - Module entry point
-  `docs/RUNNING_MULTIPLE_EXPERIMENTS.md` - Complete workflow guide
-  `docs/SETUP_UTILITY_ARCHITECTURE.md` - This document

### Modified:
-  `README.md` - Updated with new workflow
-  `config/experiment_settings.yaml` - Performance section (already done previously)
-  `scripts/prompt_evaluator.py` - Uses configured values (already done previously)

## Backward Compatibility

 **Fully backward compatible**

- Old workflow still works (manually editing config)
- New setup utility is optional but recommended
- Existing experiments continue to work
- No breaking changes

## Testing

All features tested and working:

```bash
 python -m scripts.setup --check    # Status display
 python -m scripts.setup --auto     # Auto-configuration
 python -m scripts.setup --workers  # Manual setting
 GPU detection working
 Ollama status detection working
 Configuration updates working
 Validation and warnings working
```

## Summary

The new setup utility provides:

1. **Clear separation** between setup and execution
2. **Easy validation** of configuration
3. **Intelligent recommendations** based on GPU
4. **Safety warnings** for problematic configurations
5. **Fast iteration** - run many experiments without reconfiguring

Perfect for your use case: **setup once, then freely experiment with different models and strategies!** 
