# Running Multiple Experiments - Quick Guide

## Architecture: Setup Once, Run Many Times

The project now has a clear separation between setup and execution:

```
┌─────────────────────────────────────────┐
│  ONE-TIME SETUP                         │
│  python -m scripts.setup                │
│  • Configure parallelism                │
│  • Verify GPU capacity                  │
│  • Set max_workers                      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  RUN EXPERIMENTS (as many as you want)  │
│  python scripts/main.py                 │
│  • Different models                     │
│  • Different strategies (zero/few-shot) │
│  • Different datasets                   │
│  • Uses configured parallelism          │
└─────────────────────────────────────────┘
```

## One-Time Setup

### First Time (or when changing hardware)

```bash
cd /home/barrel/coding/tremdesara/mo-prompt-project
source .venv/bin/activate

# Option 1: Interactive setup (recommended)
python -m scripts.setup

# Option 2: Auto-configure based on GPU
python -m scripts.setup --auto

# Option 3: Set specific worker count
python -m scripts.setup --workers 6

# Configure Ollama server (if needed)
./scripts/configure_parallel_ollama.sh
```

### Check Your Configuration Anytime

```bash
python -m scripts.setup --check
```

Example output:
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

## Running Experiments

Once setup is complete, you can run as many experiments as you want **without reconfiguring**.

### Scenario 1: Same Model, Different Strategies

```bash
source .venv/bin/activate

# Run with zero-shot
# (Edit experiment_settings.yaml to set strategy: zero-shot)
python scripts/main.py

# Run with few-shot (just change the config)
# (Edit experiment_settings.yaml to set strategy: few-shot)
python scripts/main.py
```

### Scenario 2: Different Models, Same Strategy

```bash
source .venv/bin/activate

# Run with llama3.1:8b
# (Edit experiment_settings.yaml, set evaluator: llama3.1:8b)
python scripts/main.py

# Run with qwen2.5:7b (just change the model)
# (Edit experiment_settings.yaml, set evaluator: qwen2.5:7b)
python scripts/main.py

# Run with deepseek-r1:7b
# (Edit experiment_settings.yaml, set evaluator: deepseek-r1:7b)
python scripts/main.py
```

### Scenario 3: Multiple Experiments in Sequence

```bash
source .venv/bin/activate

# Batch run different configurations
for model in "llama3.1:8b" "qwen2.5:7b" "deepseek-r1:7b"; do
    echo "Running experiment with $model..."
    # Update config programmatically or manually between runs
    python scripts/main.py
done
```

## Configuration Files

### What to Edit for Different Experiments

**`config/experiment_settings.yaml`** - Edit ONLY these sections between runs:

```yaml
# Change the evaluator model
evaluators:
  - model: llama3.1:8b      # ← Change this
    name: llama3.1
    tipo: ollama
    # ... rest stays the same

# Change the strategy
strategies_imdb:
  - name: zero-shot         # ← Or 'few-shot'
    template: |
      {prompt_instruction}
      Texto: "{text}"
```

**`performance` section** - Set ONCE, don't change between runs:

```yaml
performance:
  max_workers_ollama: 6     # ← Set once during setup
  max_workers_cloud: 10     # ← Don't change between experiments
```

## When to Re-run Setup

You only need to re-run setup when:

 **Upgrading GPU** - Different VRAM capacity
 **Changing parallelism needs** - Want more/less concurrent requests
 **After Ollama updates** - Configuration may have been reset
 **Troubleshooting performance** - Check if configuration is correct

You DON'T need to re-run setup when:

 Switching between models
 Changing strategies (zero-shot ↔ few-shot)
 Running different datasets
 Adjusting evolutionary parameters

## Troubleshooting

### Check if setup is still valid

```bash
python -m scripts.setup --check
```

Look for:
-  GPU detected
-  Ollama running with OLLAMA_NUM_PARALLEL set
-  max_workers_ollama matches OLLAMA_NUM_PARALLEL

### If Ollama lost configuration

```bash
# Re-configure Ollama (doesn't change Python settings)
./scripts/configure_parallel_ollama.sh
```

### If you want to change worker count

```bash
# Update Python config
python -m scripts.setup --workers 8

# Update Ollama config
OLLAMA_NUM_PARALLEL=8 ./scripts/configure_parallel_ollama.sh
```

## Example Workflow

**Day 1 - Initial Setup:**
```bash
cd /home/barrel/coding/tremdesara/mo-prompt-project
source .venv/bin/activate
python -m scripts.setup --auto
./scripts/configure_parallel_ollama.sh
```

**Day 1-N - Running Experiments:**
```bash
source .venv/bin/activate

# Experiment 1: llama3.1 zero-shot
# (edit config)
python scripts/main.py

# Experiment 2: llama3.1 few-shot
# (edit config)
python scripts/main.py

# Experiment 3: qwen2.5 few-shot
# (edit config)
python scripts/main.py

# ... keep going! No need to reconfigure
```

**Day 30 - Check if still optimal:**
```bash
python -m scripts.setup --check

# If GPU usage is low, maybe increase workers:
python -m scripts.setup --workers 8
./scripts/configure_parallel_ollama.sh
```

## Benefits of This Architecture

 **Setup Once** - No repeated configuration
 **Fast Iteration** - Just change model/strategy and run
 **Consistent Performance** - Same parallelism across all experiments
 **Easy Validation** - `--check` confirms everything is configured
 **No Downtime** - Switch experiments without restarting Ollama

## Monitoring Resources

Watch your GPU and Ollama usage in real-time:

```bash
# Live dashboard (refreshes every 5s)
python -m scripts.watch

# Fast refresh (2s)
python -m scripts.watch --fast

# Slow refresh (10s)  
python -m scripts.watch --slow

# Custom interval
python -m scripts.watch --interval 3
```

The monitor shows:
-  GPU memory and utilization
-  Ollama server status and configuration
-  Loaded models and their sizes
-  Process-level CPU and memory usage

## Summary Commands

```bash
# Setup (once)
python -m scripts.setup --auto
./scripts/configure_parallel_ollama.sh

# Check (anytime)
python -m scripts.setup --check

# Monitor (during experiments)
python -m scripts.watch

# Run experiments (many times)
source .venv/bin/activate
python scripts/main.py
```

That's it! Configure once, experiment freely! 
