# Parallel Ollama Optimization Guide

## Current Setup Analysis

### Hardware
- **GPU**: RTX 5060 Ti (16GB VRAM)
- **Available VRAM**: ~9GB (16GB - 7GB current usage)
- **CUDA**: Version 13.0
- **Driver**: 581.80

### Current Ollama Configuration
- **Running as**: systemd service (`/etc/systemd/system/ollama.service`)
- **Parallelism**: NOT CONFIGURED (running sequentially)
- **Problem**: Environment variables from `ollama-local.sh` are NOT being used

## Why You're Not Seeing Full Performance

Your Ollama is running via systemd service which doesn't use the parallel configuration from your `ollama-local.sh` script. This means:

 **Current**: Processing requests **sequentially** (1 at a time)
 **Goal**: Process **6-8 requests in parallel** (utilize all VRAM)

## Optimization Strategy

### Option 1: Update Systemd Service (Recommended)
Update the systemd service to enable parallelism:

```bash
# Edit the systemd service
sudo nano /etc/systemd/system/ollama.service
```

Add these lines under `[Service]`:
```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=6"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_KEEP_ALIVE=10m"
Environment="OLLAMA_GPU_OVERHEAD=512MiB"
```

Then reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Option 2: Stop Service, Use Script
Stop the systemd service and use your optimized script:

```bash
# Disable systemd service
sudo systemctl stop ollama
sudo systemctl disable ollama

# Start with your script
cd /home/barrel/coding/tremdesara/mo-prompt-project
./scripts/ollama-local.sh
```

## Calculate Optimal Settings

Use the built-in calculator to get personalized recommendations:

```bash
cd /home/barrel/coding/tremdesara/mo-prompt-project
source .venv/bin/activate
python scripts/calculate_optimal_workers.py
```

This will analyze your GPU and provide tailored recommendations.

## Recommended Parallel Configuration

Based on your 16GB VRAM and typical model sizes:

### For 7B Models (deepseek-r1:7b, llama3.1:8b, qwen2.5:7b)
- **Model size (q4_k_m)**: ~4.5-5GB each
- **Per-request KV cache**: ~500MB-1GB (depends on context length)
- **Safe parallelism**: 6-8 concurrent requests

```bash
export OLLAMA_NUM_PARALLEL=6          # Process 6 requests simultaneously
export OLLAMA_MAX_LOADED_MODELS=2     # Keep 2 models in VRAM
export OLLAMA_KEEP_ALIVE=15m          # Keep models loaded longer
export OLLAMA_GPU_OVERHEAD=1024MiB    # Reserve 1GB for system
```

### For 8B Models (Bode-3.1-8B)
- **Model size**: ~5-6GB
- **Safe parallelism**: 4-6 concurrent requests

```bash
export OLLAMA_NUM_PARALLEL=5
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=15m
```

## Expected Performance Gains

| Configuration | Requests/Sec | Throughput | Speedup |
|--------------|--------------|------------|---------|
| **Sequential (current)** | ~1.0 | 1x | Baseline |
| **Parallel (4 slots)** | ~2.5 | 2.5x | 2.5x faster |
| **Parallel (6 slots)** | ~3.5 | 3.5x | **3.5x faster** |
| **Parallel (8 slots)** | ~4.0 | 4.0x | **4.0x faster** |

Note: Actual speedup depends on model complexity. DeepSeek-R1 with long thinking chains will have lower parallelism efficiency.

## Code Adjustments Required

### 1. Update Python Workers
Match Python workers to Ollama parallelism:

In `scripts/prompt_evaluator.py`:
```python
# Old: MAX_WORKERS = 4 if evaluator_type == "ollama" else 10
# New: Match your OLLAMA_NUM_PARALLEL setting
MAX_WORKERS = 6 if evaluator_type == "ollama" else 10
```

### 2. Increase Timeout for Parallel Queue
In `scripts/llm_clients.py`:
```python
# Old: timeout = model_config.get("timeout", 120)
# New: Allow more time for queued requests
timeout = model_config.get("timeout", 180)  # 3 minutes
```

## VRAM Usage Estimation

```
Base Model (deepseek-r1:7b): 4.7GB
+ 6 parallel contexts (2048 tokens each): ~3-4GB
+ GPU Overhead: 1GB
----------------------------------------
Total: ~8.7-9.7GB (fits in your 16GB!)
```

## How to Verify It's Working

### 1. Check Environment Variables
```bash
cat /proc/$(pgrep -f "ollama serve")/environ | tr '\0' '\n' | grep OLLAMA
```

### 2. Monitor GPU with Better Detail
```bash
watch -n 1 nvidia-smi
```

### 3. Test Parallel Performance
```bash
# Run this test script (see below)
python scripts/test_parallel_ollama.py
```

## Monitoring Tips

### Better GPU Process Visibility (WSL)
```bash
# Install nvidia-htop for better process tracking
pip install nvidia-htop
nvidia-htop

# Or use this to see CUDA processes
nvidia-smi pmon -c 1
```

### Check Ollama Logs
```bash
# If using systemd
sudo journalctl -u ollama -f

# If using script
tail -f scripts/ollama_server.log
```

## Troubleshooting

### GPU Usage Still Low After Changes
1. Verify environment variables are set: `cat /proc/$(pgrep -f "ollama serve")/environ | tr '\0' '\n' | grep OLLAMA`
2. Check Ollama logs for errors
3. Ensure your Python code is actually sending parallel requests

### Out of Memory Errors
- Reduce `OLLAMA_NUM_PARALLEL` to 4 or 5
- Reduce model context length: Add `"num_ctx": 2048` to model config
- Use smaller quantization (q4_0 instead of q4_k_m)

### Slower Performance with Parallelism
- Check if model is memory-bound: reduce parallel slots
- DeepSeek-R1 generates long thinking chains - may need fewer parallel slots
- Monitor GPU utilization - should stay 80-95%

## Next Steps

1.  Apply systemd service configuration (Option 1)
2.  Update Python worker count to match `OLLAMA_NUM_PARALLEL`
3.  Run performance test
4.  Monitor and tune based on actual VRAM usage

## Advanced: Multiple Ollama Instances (Not Recommended Yet)

Only consider this if you have multiple GPUs or need extreme parallelism:
- Run multiple Ollama instances on different ports
- Requires complex load balancing
- Your single RTX 5060 Ti is better optimized with `OLLAMA_NUM_PARALLEL`
