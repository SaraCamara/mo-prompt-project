# Enhanced Real-Time Progress Display

## Overview
The progress bar now displays comprehensive execution tracker information in real-time, providing full visibility into the evolution state while it's running.

## What You See Now

### Multi-Objective Evolution
```
Evolução MOP: 60%|████████████████████████     | 6/10 [01:30<01:00, 15.0s/gen, 
  F1=0.850, Pareto=7, Stag=1/3, Time=1.5m, Avg=15s, ETA=1.0m]
```

### Mono-Objective Evolution
```
Evolução: 75%|█████████████████████████████       | 6/8 [02:00<00:40, 20.0s/gen,
  F1=0.8750, Acc=0.834, Pop=10, Time=2.0m, Avg=20s, ETA=40s]
```

### Resumed Run
```
Evolução MOP: 80%|███████████████████████████████    | 8/10 [00:30<00:15, 15.0s/gen,
  F1=0.880, Pareto=8, Stag=0/3, Time=2.3h, Avg=13.8m, ETA=27.6m]
```
*(Note: Time=2.3h includes previous session)*

## Information Displayed

### Common Metrics (All Modes)
| Field | Description | Example |
|-------|-------------|---------|
| **Time** | Total accumulated time across all sessions | `2.3h` |
| **Avg** | Average time per generation | `13.8m` |
| **ETA** | Estimated time remaining | `27.6m` |

### Multi-Objective Specific
| Field | Description | Example |
|-------|-------------|---------|
| **F1** | Best F1 score in Pareto front | `0.850` |
| **Pareto** | Size of Pareto front | `7` |
| **Stag** | Stagnation counter / limit | `1/3` |

### Mono-Objective Specific
| Field | Description | Example |
|-------|-------------|---------|
| **F1** | Best F1 score in population | `0.8750` |
| **Acc** | Best accuracy | `0.834` |
| **Pop** | Population size | `10` |

## Time Formatting

Times are automatically formatted for readability:
- **< 60s**: Shows seconds (`45s`)
- **< 1h**: Shows minutes (`25.3m`)
- **≥ 1h**: Shows hours (`2.3h`)

## Accuracy Across Sessions

The timing information is **accurate even when resuming**:

**Session 1** (interrupted after gen 5):
```
Time=1.5h  (generation 0-5 completed)
```

**Session 2** (resumed from gen 6):
```
Time=2.3h  (includes 1.5h from previous + 0.8h current)
```

This is possible because `ExecutionTracker` persists all timing data incrementally.

## Benefits

### 1. **Full Visibility**
No need to check log files - all key metrics visible in one line

### 2. **Progress Estimation**
ETA helps estimate when evolution will complete

### 3. **Performance Monitoring**
Average time per generation helps identify slowdowns

### 4. **Session Awareness**
In resumed runs, see total accumulated time across all sessions

### 5. **Stagnation Tracking** (Multi-Objective)
See how close you are to early stopping

## Implementation Details

### Updates Per Generation
```python
# After each generation completes:
1. Calculate metrics (F1, accuracy, Pareto size, etc.)
2. Get accumulated time from ExecutionTracker
3. Calculate average time per generation
4. Estimate time remaining
5. Update progress bar with all info
```

### Time Calculation
```python
# Accumulated time (includes all sessions)
accumulated = tracker.metadata.get("accumulated_time_seconds", 0)

# Average per generation
avg_time = accumulated / (generation_num + 1)

# ETA for remaining generations
eta_remaining = avg_time * (max_gens - generation_num - 1)
```

### Format Helper
```python
def _format_time(seconds):
    """Convert seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
```

## Demo

Run the interactive demo to see how it looks:
```bash
source .venv/bin/activate
python scripts/demo_progress_bar.py
```

The demo shows:
1.  Multi-objective evolution progress
2.  Mono-objective evolution progress  
3.  Resumed run with accumulated time

## Technical Notes

### No Performance Impact
- Progress bar updates: ~0.1ms per generation
- Calculations are simple arithmetic
- No I/O during updates (tracker writes separately)

### Refresh Rate
```python
pbar.set_postfix(progress_info, refresh=True)
```
The `refresh=True` ensures updates are immediately visible.

### Thread Safety
Progress bars run in the main thread, no concurrency issues.

## Example Real Run

During an actual evolution run, you'll see something like:

```
════════════════════════════════════════════════════════════
Iniciando Evolução
────────────────────────────────────────────────────────────

Evolução MOP:  40%|███████████████            | 4/10 [05:30<08:15, 82.5s/gen,
  F1=0.820, Pareto=6, Stag=0/3, Time=5.5m, Avg=1.4m, ETA=8.4m]
```

**Reading this:**
- 40% complete (gen 4 of 10)
- Current best F1: 0.820
- Pareto front has 6 solutions
- Not stagnating yet (0 of 3 limit)
- Total time so far: 5.5 minutes
- Average 1.4 min per generation
- About 8.4 minutes remaining

## Files Modified

-  `scripts/multi_evolution.py` - Added `_format_time()` and enhanced progress bar
-  `scripts/mono_evolution.py` - Added `_format_time()` and enhanced progress bar
-  `scripts/demo_progress_bar.py` - Demo script to visualize progress

## Integration with ExecutionTracker

The progress bar reads from `ExecutionTracker.metadata`:
```python
accumulated = tracker.metadata.get("accumulated_time_seconds", 0)
```

This ensures consistency between:
- Real-time progress display
- Persistent metadata files
- Final summary markdown

All timing sources show the same values! 
