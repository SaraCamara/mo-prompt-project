# Resume Strategy Analysis - Current State & Improvements

## Current Implementation Analysis

### What Gets Saved Per Generation 
1. **Population State** (immediately after each generation):
   - Multi-objective: `per_generation_pareto/pareto_gen_{N}.csv` + `.png`
   - Mono-objective: `generations_detail/population_sorted_gen_{N}.csv` + `results_gen_{N}.csv`
   
2. **Individual Metrics** (in CSVs):
   - Prompt text, accuracy, F1 score, tokens, rank, crowding distance

### What's Kept in Memory  (Lost on Crash)
1. **Timing Data**:
   - `generation_times[]` list - accumulated per-generation execution times
   - `start_time` - global start timestamp
   - NOT written until final summary at end

2. **Execution Metadata**:
   - Start datetime string
   - Stagnation counter state
   - Last front hash (multi-objective)

3. **Progress Context**:
   - Which generation number we're on
   - Why evolution stopped (completed vs stagnation vs error)

### Current Resume Process
1. User manually specifies generation to resume from
2. Loads population CSV (pareto or sorted)
3. Reconstructs individuals from CSV columns
4. Continues from `generation_to_load + 1`

**Problems**:
-  Timing data lost (can't calculate accurate total time)
-  Stagnation counter reset (may run longer than intended)
-  No record of why previous run stopped
-  No automatic detection of last successful generation
-  Manual intervention required

---

## Proposed Improvements

### 1. **Persistent Execution Metadata File**
Create `execution_metadata.json` in base output directory:

```json
{
  "task": "squad",
  "model": "bode3.1-8b",
  "strategy": "zero-shot",
  "objective": "multiobjetivo",
  "start_time": "2026-01-16 10:30:00",
  "last_update_time": "2026-01-16 11:45:30",
  "last_completed_generation": 7,
  "total_generations_planned": 10,
  "status": "running",  // "running" | "completed" | "stopped" | "error"
  "stagnation_counter": 2,
  "last_front_hash": "hash_value_here",
  "evolution_params": {
    "population_size": 10,
    "max_generations": 10,
    "mutation_rate": 0.8,
    ...
  },
  "accumulated_time_seconds": 4530.25,
  "generation_times": [450.2, 458.3, 462.1, ...]
}
```

**Updates**: After each generation completes

### 2. **Per-Generation Timing Log**
Create `timing_log.csv`:

```csv
generation,start_time,end_time,duration_seconds,cumulative_time
0,2026-01-16 10:30:00,2026-01-16 10:37:30,450.2,450.2
1,2026-01-16 10:37:30,2026-01-16 10:45:11,458.3,908.5
...
```

**Updates**: Append after each generation

### 3. **Incremental Summary Updates**
Update `evolution_summary.md` after each generation:
- Not just at the end
- Shows current progress
- Readable even if run crashes

### 4. **Automatic Resume Detection**
Function: `detect_resumable_state(base_output_dir)`

Logic:
1. Check if `execution_metadata.json` exists
2. If status == "running" → likely crashed/interrupted
3. Read `last_completed_generation`
4. Auto-suggest resumption from that generation
5. Load accumulated timing data

### 5. **Checkpointing Strategy**

**Write Operations Per Generation**:
```python
# AFTER generation N completes:
1. Save population CSV (existing)
2. Append to timing_log.csv (new)
3. Update execution_metadata.json (new)
4. Update evolution_summary.md (enhanced)
```

**Atomic Writes**:
- Write to `.tmp` file first
- Rename to actual filename (atomic on POSIX)
- Prevents corruption on crash mid-write

---

## Implementation Plan

### Phase 1: Metadata Persistence
1. Create `ExecutionTracker` class in `scripts/execution_tracker.py`
2. Methods:
   - `initialize(config, base_output_dir)` - create metadata file
   - `update_generation(gen_num, duration, metrics)` - append timing, update metadata
   - `mark_completed()` / `mark_stopped(reason)` - update status
   - `get_resumable_state()` - check if can resume, return last gen

### Phase 2: Resume Enhancement
1. Modify `handle_resumption()` in `main.py`:
   - Call `detect_resumable_state(base_output_dir)`
   - Auto-suggest last completed generation
   - Show accumulated time from previous run
   - Load stagnation counter state

2. Modify evolution functions:
   - Initialize `ExecutionTracker` at start
   - Call `tracker.update_generation()` after each gen
   - Pass loaded timing data if resuming

### Phase 3: Incremental Summaries
1. Modify `save_evolution_summary_markdown()`:
   - Accept `is_incremental=True` flag
   - Show "In Progress" vs "Completed" status
   - Update after each generation, not just at end

### Phase 4: Error Handling
1. Add try/catch around generation loop
2. On error: `tracker.mark_stopped(reason=str(error))`
3. Ensure last successful generation is recorded

---

## Benefits

1. **Zero Information Loss**: All timing data persisted incrementally
2. **Automatic Resume**: No manual generation number input needed
3. **Crash Recovery**: Can resume from exact point of failure
4. **Progress Visibility**: Summary updated live, viewable during run
5. **State Reconstruction**: Stagnation counter, timing, metadata preserved
6. **Debugging**: Know why previous run stopped (completed vs error vs manual)

---

## File Structure After Changes

```
logs/squad/mop/bode3.1-8b/zero-shot/
├── execution_metadata.json          # NEW: Persistent state
├── timing_log.csv                   # NEW: Per-generation timing
├── evolution_summary.md             # ENHANCED: Updated incrementally
├── final_results.csv                # Existing
├── final_pareto_front.png           # Existing
├── per_generation_pareto/           # Existing
│   ├── pareto_gen_0.csv
│   ├── pareto_gen_1.csv
│   └── ...
└── prompt_eval_logs/                # Existing
```

---

## Backward Compatibility

- Old runs without metadata files: Fall back to manual resume
- New runs: Automatic resume detection
- No breaking changes to existing CSV formats
