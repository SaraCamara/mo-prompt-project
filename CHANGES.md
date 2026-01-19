# Changes Summary - Model Name & Evolution Summary

## Overview
This update restructures evaluator configuration to use clearer field names and adds comprehensive execution summaries with timing metrics.

## 1. Configuration Restructure (model/name fields)

### Before:
```yaml
evaluators:
  - name: hf.co/mradermacher/Bode-3.1-8B-Instruct-full-GGUF:Q4_K_M
    alias: bode3.1-8b  # Optional display name
```

### After:
```yaml
evaluators:
  - model: hf.co/mradermacher/Bode-3.1-8B-Instruct-full-GGUF:Q4_K_M
    name: bode3.1-8b  # Display name for CLI/logs
```

### Field Usage:
- **`model`**: Full model identifier for LLM API calls (Ollama, OpenAI, Maritalk)
- **`name`**: Human-readable display name for CLI menus, logs, and file paths

### Files Modified:

#### config/experiment_settings.yaml
- Renamed `name` → `model` for all evaluators
- Renamed `alias` → `name` for evaluators that had aliases
- Added display names for all models (e.g., `deepseek-r1`, `qwen2.5`, `llama3.1`, `bode3.1-8b`)

#### scripts/llm_clients.py
- `query_maritalk()`: Changed `model_config.get("name")` → `model_config.get("model")`
- `query_ollama()`: Changed `model_config.get("name")` → `model_config.get("model")`
- `_call_openai_api()`: Changed to use `model_config.get("model", model_config.get("name"))` for backward compatibility

#### scripts/main.py
- Removed regex-based name extraction: `re.split(r'[:/_-]', evaluator_name)[0]`
- Now directly uses: `model_name = evaluator.get("name", "unknown")`
- Removed unused `import re`

#### scripts/results_saver.py
- Removed alias fallback logic: `evaluator.get("alias", evaluator.get("name"))`
- Now simply uses: `evaluator["name"]`

#### scripts/multi_evolution.py
- Removed alias fallback in logging
- Now uses: `evaluator_config['name']` directly

#### scripts/mono_evolution.py
- Removed alias fallback in logging
- Now uses: `evaluator_config['name']` directly

## 2. Evolution Summary with Timing Metrics

### New Feature: evolution_summary.md

Each evolution run now generates a comprehensive markdown summary at the end with:

#### Sections:
1. **Experiment Configuration**
   - Task, optimization mode, model details, strategy, execution date

2. **Evolution Parameters**
   - Population size, max generations, mutation rate, k-tournament, stagnation limit, top_k

3. **Time Metrics**
   - Start/end timestamps
   - Total duration (formatted as HH:MM:SS)
   - Generations completed
   - Average, min, max time per generation

4. **Final Results**
   - Best F1 score, accuracy, token count
   - Final population size
   - Pareto front size (multi-objective only)

5. **Time per Generation Table**
   - Detailed breakdown of each generation's execution time

6. **Output Files**
   - List of generated result files and directories

### Files Modified:

#### scripts/results_saver.py
- **New function**: `save_evolution_summary_markdown(config, metrics, output_dir)`
- Generates comprehensive markdown report with all metrics

#### scripts/multi_evolution.py
- Added timing imports: `time`, `datetime`
- Records `start_time` at function entry
- Tracks `generation_times[]` list for each generation
- Calculates final metrics (best F1, accuracy, tokens, pareto size)
- Calls `save_evolution_summary_markdown()` at end

#### scripts/mono_evolution.py
- Added timing imports: `time`, `datetime`
- Records `start_time` at function entry
- Tracks `generation_times[]` list for each generation
- Calculates final metrics (best F1, accuracy, tokens)
- Calls `save_evolution_summary_markdown()` at end
- Removed unused imports: `pandas`, `random`

## 3. Impact on Existing Runs

### Log Directory Structure:
**Before**:
```
logs/squad/mop/hf.co/zero-shot/  #  Not distinctive
```

**After**:
```
logs/squad/mop/bode3.1-8b/zero-shot/  #  Clear model identifier
```

### Backward Compatibility:
- Old log directories remain unchanged
- New runs will use the new naming convention
- Generator configs support both `model` and `name` fields for compatibility

## 4. Testing

Test script created: `scripts/test_config_changes.py`

**Output**:
```
✓ All 5 evaluators have correct structure

Bode model example:
  API will use: hf.co/mradermacher/Bode-3.1-8B-Instruct-full-GGUF:Q4_K_M
  Logs will use: bode3.1-8b
```

## 5. Example Summary Output

After each evolution run, `logs/<task>/<mode>/<model>/<strategy>/evolution_summary.md` will contain:

```markdown
# Resumo da Evolução de Prompts

## Configuração do Experimento
- **Tarefa**: squad
- **Modo de Otimização**: multiobjetivo
- **Modelo Avaliador**: bode3.1-8b (`hf.co/mradermacher/Bode-3.1-8B-Instruct-full-GGUF:Q4_K_M`)
- **Estratégia**: zero-shot
- **Data de Execução**: 2026-01-16 14:30:45

## Parâmetros Evolutivos
```yaml
population_size: 10
max_generations: 10
mutation_rate: 0.8
...
```

## Métricas de Tempo
- **Duração Total**: 1h 25m 30s
- **Gerações Completadas**: 10
- **Tempo Médio por Geração**: 512.30s

## Resultados Finais
- **Melhor F1 Score**: 0.8542
- **Tamanho da Fronteira de Pareto**: 8
...
```

## Benefits

1. **Clearer Configuration**: Explicit separation of API identifiers vs display names
2. **Better Log Organization**: Distinctive directory names (e.g., `bode3.1-8b` instead of `hf.co`)
3. **Comprehensive Tracking**: Detailed timing and metrics for each run
4. **Reproducibility**: Evolution parameters recorded in each summary
5. **Performance Analysis**: Per-generation timing helps identify bottlenecks
