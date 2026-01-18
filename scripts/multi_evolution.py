# multi_evolution.py
import os
import random 
import logging
import time
import datetime
from tqdm import tqdm
from .population_manager import evaluate_population, generate_unique_offspring, select_survivors_nsgaii # type: ignore
from .nsga2_algorithms import compute_crowding_distance, fast_non_dominated_sort # type: ignore
from .results_saver import save_pareto_front_data, save_evolution_summary_markdown # type: ignore
from .execution_tracker import ExecutionTracker # type: ignore

def _format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def run_multi_evolution(config, dataset, initial_prompts_text, output_csv_path, output_plot_path, start_generation=0, initial_population=None, loaded_state=None):
    print("[multi_evolution] Iniciando execução da evolução multiobjetivo")
    logger = logging.getLogger(__name__)
    logger.info("Iniciando execução da evolução multiobjetivo")

    # Métricas de tempo
    start_time = time.time()
    start_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Inicializa ExecutionTracker
    base_output_dir = config["base_output_dir"]
    tracker = ExecutionTracker(base_output_dir)
    
    # Carrega estado anterior se estiver retomando
    resuming = loaded_state is not None
    tracker.initialize(config, start_time, start_datetime, resuming=resuming)
    
    generation_times = tracker.get_generation_times() if resuming else []

    # Configuração Inicial
    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    population_size = config.get("evolution_params", {}).get("population_size", 10)
    base_output_dir = config["base_output_dir"]
    per_generation_pareto_log_dir = os.path.join(base_output_dir, "per_generation_pareto")
    os.makedirs(per_generation_pareto_log_dir, exist_ok=True)
    executor_config = evaluator_config
    
    logger.info(f"Avaliador: {evaluator_config['name']}")
    logger.info(f"Estratégia: {strategy_config['name']}")

    current_population = []
    current_generation = start_generation

    if initial_population:
        logger.info(f"Retomando execução da Geração {start_generation - 1} com população carregada.")
        current_population = initial_population
    else:
        # Passo 1: Avaliação da População Inicial (P_0)
        logger.info("Avaliando população inicial...")
        current_population = evaluate_population(initial_prompts_text, dataset, config, executor_config)
        logger.info(f"População inicial avaliada. Tamanho: {len(current_population)}")

        # Log para acompanhar os scores de cada indivíduo
        logger.info("Scores da população inicial:")
        for i, ind in enumerate(current_population):
            # Usando .get() para evitar erros caso uma chave não exista
            logger.info(f"  - Prompt: \"{ind.get('prompt', 'N/A')}\" | F1: {ind.get('f1', 0.0):.4f} | Acc: {ind.get('acc', 0.0):.4f} | Tokens: {ind.get('tokens', 0)}")

        # Classifica a população inicial para obter os ranks
        initial_fronts = fast_non_dominated_sort(current_population)

        # Calcula a crowding distance para cada fronteira da população inicial
        logger.info("Calculando crowding distance para a população inicial...")
        for front in initial_fronts:
            compute_crowding_distance(front)

        # O código abaixo, para salvar a fronteira de Geração 0
        if initial_fronts and initial_fronts[0]:
            save_pareto_front_data(initial_fronts[0], os.path.join(per_generation_pareto_log_dir, "pareto_gen_0.csv"), os.path.join(per_generation_pareto_log_dir, "pareto_gen_0.png"))

    # Passo 2: Ciclo de Gerações
    # Restaura estado de estagnação se estiver retomando
    if loaded_state:
        stagnation_counter = loaded_state.get("stagnation_counter", 0)
        last_front_hash = loaded_state.get("last_front_hash")
        logger.info(f"Restaurado: stagnation_counter={stagnation_counter}")
    else:
        stagnation_counter = 0
        last_front_hash = None
    
    stagnation_limit = config.get("evolution_params", {}).get("stagnation_limit", 3)
    max_gens = config["evolution_params"]["max_generations"]
    gen_range = range(current_generation, max_gens)

    with tqdm(gen_range, desc="Evolução MOP", unit="gen", initial=current_generation, total=max_gens) as pbar:
        for generation_num in pbar:
            gen_start_time = time.time()
            pbar.set_description(f"Geração {generation_num}")

            offspring_prompts = generate_unique_offspring(current_population, config, evolution_type="multi")
            if not offspring_prompts:
                logger.warning("Nenhum filho único foi gerado nesta geração.")
                continue
            
            evaluated_offspring = evaluate_population(offspring_prompts, dataset, config, executor_config)
            
            current_population = select_survivors_nsgaii(current_population, evaluated_offspring, population_size)

            current_pareto_front = [ind for ind in current_population if ind.get('rank') == 0]
            if current_pareto_front:
                # Atualiza barra de progresso com métricas detalhadas
                best_f1 = max(ind.get('f1', 0) for ind in current_pareto_front)
                
                # Calcula estatísticas de tempo
                accumulated = tracker.metadata.get("accumulated_time_seconds", 0)
                avg_time = accumulated / (generation_num + 1) if generation_num >= 0 else 0
                eta_remaining = avg_time * (max_gens - generation_num - 1) if generation_num < max_gens else 0
                
                # Monta informações para a barra
                progress_info = {
                    "F1": f"{best_f1:.3f}",
                    "Pareto": len(current_pareto_front),
                    "Stag": f"{stagnation_counter}/{stagnation_limit}",
                    "Time": _format_time(accumulated),
                    "Avg": _format_time(avg_time),
                    "ETA": _format_time(eta_remaining)
                }
                pbar.set_postfix(progress_info, refresh=True)
                
                save_pareto_front_data(
                    current_pareto_front, 
                    os.path.join(per_generation_pareto_log_dir, f"pareto_gen_{generation_num}.csv"),
                    os.path.join(per_generation_pareto_log_dir, f"pareto_gen_{generation_num}.png")
                )
                
                front_prompts_tuple = tuple(sorted([ind['prompt'] for ind in current_pareto_front]))
                current_front_hash = hash(front_prompts_tuple)

                if current_front_hash == last_front_hash:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                
                last_front_hash = current_front_hash

                if stagnation_counter >= stagnation_limit:
                    logger.info(f"Estagnação por {stagnation_limit} gerações. Parando.")
                    break
            else:
                logger.warning(f"Fronteira de Pareto da geração {generation_num} vazia.")
            
            # Registra tempo da geração
            gen_elapsed = time.time() - gen_start_time
            generation_times.append(gen_elapsed)
            
            # Atualiza tracker com estado da geração
            gen_metrics = {
                "best_f1": max(ind.get('f1', 0) for ind in current_pareto_front) if current_pareto_front else 0.0,
                "pareto_size": len(current_pareto_front)
            }
            tracker.update_generation(
                generation_num, gen_elapsed, gen_metrics,
                stagnation_counter=stagnation_counter,
                front_hash=current_front_hash if 'current_front_hash' in locals() else None
            )

    # Fim do Ciclo Evolutivo
    try:
        logger.info("Evolução multiobjetivo concluída.")
        
        final_pareto_front = [ind for ind in current_population if ind.get('rank') == 0]
        
        save_pareto_front_data(final_pareto_front, output_csv_path, output_plot_path)

        logger.info(f"Resultados finais salvos em '{output_csv_path}' e gráfico em '{output_plot_path}'.")
        
        # Métricas finais
        end_time = time.time()
        end_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_duration = tracker.metadata.get("accumulated_time_seconds", 0) + (end_time - start_time)
        
        # Extrai métricas da população final
        final_best_f1 = max((ind.get('f1', 0) for ind in final_pareto_front), default=0.0)
        final_best_acc = max((ind.get('acc', 0) for ind in final_pareto_front), default=0.0)
        best_ind = max(final_pareto_front, key=lambda x: x.get('f1', 0), default={})
        final_best_tokens = best_ind.get('tokens', 0)
        
        metrics = {
            "start_time": tracker.metadata.get("start_time", start_datetime),
            "end_time": end_datetime,
            "total_duration_seconds": total_duration,
            "generations_completed": generation_num + 1 if 'generation_num' in locals() else 0,
            "final_best_f1": final_best_f1,
            "final_best_acc": final_best_acc,
            "final_best_tokens": final_best_tokens,
            "final_population_size": len(current_population),
            "pareto_front_size": len(final_pareto_front),
            "generation_times": generation_times
        }
        
        # Marca como completo no tracker
        tracker.mark_completed(final_metrics=metrics)
        
        # Gera resumo em markdown
        save_evolution_summary_markdown(config, metrics, config["base_output_dir"])
        
    except Exception as e:
        logger.error(f"Erro ao finalizar evolução: {e}", exc_info=True)
        tracker.mark_stopped(reason="error_finalization")
        raise