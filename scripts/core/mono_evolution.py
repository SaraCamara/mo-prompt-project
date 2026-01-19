# mono_evolution.py
import os
import logging
import time
import datetime
from tqdm import tqdm
from .population_manager import evaluate_population, generate_unique_offspring
from ..utils import ExecutionTracker
from ..utils import (
    save_generation_results, save_final_results, save_sorted_population, save_evolution_summary_markdown
)

def _format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def run_mono_evolution(config, dataset, initial_prompts, output_csv_path, start_generation=0, initial_population=None, loaded_state=None):
    print("[mono_evolution] Iniciando execução da evolução mono-objetivo")
    logger = logging.getLogger(__name__)
    logger.info("Iniciando execução da evolução mono-objetivo")

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

    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    population_size = config.get("evolution_params", {}).get("population_size", 10)

    # Configuração de Caminhos
    base_output_dir = config["base_output_dir"]
    generation_log_dir = os.path.join(base_output_dir, "generations_detail")
    os.makedirs(generation_log_dir, exist_ok=True)

    logger.info(f"Avaliador: {evaluator_config['name']}")
    logger.info(f"Estratégia: {strategy_config['name']}")

    population = []
    current_generation = start_generation

    if initial_population:
        logger.info(f"Retomando execução da Geração {start_generation - 1} com população carregada.")
        population = initial_population
    else:
        # Passo 1: Avaliação da População Inicial
        logger.info("Avaliando população inicial.")
        population = evaluate_population(initial_prompts, dataset, config, evaluator_config)
        save_sorted_population(population, 0, generation_log_dir)

    # Ciclo de Gerações (ajustado para retomar)
    max_gens = config["max_generations"]
    gen_range = range(current_generation, max_gens)
    
    with tqdm(gen_range, desc="Evolução", unit="gen", initial=current_generation, total=max_gens) as pbar:
        for generation in pbar:
            gen_start_time = time.time()
            current_generation_number = generation
            pbar.set_description(f"Geração {current_generation_number}")

            try:
                # Geração de Filhos usando a função genérica do utils
                offspring_prompts_list_of_dicts = generate_unique_offspring(
                    population,
                    config,
                    evolution_type="mono"
                )
                
                if not offspring_prompts_list_of_dicts:
                    logger.warning("Nenhum descendente único foi gerado.")
                    continue

                # Avaliação dos Filhos
                evaluated_offspring = evaluate_population(
                    offspring_prompts_list_of_dicts, dataset, config, evaluator_config
                )
                
                # Seleção de Sobreviventes
                combined_population = population + evaluated_offspring
                combined_population.sort(
                    key=lambda x: (x["metrics"][1], -x["metrics"][2] if len(x["metrics"]) >= 3 else float('inf')),
                    reverse=True
                )
                
                population = combined_population[:population_size]
                
                if population:
                    best_f1 = population[0]['metrics'][1]
                    best_acc = population[0]['metrics'][0]
                    
                    # Calcula estatísticas de tempo
                    accumulated = tracker.metadata.get("accumulated_time_seconds", 0)
                    avg_time = accumulated / (current_generation_number + 1) if current_generation_number >= 0 else 0
                    eta_remaining = avg_time * (max_gens - current_generation_number - 1) if current_generation_number < max_gens else 0
                    
                    # Monta informações para a barra
                    progress_info = {
                        "F1": f"{best_f1:.4f}",
                        "Acc": f"{best_acc:.3f}",
                        "Pop": len(population),
                        "Time": _format_time(accumulated),
                        "Avg": _format_time(avg_time),
                        "ETA": _format_time(eta_remaining)
                    }
                    pbar.set_postfix(progress_info, refresh=True)
                else:
                    logger.warning("População ficou vazia após seleção.")

                # Salvando Resultados da Geração
                save_sorted_population(population, current_generation_number, generation_log_dir)
                save_generation_results(population, current_generation_number, config, generation_log_dir) 
            
            except Exception as e:
                logger.error(f"Erro na geração {current_generation_number}: {e}", exc_info=True)
                continue
            
            # Registra tempo da geração
            gen_elapsed = time.time() - gen_start_time
            generation_times.append(gen_elapsed)
            
            # Atualiza tracker com estado da geração
            gen_metrics = {
                "best_f1": population[0]['metrics'][1] if population else 0.0,
                "population_size": len(population)
            }
            tracker.update_generation(current_generation_number, gen_elapsed, gen_metrics) 

    # Fim do Ciclo Evolutivo
    try:
        logger.info("Evolução mono-objetivo concluída.")
        logger.info("Salvando resultados finais.")
        save_final_results(population, config, output_csv_path) 
        logger.info(f"Resultados salvos em {output_csv_path}")
        
        # Métricas finais
        end_time = time.time()
        end_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_duration = tracker.metadata.get("accumulated_time_seconds", 0) + (end_time - start_time)
        
        # Extrai métricas do melhor indivíduo
        final_best_f1 = population[0]['metrics'][1] if population and population[0].get('metrics') else 0.0
        final_best_acc = population[0]['metrics'][0] if population and population[0].get('metrics') else 0.0
        final_best_tokens = population[0]['metrics'][2] if population and len(population[0].get('metrics', [])) > 2 else 0
        
        metrics = {
            "start_time": tracker.metadata.get("start_time", start_datetime),
            "end_time": end_datetime,
            "total_duration_seconds": total_duration,
            "generations_completed": current_generation_number + 1 if 'current_generation_number' in locals() else 0,
            "final_best_f1": final_best_f1,
            "final_best_acc": final_best_acc,
            "final_best_tokens": final_best_tokens,
            "final_population_size": len(population),
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