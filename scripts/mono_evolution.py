# mono_evolution.py
import os
import pandas as pd
import random
import logging
from .population_manager import evaluate_population, generate_unique_offspring # type: ignore
from .results_saver import ( # type: ignore
    save_generation_results, save_final_results, save_sorted_population
)

def run_mono_evolution(config, dataset, initial_prompts, output_csv_path, start_generation=0, initial_population=None):
    print("[mono_evolution] Iniciando execução da evolução mono-objetivo")
    logger = logging.getLogger(__name__)
    logger.info("Iniciando execução da evolução mono-objetivo")

    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    population_size = config.get("population_size", 10)

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
    for generation in range(current_generation, config["max_generations"]):
        current_generation_number = generation
        logger.info(f"--- Geração {current_generation_number}---")

        try:
            # Geração de Filhos usando a função genérica do utils
            offspring_prompts_list_of_dicts = generate_unique_offspring(
                population,
                config,
                evolution_type="mono"
            )
            
            # offspring_prompts_list_of_dicts will be a list of strings here
            if not offspring_prompts_list_of_dicts:
                logger.warning("Nenhum descendente único foi gerado. Pulando para a próxima geração.")
                continue

            logger.info(f"{len(offspring_prompts_list_of_dicts)} descendentes gerados. Avaliando.")

            # Avaliação dos Filhos
            evaluated_offspring = evaluate_population(offspring_prompts_list_of_dicts, dataset, config, evaluator_config)
            
            # Seleção de Sobreviventes
            combined_population = population + evaluated_offspring
            # F1_Score
            combined_population.sort(key=lambda x: (x["metrics"][1], -x["metrics"][2] if len(x["metrics"]) >= 3 else float('inf')), reverse=True)
            
            population = combined_population[:population_size]
            
            if population:
                logger.info(f"População da próxima geração selecionada (Tamanho: {len(population)}). Melhor f1_score: {population[0]['metrics'][1]:.4f}")
            else:
                logger.warning("População ficou vazia após seleção.")


            # Salvando Resultados da Geração
            save_sorted_population(population, current_generation_number, generation_log_dir)
            save_generation_results(population, current_generation_number, config, generation_log_dir) 
        
        except Exception as e:
            logger.error(f"Erro na geração {current_generation_number}: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            continue 

    # Fim do Ciclo Evolutivo
    logger.info("Evolução mono-objetivo concluída.")
    logger.info("Salvando resultados finais.")
    save_final_results(population, config, output_csv_path) 
    logger.info(f"Resultados salvos em {output_csv_path}")