# multi_evolution.py
import os
import random 
import logging
from scripts.population_manager import evaluate_population, generate_unique_offspring, select_survivors_nsgaii # type: ignore
from scripts.nsga2_algorithms import compute_crowding_distance, fast_non_dominated_sort
from scripts.results_saver import save_pareto_front_data

def run_multi_evolution(config, dataset, initial_prompts_text, output_csv_path, output_plot_path, start_generation=0, initial_population=None):
    print("[multi_evolution] Iniciando execução da evolução multiobjetivo")
    logger = logging.getLogger(__name__)
    logger.info("Iniciando execução da evolução multiobjetivo")

    # Configuração Inicial
    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    population_size = config.get("population_size", 10)
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
        print("[multi_evolution] Scores da população inicial:")
        for i, ind in enumerate(current_population):
            # Usando .get() para evitar erros caso uma chave não exista
            print(f"  - Prompt: \"{ind.get('prompt', 'N/A')}\" | F1: {ind.get('f1', 0.0):.4f} | Acc: {ind.get('acc', 0.0):.4f} | Tokens: {ind.get('tokens', 0)}")

        # Classifica a população inicial para obter os ranks
        initial_fronts = fast_non_dominated_sort(current_population)

        # Calcula a crowding distance para cada fronteira da população inicial
        print("[multi_evolution] Calculando crowding distance para a população inicial...")
        for front in initial_fronts:
            compute_crowding_distance(front)

        # O código abaixo, para salvar a fronteira de Geração 0
        if initial_fronts and initial_fronts[0]:
            save_pareto_front_data(initial_fronts[0], os.path.join(per_generation_pareto_log_dir, "pareto_gen_0.csv"), os.path.join(per_generation_pareto_log_dir, "pareto_gen_0.png"))

    # Passo 2: Ciclo de Gerações
    stagnation_counter = 0
    last_front_hash = None
    stagnation_limit = config.get("stagnation_limit", 3)

    for generation_num in range(current_generation, config["max_generations"]):
        current_gen_display = generation_num # Ajustado para corresponder ao número da geração
        logger.info(f"--- Geração {current_gen_display}---")

        offspring_prompts = generate_unique_offspring(current_population, config, evolution_type="multi")
        logger.info(f"Offspring gerados ({len(offspring_prompts)}):")
        for i, p in enumerate(offspring_prompts[:5]): # Loga os primeiros 5 offsprings
            logger.info(f"  {i+1}) '{p}'")
        if not offspring_prompts:
            logger.warning("Nenhum filho único foi gerado nesta geração.")
        
        evaluated_offspring = evaluate_population(offspring_prompts, dataset, config, executor_config)
        logger.info("Scores dos offspring avaliados:")
        for i, ind in enumerate(evaluated_offspring[:5]): # Loga os scores dos primeiros 5 offsprings avaliados
            print(f"  - Prompt: \"{ind.get('prompt', 'N/A')[:50]}...\" | F1: {ind.get('f1', 0.0):.4f} | Acc: {ind.get('acc', 0.0):.4f} | Tokens: {ind.get('tokens', 0)}")
        
        current_population = select_survivors_nsgaii(current_population, evaluated_offspring, population_size)
        print(f"[multi_evolution] Nova população selecionada. Tamanho: {len(current_population)}")

        current_pareto_front = [ind for ind in current_population if ind.get('rank') == 0]
        if current_pareto_front:
            save_pareto_front_data(
                current_pareto_front, 
                os.path.join(per_generation_pareto_log_dir, f"pareto_gen_{current_gen_display}.csv"),
                os.path.join(per_generation_pareto_log_dir, f"pareto_gen_{current_gen_display}.png")
            )
            
            front_prompts_tuple = tuple(sorted([ind['prompt'] for ind in current_pareto_front]))
            current_front_hash = hash(front_prompts_tuple)

            if current_front_hash == last_front_hash:
                stagnation_counter += 1
                logger.warning(f"Fronteira de Pareto não mudou. Estagnação: {stagnation_counter}/{stagnation_limit}")
            else:
                stagnation_counter = 0
                logger.info("Fronteira de Pareto evoluiu.")
            
            last_front_hash = current_front_hash

            if stagnation_counter >= stagnation_limit:
                logger.info(f"EVOLUÇÃO ESTAGNOU por {stagnation_limit} gerações. Interrompendo.")
                break
        else:
            logger.warning(f"Fronteira de Pareto da geração {current_gen_display} vazia.")

    # Fim do Ciclo Evolutivo
    logger.info("Evolução multiobjetivo concluída.")
    
    final_pareto_front = [ind for ind in current_population if ind.get('rank') == 0]
    
    save_pareto_front_data(final_pareto_front, output_csv_path, output_plot_path)

    logger.info(f"Resultados finais salvos em '{output_csv_path}' e gráfico em '{output_plot_path}'.")