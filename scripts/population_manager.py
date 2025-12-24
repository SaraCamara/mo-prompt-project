import os
import logging
from .prompt_evaluator import evaluate_prompt # type: ignore
from .selection_algorithms import roulette_wheel_selection, tournament_selection_multiobjective # type: ignore
from .evolutionary_operators import crossover_and_mutation_ga, mop_crossover_and_mutation_ga # type: ignore
from .nsga2_algorithms import fast_non_dominated_sort, compute_crowding_distance # type: ignore

# Seção: Funções Auxiliares de Evolução (Gerenciamento de População)
logger = logging.getLogger(__name__)

def evaluate_population(prompts_to_evaluate, dataset, config, executor_config):
    evaluated_population = []
    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    base_output_dir = config["base_output_dir"]
    eval_log_dir = os.path.join(base_output_dir, "prompt_eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)
    
    # Garante que prompts_to_evaluate é uma lista de strings
    prompt_list = []
    for p in prompts_to_evaluate:
        if isinstance(p, dict) and "prompt" in p:
            prompt_list.append(p["prompt"])
        elif isinstance(p, str):
            prompt_list.append(p)
        else:
            logger.warning(f"Formato de prompt inesperado encontrado: {p}. Ignorando.")

    logger.info(f"Iniciando avaliação de {len(prompt_list)} prompts.")
    for p_text in prompt_list:
        metrics = evaluate_prompt(p_text, dataset, executor_config, strategy_config, config, eval_log_dir)
        acc, f1, tokens, _ = metrics
        evaluated_population.append({"prompt": p_text, "acc": acc, "f1": f1, "tokens": tokens, "metrics": metrics})
    return evaluated_population


def generate_unique_offspring(current_population, config, evolution_type="mono"):
    """
    Gera uma nova população de descendentes únicos a partir da população atual,
    utilizando funções de seleção e crossover/mutação apropriadas para o tipo de evolução.

    Args:
        current_population (list): A população atual de indivíduos.
        config (dict): Dicionário de configurações do experimento.
        evolution_type (str): "mono" para mono-objetivo ou "multi" para multi-objetivo.

    Returns:
        list: Uma lista de strings, onde cada string é um prompt de um descendente único.
    """
    offspring_prompts_dicts = []
    existing_prompts = {ind['prompt'] for ind in current_population}
    population_size = config.get("population_size", 10)
    num_parents_for_crossover = 2 # Geralmente 2 pais para crossover
    max_attempts = population_size * 3
    attempts = 0
    
    logger.info(f"Gerando até {population_size} descendentes únicos ({evolution_type}-objetivo).")

    while len(offspring_prompts_dicts) < population_size and attempts < max_attempts:
        attempts += 1
        if len(current_population) < num_parents_for_crossover:
            logger.warning(f"[Offspring {evolution_type.capitalize()}] População atual menor que {num_parents_for_crossover}, não é possível gerar offspring.")
            break
        
        parent_pair = []
        crossover_mutation_func = None

        if evolution_type == "mono":
            parent_pair = roulette_wheel_selection(current_population, num_parents_for_crossover)
            crossover_mutation_func = crossover_and_mutation_ga
        elif evolution_type == "multi":
            k_tournament_parents = config.get("k_tournament_parents", 2)
            parent_pair = tournament_selection_multiobjective(current_population, k_tournament_parents, num_parents_for_crossover)
            crossover_mutation_func = mop_crossover_and_mutation_ga
        else: # type: ignore
            logger.error(f"[Offspring] Tipo de evolução desconhecido: {evolution_type}")
            break

        if len(parent_pair) < num_parents_for_crossover:
            continue 
        
        child_dict_list = crossover_mutation_func(parent_pair, config)
        
        if child_dict_list and "prompt" in child_dict_list[0] and "erro_" not in child_dict_list[0]["prompt"]:
            new_prompt = child_dict_list[0]["prompt"]
            if new_prompt not in existing_prompts:
                offspring_prompts_dicts.append({"prompt": new_prompt})
                existing_prompts.add(new_prompt)

    if len(offspring_prompts_dicts) < population_size:
        logger.warning(f"Apenas {len(offspring_prompts_dicts)} filhos únicos foram gerados ({evolution_type}-objetivo).")
    
    return [p["prompt"] for p in offspring_prompts_dicts] # Retorna uma lista de strings


def select_survivors_nsgaii(parent_population, offspring_population, population_size):
    combined_population = parent_population + offspring_population
    all_fronts = fast_non_dominated_sort(combined_population)
    next_generation = []
    for front in all_fronts:
        if not front: continue
        compute_crowding_distance(front)
        if len(next_generation) + len(front) <= population_size:
            next_generation.extend(front)
        else:
            num_needed = population_size - len(next_generation)
            front.sort(key=lambda x: x['crowding_distance'], reverse=True)
            next_generation.extend(front[:num_needed])
            break
    return next_generation