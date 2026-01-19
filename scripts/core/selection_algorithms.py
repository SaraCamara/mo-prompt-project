import random
import numpy as np
import logging

# Seção: Algoritmos de Seleção
logger = logging.getLogger(__name__)

def roulette_wheel_selection(population, num_parents_to_select):
    if not population:
        logger.warning("População vazia para seleção por roleta.")
        return []
    valid_individuals = [ind for ind in population if "metrics" in ind and ind["metrics"][0] >= 0]
    if not valid_individuals:
        logger.warning("Nenhum indivíduo com fitness válida para seleção.")
        return random.sample(population, min(num_parents_to_select, len(population)))
    fitness_values = [ind["metrics"][0] for ind in valid_individuals]
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return random.sample(valid_individuals, min(num_parents_to_select, len(valid_individuals)))
    probabilities = [f / total_fitness for f in fitness_values]
    num_to_sample = min(num_parents_to_select, len(valid_individuals))
    selected_indices = np.random.choice(len(valid_individuals), size=num_to_sample, p=probabilities, replace=False)
    return [valid_individuals[i] for i in selected_indices]


def tournament_selection_multiobjective(population_with_rank_and_crowding, k_tournament_size, num_to_select):
    selected_parents = []
    population_size = len(population_with_rank_and_crowding)
    if population_size == 0: return []
    if k_tournament_size > population_size: k_tournament_size = population_size
    for _ in range(num_to_select):
        tournament_candidates_indices = random.sample(range(population_size), k_tournament_size)
        tournament_candidates = [population_with_rank_and_crowding[i] for i in tournament_candidates_indices]
        best_candidate = tournament_candidates[0]
        for i in range(1, k_tournament_size):
            candidate = tournament_candidates[i]
            # Prioriza menor rank (melhor fronteira)
            if candidate['rank'] < best_candidate['rank']:
                best_candidate = candidate
            # Se ranks iguais, prioriza maior crowding distance (maior diversidade)
            elif candidate['rank'] == best_candidate['rank']:
                if candidate['crowding_distance'] > best_candidate['crowding_distance']:
                    best_candidate = candidate
        selected_parents.append(best_candidate)
    return selected_parents