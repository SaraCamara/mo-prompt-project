# Seção: Lógica NSGA-II
import logging
logger = logging.getLogger(__name__)

def dominates(ind_a_objectives, ind_b_objectives):
    """
    Verifica se o indivíduo A domina o indivíduo B.
    Assume que 'f1' deve ser maximizado e 'tokens' minimizado.
    """
    a_is_better_or_equal = (ind_a_objectives["f1"] >= ind_b_objectives["f1"] and ind_a_objectives["tokens"] <= ind_b_objectives["tokens"])
    a_is_strictly_better = (ind_a_objectives["f1"] > ind_b_objectives["f1"] or ind_a_objectives["tokens"] < ind_b_objectives["tokens"])
    return a_is_better_or_equal and a_is_strictly_better


def fast_non_dominated_sort(population_with_objectives):
    """
    Implementa o algoritmo Fast Non-Dominated Sort para classificar a população em fronteiras.
    """
    fronts = [[]] # F_0
    for p_ind in population_with_objectives:
        p_ind['dominated_solutions_indices'] = [] # Lista de índices de soluções dominadas por p
        p_ind['domination_count'] = 0 # Número de soluções que dominam p
        for q_idx, q_ind in enumerate(population_with_objectives):
            if p_ind == q_ind: continue # Não comparar com ele mesmo
            if dominates(p_ind, q_ind):
                p_ind['dominated_solutions_indices'].append(q_idx)
            elif dominates(q_ind, p_ind):
                p_ind['domination_count'] += 1
        
        # Se p não é dominado por ninguém, ele pertence à primeira fronteira (F_0)
        if p_ind['domination_count'] == 0:
            p_ind['rank'] = 0
            fronts[0].append(p_ind)
    
    current_rank_idx = 0
    while fronts[current_rank_idx]:
        next_front_individuals = []
        for p_ind in fronts[current_rank_idx]:
            for q_idx in p_ind['dominated_solutions_indices']:
                # Encontra a referência original do indivíduo q
                q_ind_ref = population_with_objectives[q_idx] 
                q_ind_ref['domination_count'] -= 1
                if q_ind_ref['domination_count'] == 0:
                    q_ind_ref['rank'] = current_rank_idx + 1
                    next_front_individuals.append(q_ind_ref)
        
        current_rank_idx += 1
        if next_front_individuals:
            fronts.append(next_front_individuals)
        else:
            break # Não há mais indivíduos para formar novas fronteiras
    return fronts


def compute_crowding_distance(front_individuals):
    """
    Calcula a crowding distance para os indivíduos em uma fronteira específica.
    """
    if not front_individuals: return []
    num_individuals = len(front_individuals)
    for ind in front_individuals: ind['crowding_distance'] = 0.0
    objectives = {'f1': True, 'tokens': False} # f1 a maximizar, tokens a minimizar
    for obj_key, maximize in objectives.items():
        sorted_front = sorted(front_individuals, key=lambda x: x[obj_key])
        sorted_front[0]['crowding_distance'] = float('inf') # Pontos extremos têm distância infinita
        if num_individuals > 1: sorted_front[num_individuals - 1]['crowding_distance'] = float('inf')
        min_obj_val = sorted_front[0][obj_key]
        max_obj_val = sorted_front[num_individuals - 1][obj_key]
        range_obj = max_obj_val - min_obj_val
        if range_obj == 0: continue # Evita divisão por zero
        for i in range(1, num_individuals - 1):
            if sorted_front[i]['crowding_distance'] != float('inf'): # Não sobrescreve infinito
                sorted_front[i]['crowding_distance'] += (sorted_front[i+1][obj_key] - sorted_front[i-1][obj_key]) / range_obj
    return front_individuals