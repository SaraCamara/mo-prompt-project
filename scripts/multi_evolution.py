# multi_evolution.py
import os
import pandas as pd
import random 
from utils import (
    evaluate_prompt, crossover_and_mutation_ga, 
    fast_non_dominated_sort, compute_crowding_distance,
    tournament_selection_multiobjective, save_pareto_front_data, 
)

def _transform_metrics_to_objectives(evaluated_population_raw):
    population_with_objectives = []
    for ind_raw in evaluated_population_raw:
        acc, f1, tokens, _ = ind_raw["metrics"]
        population_with_objectives.append({
            "prompt": ind_raw["prompt"],
            "acc": acc,
            "f1": f1, 
            "tokens": tokens
        })
    return population_with_objectives

def run_multi_evolution(config, dataset, initial_prompts_text, output_csv_path, output_plot_path):
    print("[multi_evolution] Iniciando execução da evolução multiobjetivo")

    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    population_size = config.get("population_size", 10) 
    k_tournament_parents = config.get("k_tournament_parents", 2)

    base_output_dir = config["base_output_dir"]

    generation_log_dir = os.path.join(base_output_dir, "generations_detail")
    os.makedirs(generation_log_dir, exist_ok=True)
    per_generation_pareto_log_dir = os.path.join(base_output_dir, "per_generation_pareto")
    os.makedirs(per_generation_pareto_log_dir, exist_ok=True)

    print(f"[multi_evolution] Avaliador: {evaluator_config['name']}")
    print(f"[multi_evolution] Estratégia: {strategy_config['name']}")
    
    # Passo 1: Avaliação da População Inicial (P_0)
    print("[multi_evolution] Avaliando população inicial")
    current_population_raw = [] 
    for i, p_text in enumerate(initial_prompts_text):
        print(f"[multi_evolution] Avaliando prompt inicial {i + 1}/{len(initial_prompts_text)}: \"{p_text[:100]}\"")
        metrics = evaluate_prompt(p_text, dataset, evaluator_config, strategy_config, config, base_output_dir)
        current_population_raw.append({"prompt": p_text, "metrics": metrics})

    # Transforma para o formato de indivíduo com objetivos nomeados
    current_population_individuals = _transform_metrics_to_objectives(current_population_raw)

    # Classifica e calcula crowding para a população inicial 
    fronts_initial = fast_non_dominated_sort(current_population_individuals)
    for front in fronts_initial:
        compute_crowding_distance(front)

    print(f"[multi_evolution] População inicial avaliada. Tamanho: {len(current_population_individuals)}")
    if fronts_initial and fronts_initial[0]:
        save_pareto_front_data(
            fronts_initial[0], 
            os.path.join(per_generation_pareto_log_dir, f"pareto_gen_0.csv"),
            os.path.join(per_generation_pareto_log_dir, f"pareto_gen_0.png")
        )

    # Ciclo de Gerações 
    stagnation_counter = 0
    last_front_hash = None
    stagnation_limit = config.get("stagnation_limit", 3) # Número de gerações sem mudança para parar
    
    for generation_num in range(config["max_generations"]):
        current_gen_display = generation_num + 1
        print(f"\n[multi_evolution] Geração {current_gen_display} iniciada.")

        # Geração de Filhos
        offspring_prompts_generated = []
        # Crie um conjunto (set) com os prompts existentes para busca rápida
        existing_prompts = {ind['prompt'] for ind in current_population_individuals}

        num_children_to_generate = population_size
        print(f"[multi_evolution] Gerando até {num_children_to_generate} filhos únicos...")

        # Adicione um limite de tentativas para evitar loops infinitos
        max_generation_attempts = num_children_to_generate * 3 
        attempts = 0

        while len(offspring_prompts_generated) < num_children_to_generate and attempts < max_generation_attempts:
            attempts += 1
            parent_pair = tournament_selection_multiobjective(current_population_individuals, k_tournament_parents, 2)
            # ... (lógica para garantir que parent_pair tenha 2 pais) ...

            child_dict_list = crossover_and_mutation_ga(parent_pair, config)
            
            if child_dict_list and "prompt" in child_dict_list[0] and not "erro_" in child_dict_list[0]["prompt"]:
                new_prompt = child_dict_list[0]["prompt"]
                
                # --- VERIFICAÇÃO DE DUPLICATAS ---
                if new_prompt not in existing_prompts:
                    offspring_prompts_generated.append(child_dict_list[0])
                    existing_prompts.add(new_prompt) # Adiciona ao conjunto para futuras verificações
                # else:
                    # print(f"[multi_evolution] [!] Filho duplicado descartado.")
            # else:
                # print(f"[multi_evolution] [!] Erro ao gerar filho. Descartado.")

        if len(offspring_prompts_generated) < num_children_to_generate:
            print(f"[multi_evolution] [!] Apenas {len(offspring_prompts_generated)} filhos únicos foram gerados após {attempts} tentativas.")

        # Certificar que a população tem indivíduos para seleção
        if not current_population_individuals:
            print("[multi_evolution] [!] População atual vazia. Não é possível selecionar pais. Encerrando evolução prematuramente.")
            break

        for i in range(num_children_to_generate):
            # Selecionar 2 pais usando torneio MOGA
            # `tournament_selection_multiobjective` espera indivíduos com 'rank' e 'crowding_distance'
            parent_pair = tournament_selection_multiobjective(current_population_individuals, k_tournament_parents, 2)
            if len(parent_pair) < 2:
                # print(f"[multi_evolution] [!] Não foi possível selecionar 2 pais para o filho {i+1}. Usando pais aleatórios da população.")
                if len(current_population_individuals) >= 2:
                    parent_pair = random.sample(current_population_individuals, 2)
                elif current_population_individuals: 
                    parent_pair = [current_population_individuals[0], current_population_individuals[0]] 
                else: 
                    print("[multi_evolution] [!] População exaurida. Parando de gerar filhos.")
                    break 
        
            # Aplicar Evo (crossover + mutação)
            # `crossover_and_mutation_ga` espera uma lista de 2 dicts de prompt e retorna lista com 1 dict de prompt
            child_dict_list = crossover_and_mutation_ga(parent_pair, config) 
            
            if child_dict_list and "prompt" in child_dict_list[0] and not "erro_" in child_dict_list[0]["prompt"]:
                offspring_prompts_generated.append(child_dict_list[0]) 
            else:
                error_val = child_dict_list[0]["prompt"] if child_dict_list and "prompt" in child_dict_list[0] else "erro_desconhecido_evo"
                # print(f"[multi_evolution] [!] Erro ao gerar filho {i+1} via Evo: {error_val}. Filho descartado.")
        
        # if not offspring_prompts_generated:
        #     print("[multi_evolution] [!] Nenhum filho gerado com sucesso nesta geração. A evolução pode estagnar.")
        # else:
        #     print(f"[multi_evolution] {len(offspring_prompts_generated)} filhos gerados. Avaliando filhos")

        # Avaliar os filhos gerados (Q_t)
        evaluated_offspring_raw = []
        for i, offspring_dict in enumerate(offspring_prompts_generated):
            p_text = offspring_dict["prompt"]
            metrics = evaluate_prompt(p_text, dataset, evaluator_config, strategy_config, config)
            evaluated_offspring_raw.append({"prompt": p_text, "metrics": metrics})
        evaluated_offspring_individuals = _transform_metrics_to_objectives(evaluated_offspring_raw)


        # Seleção de Sobreviventes (para P_{t+1})
        # Combinar população de pais (P_t) com população de filhos (Q_t) -> R_t
        combined_population_individuals = current_population_individuals + evaluated_offspring_individuals


        # Realizar classificação não-dominada em R_t
        fronts_combined = fast_non_dominated_sort(combined_population_individuals)

        next_population_individuals = []
        for front in fronts_combined:
            if not front: continue
            
            # Calcular crowding distance para a fronteira atual
            compute_crowding_distance(front)
            
            if len(next_population_individuals) + len(front) <= population_size:
                next_population_individuals.extend(front)
            else:
                num_needed = population_size - len(next_population_individuals)
                # Ordena a fronteira atual pela crowding distance (maior é melhor)
                front.sort(key=lambda x: x['crowding_distance'], reverse=True)
                next_population_individuals.extend(front[:num_needed])
                break 

        current_population_individuals = next_population_individuals 
        print(f"[multi_evolution] Nova população (P_{{t+1}}) selecionada. Tamanho: {len(current_population_individuals)}")

        # Salvar a fronteira de Pareto da Geração Atual (F_1 de P_{t+1})
        if current_population_individuals:
            current_pareto_front = [ind for ind in current_population_individuals if ind['rank'] == 0]
            if not current_pareto_front and fronts_combined and fronts_combined[0]: 
                current_pareto_front = fronts_combined[0]

            if current_pareto_front:
                save_pareto_front_data(
                    current_pareto_front, 
                    os.path.join(per_generation_pareto_log_dir, f"pareto_gen_{current_gen_display}.csv"),
                    os.path.join(per_generation_pareto_log_dir, f"pareto_gen_{current_gen_display}.png")
                )
            else:
                print(f"[multi_evolution] [!] Fronteira de Pareto da geração {current_gen_display} vazia ou não encontrada.")


        # Critério de parada 
        if generation_num == config["max_generations"] - 1:
            print(f"[multi_evolution] Critério de parada atingido (Máximo de {config['max_generations']} gerações).")
            break

    # Fim do ciclo evolutivo
    print("\n[multi_evolution] Evolução multiobjetivo concluída.")
    
    # A fronteira de Pareto final é a primeira fronteira da última `current_population_individuals`
    final_pareto_front = [ind for ind in current_population_individuals if ind['rank'] == 0]
    if not final_pareto_front and current_population_individuals: 
        final_fronts = fast_non_dominated_sort(current_population_individuals)
        final_pareto_front = final_fronts[0] if final_fronts and final_fronts[0] else []


    if final_pareto_front:
        # print(f"[multi_evolution] Fronteira de Pareto final com {len(final_pareto_front)} soluções.")
        save_pareto_front_data(final_pareto_front, output_csv_path, output_plot_path)
    else:
        # print("[multi_evolution] [!] Nenhuma solução na fronteira de Pareto final. Salvando arquivo vazio.")
        save_pareto_front_data([], output_csv_path, output_plot_path)

    print(f"\n[multi_evolution] [✓] Resultados finais salvos em '{output_csv_path}' e gráfico em '{output_plot_path}'.")

    if current_pareto_front:
    # Cria uma representação única da fronteira atual para comparar com a anterior
    # Ordena para garantir que a ordem não afete o hash
    front_prompts_tuple = tuple(sorted([ind['prompt'] for ind in current_pareto_front]))
    current_front_hash = hash(front_prompts_tuple)

    if current_front_hash == last_front_hash:
        stagnation_counter += 1
        print(f"[multi_evolution] [!] A fronteira de Pareto não mudou. Contador de estagnação: {stagnation_counter}/{stagnation_limit}")
    else:
        stagnation_counter = 0 # Reseta o contador se houver mudança
        print("[multi_evolution] A fronteira de Pareto evoluiu.")
    
    last_front_hash = current_front_hash

    if stagnation_counter >= stagnation_limit:
        print(f"\n[multi_evolution] A evolução estagnou por {stagnation_limit} gerações. Interrompendo.")
        break # Sai do loop de gerações
