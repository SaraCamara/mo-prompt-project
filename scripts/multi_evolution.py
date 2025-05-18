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
    """Converte o output de evaluate_prompt para o formato usado nas funções MOGA."""
    population_with_objectives = []
    for ind_raw in evaluated_population_raw:
        # metrics é (acc, f1, tokens, alert_message)
        acc, f1, tokens, _ = ind_raw["metrics"]
        population_with_objectives.append({
            "prompt": ind_raw["prompt"],
            "acc": acc,
            "f1": f1, 
            "tokens": tokens
            # 'rank' e 'crowding_distance' serão adicionados depois
        })
    return population_with_objectives

def run_multi_evolution(config, dataset, initial_prompts_text, output_csv_path, output_plot_path):
    print("[multi_evolution] Iniciando execução da evolução multiobjetivo")

    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    population_size = config.get("population_size", 10) # N
    k_tournament_parents = config.get("k_tournament_parents", 2) # Tamanho do torneio para selecionar pais

    # Cria diretórios de log 
    generation_log_dir = config.get("generation_log_dir_emo", "logs/emo/generations_detail")
    os.makedirs(generation_log_dir, exist_ok=True)
    # Log de cada geração (fronteira de Pareto da geração)
    per_generation_pareto_log_dir = os.path.join(os.path.dirname(output_csv_path), "per_generation_pareto")
    os.makedirs(per_generation_pareto_log_dir, exist_ok=True)


    print(f"[multi_evolution] Avaliador: {evaluator_config['name']}")
    print(f"[multi_evolution] Estratégia: {strategy_config['name']}")
    
    # Passo 1: Avaliação da População Inicial (P_0)
    print("[multi_evolution] Avaliando população inicial...")
    current_population_raw = [] # Lista de {"prompt": str, "metrics": tuple}
    for i, p_text in enumerate(initial_prompts_text):
        print(f"[multi_evolution] Avaliando prompt inicial {i + 1}/{len(initial_prompts_text)}: \"{p_text[:100]}...\"")
        metrics = evaluate_prompt(p_text, dataset, evaluator_config, strategy_config, config)
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


    # Ciclo de Gerações (t = 0, 1, ..., max_generations-1)
    for generation_num in range(config["max_generations"]):
        current_gen_display = generation_num + 1
        print(f"\n[multi_evolution] Geração {current_gen_display} iniciada.")

        # Geração de Filhos (Q_t)
        # Objetivo: gerar `population_size` filhos
        offspring_prompts_generated = [] # Lista de {"prompt": str}
        
        # Selecionar pais e gerar filhos até termos `population_size` filhos
        # A `current_population_individuals` já tem 'rank' e 'crowding_distance' da iteração anterior
        # ou da avaliação inicial.
        num_children_to_generate = population_size
        print(f"[multi_evolution] Gerando {num_children_to_generate} filhos...")

        # Certificar que a população tem indivíduos para seleção
        if not current_population_individuals:
            print("[multi_evolution] [!] População atual vazia. Não é possível selecionar pais. Encerrando evolução prematuramente.")
            break

        for i in range(num_children_to_generate):
            # Selecionar 2 pais usando torneio MOGA
            # `tournament_selection_multiobjective` espera indivíduos com 'rank' e 'crowding_distance'
            parent_pair = tournament_selection_multiobjective(current_population_individuals, k_tournament_parents, 2)
            if len(parent_pair) < 2:
                print(f"[multi_evolution] [!] Não foi possível selecionar 2 pais para o filho {i+1}. Usando pais aleatórios da população.")
                if len(current_population_individuals) >= 2:
                    parent_pair = random.sample(current_population_individuals, 2)
                elif current_population_individuals: # Apenas 1 indivíduo
                    parent_pair = [current_population_individuals[0], current_population_individuals[0]] # Crossover consigo mesmo
                else: # Impossível continuar
                    print("[multi_evolution] [!] População exaurida. Parando de gerar filhos.")
                    break 
        
            # Aplicar Evo(·) (crossover + mutação)
            # `crossover_and_mutation_ga` espera uma lista de 2 dicts de prompt e retorna lista com 1 dict de prompt
            child_dict_list = crossover_and_mutation_ga(parent_pair, config) 
            
            if child_dict_list and "prompt" in child_dict_list[0] and not "erro_" in child_dict_list[0]["prompt"]:
                offspring_prompts_generated.append(child_dict_list[0]) # Adiciona {"prompt": "texto do filho"}
            else:
                error_val = child_dict_list[0]["prompt"] if child_dict_list and "prompt" in child_dict_list[0] else "erro_desconhecido_evo"
                print(f"[multi_evolution] [!] Erro ao gerar filho {i+1} via Evo(·): {error_val}. Filho descartado.")
        
        if not offspring_prompts_generated:
            print("[multi_evolution] [!] Nenhum filho gerado com sucesso nesta geração. A evolução pode estagnar.")
            # Opcionalmente, pode-se pular para a próxima geração ou parar.
            # Por enquanto, vamos continuar, a população de pais ainda existe.
        else:
            print(f"[multi_evolution] {len(offspring_prompts_generated)} filhos gerados. Avaliando filhos...")

        # Avaliar os filhos gerados (Q_t)
        evaluated_offspring_raw = []
        for i, offspring_dict in enumerate(offspring_prompts_generated):
            p_text = offspring_dict["prompt"]
            # print(f"[multi_evolution] Avaliando filho {i + 1}/{len(offspring_prompts_generated)}: \"{p_text[:100]}...\"")
            metrics = evaluate_prompt(p_text, dataset, evaluator_config, strategy_config, config)
            evaluated_offspring_raw.append({"prompt": p_text, "metrics": metrics})
        
        evaluated_offspring_individuals = _transform_metrics_to_objectives(evaluated_offspring_raw)
        print(f"[multi_evolution] {len(evaluated_offspring_individuals)} filhos avaliados.")

        # Seleção de Sobreviventes (para P_{t+1})
        # Combinar população de pais (P_t) com população de filhos (Q_t) -> R_t
        # `current_population_individuals` são os pais (P_t)
        # `evaluated_offspring_individuals` são os filhos (Q_t)
        combined_population_individuals = current_population_individuals + evaluated_offspring_individuals
        print(f"[multi_evolution] População combinada (pais+filhos) tem {len(combined_population_individuals)} indivíduos.")

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
                # A fronteira atual não cabe inteira, precisa selecionar por crowding distance
                num_needed = population_size - len(next_population_individuals)
                # Ordena a fronteira atual pela crowding distance (maior é melhor)
                front.sort(key=lambda x: x['crowding_distance'], reverse=True)
                next_population_individuals.extend(front[:num_needed])
                break 

        current_population_individuals = next_population_individuals # Esta é P_{t+1}
        print(f"[multi_evolution] Nova população (P_{{t+1}}) selecionada. Tamanho: {len(current_population_individuals)}")

        # Salvar a fronteira de Pareto da Geração Atual (F_1 de P_{t+1})
        if current_population_individuals:
            # É preciso re-calcular a F_1 da nova população P_{t+1} ou pegar fronts_combined[0]
            # Se current_population_individuals foi formada corretamente, sua F_1 é fronts_combined[0] (ou parte dela)
            # Para ser preciso, vamos pegar a F_1 da população que vai para a próxima geração
            current_pareto_front = [ind for ind in current_population_individuals if ind['rank'] == 0]
            if not current_pareto_front and fronts_combined and fronts_combined[0]: # Fallback se F1 não estiver na pop truncada
                current_pareto_front = fronts_combined[0]

            if current_pareto_front:
                save_pareto_front_data(
                    current_pareto_front, 
                    os.path.join(per_generation_pareto_log_dir, f"pareto_gen_{current_gen_display}.csv"),
                    os.path.join(per_generation_pareto_log_dir, f"pareto_gen_{current_gen_display}.png")
                )
            else:
                print(f"[multi_evolution] [!] Fronteira de Pareto da geração {current_gen_display} vazia ou não encontrada.")


        # Critério de Parada (além do número de gerações, pode-se adicionar outros)
        if generation_num == config["max_generations"] - 1:
            print(f"[multi_evolution] Critério de parada atingido (Máximo de {config['max_generations']} gerações).")
            break

    # Fim do Ciclo Evolutivo

    print("\n[multi_evolution] Evolução multiobjetivo concluída.")
    
    # A fronteira de Pareto final é a primeira fronteira da última `current_population_individuals`
    final_pareto_front = [ind for ind in current_population_individuals if ind['rank'] == 0]
    if not final_pareto_front and current_population_individuals: 
        print("[multi_evolution] [!] A fronteira de Pareto (rank 0) não está na população final. Recalculando F1 da população final.")
        # Recalcula F1 da população final
        final_fronts = fast_non_dominated_sort(current_population_individuals)
        final_pareto_front = final_fronts[0] if final_fronts and final_fronts[0] else []


    if final_pareto_front:
        print(f"[multi_evolution] Fronteira de Pareto final com {len(final_pareto_front)} soluções.")
        save_pareto_front_data(final_pareto_front, output_csv_path, output_plot_path)
    else:
        print("[multi_evolution] [!] Nenhuma solução na fronteira de Pareto final. Salvando arquivo vazio.")
        save_pareto_front_data([], output_csv_path, output_plot_path)

    print(f"\n[multi_evolution] [✓] Resultados finais salvos em '{output_csv_path}' e gráfico em '{output_plot_path}'.")