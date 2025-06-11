# multi_evolution.py
import os
import random 
from utils import (
    evaluate_population,      # Função refatorada
    generate_unique_offspring, # Função refatorada
    select_survivors_nsgaii,   # Função refatorada
    fast_non_dominated_sort,
    save_pareto_front_data
)

def run_multi_evolution(config, dataset, initial_prompts_text, output_csv_path, output_plot_path):
    print("[multi_evolution] Iniciando execução da evolução multiobjetivo")

    # Configuração Inicial
    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    population_size = config.get("population_size", 10)
    base_output_dir = config["base_output_dir"]
    per_generation_pareto_log_dir = os.path.join(base_output_dir, "per_generation_pareto")
    os.makedirs(per_generation_pareto_log_dir, exist_ok=True)
    
    print(f"[multi_evolution] Avaliador: {evaluator_config['name']}")
    print(f"[multi_evolution] Estratégia: {strategy_config['name']}")
    
    # Passo 1: População Inicial (P_0)
    print("\n[multi_evolution] Avaliando população inicial...")
    current_population = evaluate_population(initial_prompts_text, dataset, config)
    print(f"[multi_evolution] População inicial avaliada. Tamanho: {len(current_population)}")
    
    initial_fronts = fast_non_dominated_sort(current_population)
    if initial_fronts and initial_fronts[0]:
        save_pareto_front_data(
            initial_fronts[0], 
            os.path.join(per_generation_pareto_log_dir, "pareto_gen_0.csv"),
            os.path.join(per_generation_pareto_log_dir, "pareto_gen_0.png")
        )

    # Passo 2: Ciclo de Gerações
    stagnation_counter = 0
    last_front_hash = None
    stagnation_limit = config.get("stagnation_limit", 3)

    for generation_num in range(config["max_generations"]):
        current_gen_display = generation_num + 1
        print(f"\n[multi_evolution]--- Geração {current_gen_display}---")

        offspring_prompts = generate_unique_offspring(current_population, config)
        if not offspring_prompts:
            print("[multi_evolution] [!] Nenhum filho único foi gerado nesta geração.")
        
        evaluated_offspring = evaluate_population(offspring_prompts, dataset, config)
        
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
                print(f"[multi_evolution] [!] Fronteira de Pareto não mudou. Estagnação: {stagnation_counter}/{stagnation_limit}")
            else:
                stagnation_counter = 0
                print("[multi_evolution] Fronteira de Pareto evoluiu.")
            
            last_front_hash = current_front_hash

            if stagnation_counter >= stagnation_limit:
                print(f"\n[multi_evolution] EVOLUÇÃO ESTAGNOU por {stagnation_limit} gerações. Interrompendo.")
                break
        else:
            print(f"[multi_evolution] [!] Fronteira de Pareto da geração {current_gen_display} vazia.")

    # Fim do Ciclo Evolutivo
    print("\n[multi_evolution] Evolução multiobjetivo concluída.")
    
    final_pareto_front = [ind for ind in current_population if ind.get('rank') == 0]
    
    save_pareto_front_data(final_pareto_front, output_csv_path, output_plot_path)

    print(f"\n[multi_evolution] [✓] Resultados finais salvos em '{output_csv_path}' e gráfico em '{output_plot_path}'.")