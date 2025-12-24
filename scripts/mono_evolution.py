# mono_evolution.py
import os
import pandas as pd
import random
from utils import (
    evaluate_population, generate_unique_offspring, save_generation_results,
    save_final_results, save_sorted_population
)

def run_mono_evolution(config, dataset, initial_prompts, output_csv_path, start_generation=0, initial_population=None):
    print("[mono_evolution] Iniciando execução da evolução mono-objetivo")

    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    population_size = config.get("population_size", 10)

    # Configuração de Caminhos
    base_output_dir = config["base_output_dir"]
    generation_log_dir = os.path.join(base_output_dir, "generations_detail")
    os.makedirs(generation_log_dir, exist_ok=True)

    print(f"[mono_evolution] Avaliador: {evaluator_config['name']}")
    print(f"[mono_evolution] Estratégia: {strategy_config['name']}")

    population = []
    current_generation = start_generation

    if initial_population:
        print(f"\n[mono_evolution] Retomando execução da Geração {start_generation - 1} com população carregada.")
        population = initial_population
    else:
        # Passo 1: Avaliação da População Inicial
        print("\n[mono_evolution] Avaliando população inicial.")
        population = evaluate_population(initial_prompts, dataset, config, evaluator_config)
        save_sorted_population(population, 0, generation_log_dir)

    # Ciclo de Gerações (ajustado para retomar)
    for generation in range(current_generation, config["max_generations"]):
        current_generation_number = generation
        print(f"\n[mono_evolution]--- Geração {current_generation_number}---")

        try:
            # Geração de Filhos usando a função genérica do utils
            offspring_prompts_list_of_dicts = generate_unique_offspring(
                population,
                config,
                evolution_type="mono"
            )
            
            # offspring_prompts_list_of_dicts will be a list of strings here
            if not offspring_prompts_list_of_dicts:
                print("[mono_evolution] [!] Nenhum descendente único foi gerado. Pulando para a próxima geração.")
                continue

            print(f"[mono_evolution] {len(offspring_prompts_list_of_dicts)} descendentes gerados. Avaliando.")

            # Avaliação dos Filhos
            evaluated_offspring = evaluate_population(offspring_prompts_list_of_dicts, dataset, config, evaluator_config)
            
            # Seleção de Sobreviventes
            combined_population = population + evaluated_offspring
            # F1_Score
            combined_population.sort(key=lambda x: (x["metrics"][1], -x["metrics"][2] if len(x["metrics"]) >= 3 else float('inf')), reverse=True)
            
            population = combined_population[:population_size]
            
            if population:
                print(f"[mono_evolution] População da próxima geração selecionada (Tamanho: {len(population)}). Melhor f1_score: {population[0]['metrics'][1]:.4f}")
            else:
                print("[mono_evolution] [!] População ficou vazia após seleção.")


            # Salvando Resultados da Geração
            save_sorted_population(population, current_generation_number, generation_log_dir)
            save_generation_results(population, current_generation_number, config, generation_log_dir) 
        
        except Exception as e:
            print(f"[mono_evolution] [✗] Erro na geração {current_generation_number}: {e}")
            import traceback
            traceback.print_exc()
            continue 

    # Fim do Ciclo Evolutivo
    print("\n[mono_evolution] Evolução mono-objetivo concluída.")
    print("[mono_evolution] Salvando resultados finais.")
    save_final_results(population, config, output_csv_path) 
    print(f"[mono_evolution] [✓] Resultados salvos em {output_csv_path}")