# mono_evolution.py
import os
import pandas as pd
import random
from utils import (
    evaluate_prompt, save_generation_results,
    save_final_results, save_sorted_population,
    crossover_and_mutation_ga, roulette_wheel_selection
)

def run_mono_evolution(config, dataset, initial_prompts, output_csv_path):
    print("[mono_evolution] Iniciando execução da evolução mono-objetivo")

    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    population_size = config.get("population_size", 10)

    # Configuração de Caminhos
    base_output_dir = config["base_output_dir"]
    eval_log_dir = os.path.join(base_output_dir, "prompt_eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)
    generation_log_dir = os.path.join(base_output_dir, "generations_detail")
    os.makedirs(generation_log_dir, exist_ok=True)

    print(f"[mono_evolution] Avaliador: {evaluator_config['name']}")
    print(f"[mono_evolution] Estratégia: {strategy_config['name']}")

    # Passo 1: Avaliação da População Inicial
    print("\n[mono_evolution] Avaliando população inicial...")
    population = []
    for i, p_text in enumerate(initial_prompts):
        print(f"[mono_evolution] Avaliando prompt inicial {i + 1}/{len(initial_prompts)}: \"{p_text[:100]}...\"")
        metrics = evaluate_prompt(p_text, dataset, evaluator_config, strategy_config, config, eval_log_dir)
        population.append({"prompt": p_text, "metrics": metrics})

    save_sorted_population(population, 0, generation_log_dir)  # Geração 0

    # Ciclo de Gerações
    for generation in range(config["max_generations"]):
        current_generation_number = generation + 1
        print(f"\n[mono_evolution]--- Geração {current_generation_number}---")

        try:
            # Geração de Filhos com Verificação de Duplicatas
            offspring_prompts_list_of_dicts = []
            existing_prompts = {ind['prompt'] for ind in population}
            
            print(f"[mono_evolution] Gerando até {population_size} descendentes únicos...")
            max_attempts = population_size * 3
            attempts = 0

            while len(offspring_prompts_list_of_dicts) < population_size and attempts < max_attempts:
                attempts += 1
                if len(population) < 2:
                    print("[mono_evolution] [!] População de pais insuficiente. Parando de gerar filhos.")
                    break
                
                parent_pair = roulette_wheel_selection(population, num_parents_to_select=2)
                
                if len(parent_pair) < 2:
                    # print(f"[mono_evolution] [!] Não foi possível selecionar 2 pais. Tentando novamente.")
                    continue 
                
                child_dict_list = crossover_and_mutation_ga(parent_pair, config)
                
                if child_dict_list and "prompt" in child_dict_list[0] and "erro_" not in child_dict_list[0]["prompt"]:
                    new_prompt = child_dict_list[0]["prompt"]
                    if new_prompt not in existing_prompts:
                        offspring_prompts_list_of_dicts.append({"prompt": new_prompt})
                        existing_prompts.add(new_prompt)
                # else:
                    # print("[mono_evolution] [!] Erro ou falha na geração do filho.")

            if not offspring_prompts_list_of_dicts:
                print("[mono_evolution] [!] Nenhum descendente único foi gerado. Pulando para a próxima geração.")
                continue

            print(f"[mono_evolution] {len(offspring_prompts_list_of_dicts)} descendentes gerados. Avaliando...")

            # Avaliação dos Filhos
            evaluated_offspring = []
            for i, offspring_dict in enumerate(offspring_prompts_list_of_dicts):
                offspring_prompt_text = offspring_dict["prompt"]
                metrics = evaluate_prompt(offspring_prompt_text, dataset, evaluator_config, strategy_config, config, eval_log_dir)
                evaluated_offspring.append({"prompt": offspring_prompt_text, "metrics": metrics})
            
            # Seleção de Sobreviventes
            combined_population = population + evaluated_offspring
            combined_population.sort(key=lambda x: (x["metrics"][0], -x["metrics"][2] if len(x["metrics"]) >= 3 else float('inf')), reverse=True)
            
            population = combined_population[:population_size]
            
            if population:
                print(f"[mono_evolution] População da próxima geração selecionada (Tamanho: {len(population)}). Melhor acurácia: {population[0]['metrics'][0]:.4f}")
            else:
                print("[mono_evolution] [!] População ficou vazia após seleção.")


            # Salvando Resultados da Geração
            save_sorted_population(population, current_generation_number, generation_log_dir)
            save_generation_results(population, current_generation_number, config, generation_log_dir) 
        
        except Exception as e:
            print(f"[mono_evolution] [✗] Erro na geração {current_generation_number}: {e}")
            import traceback
            traceback.print_exc()
            continue # Continua para a próxima geração em caso de erro

    # Fim do Ciclo Evolutivo
    print("\n[mono_evolution] Evolução mono-objetivo concluída.")
    print("[mono_evolution] Salvando resultados finais...")
    save_final_results(population, config, output_csv_path) 
    print(f"[mono_evolution] [✓] Resultados salvos em {output_csv_path}")