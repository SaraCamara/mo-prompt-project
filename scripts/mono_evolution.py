import os
import pandas as pd
from utils import (
    evaluate_prompt, save_generation_results, 
    save_final_results, save_sorted_population,
    crossover_and_mutation_ga, roulette_wheel_selection
)

def run_mono_evolution(config, dataset, initial_prompts, output_csv_path): # output_plot removido
    print("[mono_evolution] Iniciando execução da evolução monoobjetivo")

    evaluator_config = config["evaluators"][0] # O avaliador selecionado
    strategy_config = config["strategies"][0] # A estratégia selecionada
    
    # Cria diretório para logs de avaliação detalhada se não existir
    eval_log_dir = "logs/evo/prompt_eval_logs"
    os.makedirs(eval_log_dir, exist_ok=True)
    
    # Cria diretório para logs de população por geração
    generation_log_dir = config.get("generation_log_dir", "logs/evo/generations_detail")
    os.makedirs(generation_log_dir, exist_ok=True)


    print(f"[mono_evolution] Avaliador: {evaluator_config['name']}")
    print(f"[mono_evolution] Estratégia: {strategy_config['name']}")
    
    # Passo 1: Avaliação da População Inicial
    print("[mono_evolution] Avaliando população inicial...")
    population = []
    for i, p_text in enumerate(initial_prompts):
        print(f"[mono_evolution] Avaliando prompt inicial {i + 1}/{len(initial_prompts)}: \"{p_text[:100]}...\"")
        # Passar experiment_settings (o config completo) para evaluate_prompt
        metrics = evaluate_prompt(p_text, dataset, evaluator_config, strategy_config, config)
        population.append({"prompt": p_text, "metrics": metrics})

    # Salva a população inicial avaliada e ordenada
    save_sorted_population(population, 0, base_log_path=generation_log_dir) # Geração 0

    # Ciclo de Gerações
    for generation in range(config["max_generations"]):
        current_generation_number = generation + 1
        print(f"\n[mono_evolution] Geração {current_generation_number} iniciada.")

        try:
            # Gerar filhos
            num_offspring_to_generate = config["population_size"]
            offspring_prompts_list_of_dicts = [] # Lista para armazenar os filhos gerados (dicts)

            print(f"[mono_evolution] Gerando {num_offspring_to_generate} descendentes...")
            for i in range(num_offspring_to_generate):
                # Passo: Seleção de 2 pais por Roleta para CADA filho
                # A população aqui deve ser a `population` da geração anterior.
                pair_of_parents = roulette_wheel_selection(population, num_parents_to_select=2)
                
                if len(pair_of_parents) < 2:
                    print(f"[mono_evolution] [!] Não foi possível selecionar 2 pais na iteração {i+1}. "
                        "Pode ser devido a uma população pequena ou sem fitness válida. Parando de gerar filhos para esta geração.")
                    # Se não puder selecionar pais, pode pular a geração de mais filhos ou
                    # usar alguma estratégia de fallback (ex: pegar aleatórios da população).
                    # Por ora, vamos parar de gerar se não conseguir pais.
                    break 
                
                # Passo: Crossover e Mutação (Evo(·))
                # crossover_and_mutation_ga recebe a lista de 2 pais e retorna uma lista com 1 filho (dict)
                child_dict_list = crossover_and_mutation_ga(pair_of_parents, config)
                
                if child_dict_list and "prompt" in child_dict_list[0] and not "erro_" in child_dict_list[0]["prompt"]:
                    offspring_prompts_list_of_dicts.append(child_dict_list[0])
                else:
                    # Logar o erro ou o prompt de erro
                    error_prompt_val = child_dict_list[0]["prompt"] if child_dict_list and "prompt" in child_dict_list[0] else "erro_desconhecido_na_geracao_filho"
                    print(f"[mono_evolution] [!] Erro ao gerar filho {i+1} via Evo(·): {error_prompt_val}")
                    # Adicionar um placeholder para manter o tamanho ou simplesmente não adicionar.
                    # Se adicionar um placeholder, ele precisa ser avaliável ou filtrado depois.
                    # Por enquanto, vamos apenas não adicionar o filho com erro.

            if not offspring_prompts_list_of_dicts:
                print("[mono_evolution] [!] Nenhum descendente foi gerado com sucesso nesta geração. Pulando para a próxima.")
                if generation == config["max_generations"] - 1:
                    print(f"[mono_evolution] Critério de parada atingido (Geração {current_generation_number}).")
                continue # Pula para a próxima iteração do loop de geração


            print(f"[mono_evolution] {len(offspring_prompts_list_of_dicts)} descendentes gerados com sucesso.")

            # Passo: Avaliação dos Filhos
            print("[mono_evolution] Avaliando descendentes...")
            evaluated_offspring = []
            for i, offspring_dict in enumerate(offspring_prompts_list_of_dicts):
                offspring_prompt_text = offspring_dict["prompt"]
                print(f"[mono_evolution] Avaliando descendente {i+1}/{len(offspring_prompts_list_of_dicts)}: \"{offspring_prompt_text[:100]}...\"")
                metrics = evaluate_prompt(offspring_prompt_text, dataset, evaluator_config, strategy_config, config)
                evaluated_offspring.append({"prompt": offspring_prompt_text, "metrics": metrics})
            
            print("[mono_evolution] Descendentes avaliados.")

            # Passo: Unir Pais e Filhos, Aplicar Elitismo e Ordenação
            # A `population` atual são os pais da geração anterior.
            combined_population = population + evaluated_offspring

            # Ordenar por Acurácia (metrics[0] desc) e depois Tokens (metrics[2] asc) para desempate
            combined_population.sort(key=lambda x: (x["metrics"][0], -x["metrics"][2] if len(x["metrics"]) >=3 else float('inf')), reverse=True)
            
            # Selecionar os melhores para a próxima geração (elitismo)
            population = combined_population[:config["population_size"]]
            print(f"[mono_evolution] População da próxima geração selecionada (Tamanho: {len(population)}). Melhor acurácia: {population[0]['metrics'][0]:.4f} se população não vazia.")

            # Salvar Resultados da Geração
            save_sorted_population(population, current_generation_number, base_log_path=generation_log_dir)
            # save_generation_results também salva de forma similar, pode ser redundante ou ter um propósito diferente
            # No momento, save_generation_results está configurado para salvar com nome de modelo e estratégia.
            save_generation_results(population, current_generation_number, config) 
            
            # Log dos prompts gerados (apenas os textos dos prompts) para análise
            # pd.DataFrame([p_d["prompt"] for p_d in offspring_prompts_list_of_dicts], columns=["generated_prompt_text"])\
            #    .to_csv(os.path.join(generation_log_dir, f"generated_texts_gen_{current_generation_number}.csv"), index=False)
        
        except Exception as e:
            print(f"[mono_evolution] [✗] Erro na geração {current_generation_number}: {e}")
            import traceback
            traceback.print_exc() # Para debug detalhado

        # Critério de Parada
        if generation == config["max_generations"] - 1: # -1 porque generation é 0-indexed
            print(f"[mono_evolution] Critério de parada atingido (Máximo de {config['max_generations']} gerações).")
            break # Sai do loop de gerações

    # Salvar Resultados Finais
    print("[mono_evolution] Salvando resultados finais...")
    # output_csv_path é o caminho final para os top_k resultados
    save_final_results(population, config, output_csv_path) 
    print(f"[mono_evolution] [✓] Evolução monoobjetivo concluída. Resultados salvos em {output_csv_path}")