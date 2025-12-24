# main.py
import subprocess
import sys
import os
import re
import yaml
import json
import pandas as pd
from mono_evolution import run_mono_evolution
from multi_evolution import run_multi_evolution
from utils import load_credentials_from_yaml, load_settings, load_dataset, load_initial_prompts, load_population_for_resumption, install_requirements, get_validated_numerical_input

if __name__ == "__main__":
    install_requirements()
    credentials = load_credentials_from_yaml("config/credentials.yaml")
    if not credentials:
        print("[main] ERRO FATAL: Falha ao carregar 'credentials.yaml'. Encerrando.")
        sys.exit(1)
    #print("[main] Credenciais carregadas.")

    config = load_settings("config/experiment_settings.yaml", credentials)
    if not config:
        print("[main] ERRO FATAL: Falha ao carregar 'experiment_settings.yaml'. Encerrando.")
        sys.exit(1)
    #print(f"[main] Configurações principais carregadas e processadas: {json.dumps(config, indent=2)}")


    # Seleção de dados para tarefa (IMDB/SQuAD)
    print("\n\n[main] [>] Selecione a tarefa a ser executada:\n  0) Análise de Sentimentos - IMDB-PT \n  1) Perguntas e Respostas - SQuAD-PT")    
    task_choice = get_validated_numerical_input("Digite o número da opção desejada (0 ou 1): ", 2)
    is_squad = (task_choice == 1)
    task_name = "squad" if is_squad else "imdb"
    config["task"] = task_name
    print(f"[main] Tarefa selecionada: {task_name.upper()}")

    # Carrega as configurações específicas da tarefa
    config["dataset_path"] = config[f"dataset_path_{task_name}"]
    config["generator"] = config[f"generator_{task_name}"]
    config["strategies"] = config[f"strategies_{task_name}"]
    
    
    # Carregamento do Dataset
    df_sample = load_dataset(config)
    if df_sample is None:
        print("[main] ERRO FATAL: Falha ao carregar o dataset. Encerrando.")
        sys.exit(1)

    # Carregamento da População Inicial
    print("\n\n[main] Carregamento da população inicial")
    prompts_path = f"data/initial_prompts_{task_name}.txt"
    initial_prompts = load_initial_prompts(prompts_path)
    if not initial_prompts:
        print(f"[main] ERRO FATAL: Falha ao carregar prompts iniciais de '{prompts_path}'. Encerrando.")
        sys.exit(1)


    # Seleção de Modo (Mono/Multi)
    print("\n\n[main] [>] Selecione estratégia de otimização:\n  0) Mono-objetivo\n  1) Multiobjetivo")
    optimization_type_choice = get_validated_numerical_input("Digite o número da opção desejada (0 ou 1): ", 2)
    is_multiobjective = (optimization_type_choice == 1)
    config["objective"] = "multiobjetivo" if is_multiobjective else "mono-objetivo"
    print(f"[main] Tipo de otimização selecionado: [{config['objective'].capitalize()}]")


    # Seleção de Avaliador
    print("\n\n[main] [>] Selecione do modelo avaliador:")
    available_evaluators = config.get("evaluators", [])
    if not available_evaluators:
        print("[main] Nenhum avaliador definido em 'experiment_settings.yaml'. Encerrando.")
        sys.exit(1)

    for i, evaluator_config in enumerate(available_evaluators):
        print(f"  {i}) {evaluator_config.get('name', 'Avaliador Desconhecido')}")

    evaluator_choice_idx = get_validated_numerical_input("Digite o número da opção desejada: ", len(available_evaluators))
    selected_evaluator_config = available_evaluators[evaluator_choice_idx]
    evaluator_name = selected_evaluator_config.get("name", "unknown_model")
    print(f"[main] Avaliador selecionado: {evaluator_name}")
    config["evaluators"] = [selected_evaluator_config]

    # Extrai o nome base do modelo para usar no caminho do diretório
    parts = re.split(r'[:/_-]', evaluator_name)
    output_model_name = parts[0]


    # Seleção de Estratégia
    print("\n\n[main] [>] Selecione da estratégia de prompt:")
    available_strategies = config.get("strategies", [])
    if not available_strategies:
        print("[main] Nenhuma estratégia definida em 'experiment_settings.yaml'. Encerrando.")
        sys.exit(1)

    for i, strategy_config in enumerate(available_strategies):
        print(f"  {i}) {strategy_config.get('name', 'Estratégia Desconhecida')}")

    strategy_choice_idx = get_validated_numerical_input("Digite o número da opção desejada: ", len(available_strategies))
    selected_strategy_config = available_strategies[strategy_choice_idx]
    strategy_name = selected_strategy_config["name"]
    print(f"[main] Estratégia de prompt selecionada: {strategy_name}")
    config["strategies"] = [selected_strategy_config]


    # Configuração de Caminhos de Saída 
    print("\n\n[main] Configurando diretório de saída para o experimento...")
    objective_path_name = "mop" if is_multiobjective else "evo"

    base_output_dir = os.path.join("logs", task_name, objective_path_name, output_model_name, strategy_name)
    print(f"[main] Todos os resultados e logs para esta execução serão salvos em: '{base_output_dir}'")
    os.makedirs(base_output_dir, exist_ok=True)

    config["base_output_dir"] = base_output_dir

    output_csv = os.path.join(base_output_dir, "final_results.csv")
    output_plot = os.path.join(base_output_dir, "final_pareto_front.png") if is_multiobjective else ""

    print(f"[main] Caminhos configurados:\n - CSV: {output_csv}")
    if output_plot:
        print(f" - Plot: {output_plot}")
        
    # Lógica para iniciar ou retomar
    print("\n\n[main] [>] Deseja retomar uma execução anterior?")
    print("  0) Iniciar nova execução")
    print("  1) Retomar de uma geração específica")
    resume_choice = get_validated_numerical_input("Digite o número da opção desejada (0 ou 1): ", 2)

    start_generation = 0
    loaded_population = None

    if resume_choice == 1:
        while True:
            try:
                resume_gen_input = input("De qual geração você deseja retomar? (Ex: 5): ")
                resume_from_generation = int(resume_gen_input)
                if resume_from_generation >= 0:
                    loaded_population, next_gen_num = load_population_for_resumption(resume_from_generation, base_output_dir, is_multiobjective)
                    if loaded_population is not None:
                        start_generation = next_gen_num
                        print(f"[main] Retomando da Geração {resume_from_generation}. Próxima geração será {start_generation}.")
                        break
                    else:
                        print(f"[main] Não foi possível carregar a população da Geração {resume_from_generation}. Por favor, verifique o diretório '{base_output_dir}'.")
                        print("Deseja tentar outra geração ou iniciar uma nova execução? (s/n para tentar outra, qualquer outra tecla para nova execução)")
                        retry_input = input().lower()
                        if retry_input != 's':
                            print("[main] Iniciando nova execução.")
                            break 
            except ValueError:
                print("[main] Entrada inválida. Por favor, insira um número inteiro.")
    else:
        print("[main] Iniciando nova execução.")

    # Execução do Algoritmo
    print("\n\n[main] [>] Iniciando execução do algoritmo evolutivo.\n")

    if is_multiobjective:
        run_multi_evolution(config, df_sample, initial_prompts, output_csv, output_plot,
                            start_generation=start_generation, initial_population=loaded_population)
    else:
        run_mono_evolution(config, df_sample, initial_prompts, output_csv,
                            start_generation=start_generation, initial_population=loaded_population)

    print(f"\n[main] Execução finalizada. Resultados disponíveis em:\n - {output_csv}\n")

