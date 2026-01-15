# main.py
"""Ponto de entrada principal para o mo-prompt-project."""
import os
import re
import sys
import logging

from .mono_evolution import run_mono_evolution
from .multi_evolution import run_multi_evolution
from .cli_interface import select_from_menu, confirm_action, print_header, print_config_summary
from .config_data_loader import (
    load_credentials_from_yaml, load_settings, load_dataset,
    load_initial_prompts, load_population_for_resumption
)
from .logger_config import setup_logging


def setup_experiment_config(config: dict) -> dict:
    """Configura o experimento através de seleções interativas."""
    
    # 1. Seleção de Tarefa
    tasks = [
        {"name": "Análise de Sentimentos - IMDB-PT", "key": "imdb"},
        {"name": "Perguntas e Respostas - SQuAD-PT", "key": "squad"}
    ]
    _, task = select_from_menu("Selecione a Tarefa", tasks)
    task_name = task["key"]
    config["task"] = task_name
    
    # Carrega configurações específicas da tarefa
    config["dataset_path"] = config[f"dataset_path_{task_name}"]
    config["generator"] = config[f"generator_{task_name}"]
    config["strategies"] = config[f"strategies_{task_name}"]
    
    # 2. Seleção de Modo de Otimização
    modes = [
        {"name": "Mono-objetivo (maximiza F1)", "key": "mono"},
        {"name": "Multiobjetivo (F1 vs Tokens)", "key": "multi"}
    ]
    _, mode = select_from_menu("Modo de Otimização", modes)
    is_multiobjective = mode["key"] == "multi"
    config["objective"] = "multiobjetivo" if is_multiobjective else "mono-objetivo"
    
    # 3. Seleção de Avaliador
    evaluators = config.get("evaluators", [])
    if not evaluators:
        raise ValueError("Nenhum avaliador definido em experiment_settings.yaml")
    
    _, evaluator = select_from_menu("Modelo Avaliador", evaluators)
    config["evaluators"] = [evaluator]
    
    # 4. Seleção de Estratégia
    strategies = config.get("strategies", [])
    if not strategies:
        raise ValueError("Nenhuma estratégia definida para a tarefa")
    
    _, strategy = select_from_menu("Estratégia de Prompt", strategies)
    config["strategies"] = [strategy]
    
    # 5. Configurar diretórios de saída
    evaluator_name = evaluator.get("name", "unknown")
    model_name = re.split(r'[:/_-]', evaluator_name)[0]
    strategy_name = strategy["name"]
    objective_dir = "mop" if is_multiobjective else "evo"
    
    base_output_dir = os.path.join("logs", task_name, objective_dir, model_name, strategy_name)
    os.makedirs(base_output_dir, exist_ok=True)
    config["base_output_dir"] = base_output_dir
    
    return config


def handle_resumption(base_output_dir: str, is_multiobjective: bool) -> tuple:
    """Gerencia a lógica de retomar uma execução anterior."""
    
    if not confirm_action("\nDeseja retomar uma execução anterior?"):
        return 0, None
    
    while True:
        try:
            gen = int(input("De qual geração retomar? "))
            if gen < 0:
                print("⚠ Número de geração deve ser >= 0")
                continue
                
            population, next_gen = load_population_for_resumption(
                gen, base_output_dir, is_multiobjective
            )
            
            if population is not None:
                print(f"✓ População carregada. Continuando da geração {next_gen}")
                return next_gen, population
            
            print(f"⚠ Não foi possível carregar geração {gen}")
            if not confirm_action("Tentar outra geração?"):
                return 0, None
                
        except ValueError:
            print("⚠ Digite um número válido")


def main():
    """Função principal."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print_header("MO-PROMPT-PROJECT", char="█", width=60)
    print("  Otimização Evolutiva de Prompts para NLP")
    print("  " + "─" * 40)
    
    # Carrega configurações
    credentials = load_credentials_from_yaml("config/credentials.yaml")
    if not credentials:
        logger.critical("Falha ao carregar credentials.yaml")
        sys.exit(1)
    
    config = load_settings("config/experiment_settings.yaml", credentials)
    if not config:
        logger.critical("Falha ao carregar experiment_settings.yaml")
        sys.exit(1)
    
    # Configuração interativa
    try:
        config = setup_experiment_config(config)
    except ValueError as e:
        logger.critical(str(e))
        sys.exit(1)
    
    # Carrega dataset
    dataset = load_dataset(config)
    if dataset is None:
        logger.critical("Falha ao carregar dataset")
        sys.exit(1)
    
    # Carrega prompts iniciais
    prompts_path = f"data/initial_prompts_{config['task']}.txt"
    initial_prompts = load_initial_prompts(prompts_path)
    if not initial_prompts:
        logger.critical(f"Falha ao carregar prompts de {prompts_path}")
        sys.exit(1)
    
    # Exibe resumo
    print_config_summary(config)
    
    # Verifica retomada
    is_multi = config["objective"] == "multiobjetivo"
    start_gen, loaded_pop = handle_resumption(config["base_output_dir"], is_multi)
    
    # Configura caminhos de saída
    output_csv = os.path.join(config["base_output_dir"], "final_results.csv")
    output_plot = os.path.join(config["base_output_dir"], "final_pareto_front.png") if is_multi else ""
    
    # Executa evolução
    print_header("Iniciando Evolução", char="─")
    
    if is_multi:
        run_multi_evolution(
            config, dataset, initial_prompts, output_csv, output_plot,
            start_generation=start_gen, initial_population=loaded_pop
        )
    else:
        run_mono_evolution(
            config, dataset, initial_prompts, output_csv,
            start_generation=start_gen, initial_population=loaded_pop
        )
    
    print_header("Execução Finalizada", char="█")
    print(f"  Resultados salvos em: {output_csv}")
    if output_plot:
        print(f"  Gráfico Pareto: {output_plot}")
    print()


if __name__ == "__main__":
    main()
