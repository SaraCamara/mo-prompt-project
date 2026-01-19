# main.py
"""Ponto de entrada principal para o mo-prompt-project."""
import os
import sys
import logging

from .core import run_mono_evolution, run_multi_evolution
from .utils import (
    select_from_menu, confirm_action, print_header,
    load_credentials_from_yaml, load_settings, load_dataset,
    load_initial_prompts, load_population_for_resumption,
    setup_logging, detect_resumable_run
)


def print_config_summary(config):
    """Display configuration summary before evolution."""
    print("\n" + "─" * 60)
    print("CONFIGURAÇÃO DO EXPERIMENTO")
    print("─" * 60)
    print(f"Tarefa: {config['task']}")
    print(f"Modo: {config['objective']}")
    model_name = config['evaluators'][0].get('name', 'unknown')
    print(f"Modelo: {model_name}")
    strategy_name = config['strategies'][0]['name']
    print(f"Estratégia: {strategy_name}")
    print(f"Saída: {config['base_output_dir']}")
    print("─" * 60 + "\n")


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
    model_name = evaluator.get("name", "unknown")
    strategy_name = strategy["name"]
    objective_dir = "mop" if is_multiobjective else "evo"
    
    base_output_dir = os.path.join("logs", task_name, objective_dir, model_name, strategy_name)
    os.makedirs(base_output_dir, exist_ok=True)
    config["base_output_dir"] = base_output_dir
    
    return config


def handle_resumption(base_output_dir: str, is_multiobjective: bool) -> tuple:
    """Gerencia a lógica de retomar uma execução anterior."""
    
    # Tenta detectar automaticamente um estado retomável
    resumable_state = detect_resumable_run(base_output_dir)
    
    if resumable_state:
        last_gen = resumable_state["last_completed_generation"]
        next_gen = resumable_state["next_generation"]
        accumulated_time = resumable_state["accumulated_time"]
        stop_reason = resumable_state.get("stop_reason", "unknown")
        
        hours, remainder = divmod(accumulated_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        print("\n" + "="*60)
        print(" EXECUÇÃO ANTERIOR DETECTADA")
        print("="*60)
        print(f"  Última geração completada: {last_gen}")
        print(f"  Tempo acumulado: {time_str}")
        print(f"  Razão da parada: {stop_reason}")
        print(f"  Próxima geração: {next_gen}")
        print("="*60)
        
        if confirm_action("\nDeseja retomar desta execução?"):
            population, _ = load_population_for_resumption(
                last_gen, base_output_dir, is_multiobjective
            )
            
            if population is not None:
                print(f"✓ População carregada. Continuando da geração {next_gen}")
                print(f"✓ Estado restaurado: stagnation_counter={resumable_state.get('stagnation_counter', 0)}")
                return next_gen, population, resumable_state
            else:
                print("⚠ Falha ao carregar população")
    
    # Fallback: pergunta manualmente
    if not resumable_state or not confirm_action("\nDeseja retomar uma execução anterior (entrada manual)?"):
        return 0, None, None
    
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
                # Sem estado carregado em modo manual
                return next_gen, population, None
            
            print(f"⚠ Não foi possível carregar geração {gen}")
            if not confirm_action("Tentar outra geração?"):
                return 0, None, None
                
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
    start_gen, loaded_pop, loaded_state = handle_resumption(config["base_output_dir"], is_multi)
    
    # Configura caminhos de saída
    output_csv = os.path.join(config["base_output_dir"], "final_results.csv")
    output_plot = os.path.join(config["base_output_dir"], "final_pareto_front.png") if is_multi else ""
    
    # Executa evolução
    print_header("Iniciando Evolução", char="─")
    
    if is_multi:
        run_multi_evolution(
            config, dataset, initial_prompts, output_csv, output_plot,
            start_generation=start_gen, initial_population=loaded_pop,
            loaded_state=loaded_state
        )
    else:
        run_mono_evolution(
            config, dataset, initial_prompts, output_csv,
            start_generation=start_gen, initial_population=loaded_pop,
            loaded_state=loaded_state
        )
    
    print_header("Execução Finalizada", char="█")
    print(f"  Resultados salvos em: {output_csv}")
    if output_plot:
        print(f"  Gráfico Pareto: {output_plot}")
    print()


if __name__ == "__main__":
    main()
