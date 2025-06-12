# main.py
import subprocess
import sys
import os
import re
import yaml
import pandas as pd

from mono_evolution import run_mono_evolution
from multi_evolution import run_multi_evolution
from utils import load_credentials_from_yaml, load_settings

# Instalação de dependências
def install_requirements():
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("[main] Dependências instaladas com sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"[main] Erro ao instalar dependências: {e}")
            sys.exit(1)
    else:
        print(f"[main] Arquivo {requirements_file} não encontrado.")
        sys.exit(1)


# Função auxiliar para obter input numérico validado
def get_validated_numerical_input(prompt_message, num_options):
    while True:
        try:
            user_input = input(prompt_message)
            choice = int(user_input)
            if 0 <= choice < num_options:
                return choice
            else:
                print(f"[main] Opção inválida. Por favor, insira um número entre 0 e {num_options - 1}.")
        except ValueError:
            print("[main] Entrada inválida. Por favor, insira um número.")

if __name__ == "__main__":
    install_requirements()
    credentials = load_credentials_from_yaml("config/credentials.yaml")
    if not credentials:
        print("[main] [✗] ERRO FATAL: Falha ao carregar 'credentials.yaml'. Encerrando.")
        sys.exit(1)
    print("[main] [✓] Credenciais carregadas.")

    config = load_settings("config/experiment_settings.yaml", credentials)
    if not config:
        print("[main] [✗] ERRO FATAL: Falha ao carregar 'experiment_settings.yaml'. Encerrando.")
        sys.exit(1)
    print("[main] [✓] Configurações principais carregadas e processadas.")

    # Seleção de Modo (Mono/Multi)
    print("\n\n[main] [>] Selecione estratégia de otimização:\n  0) Mono-objetivo\n  1) Multiobjetivo")
    while True:
        optimization_type_input = input("Digite o número da opção desejada (0 ou 1): ")
        if optimization_type_input in ["0", "1"]:
            is_multiobjective = (optimization_type_input == "1")
            break
        else:
            print("[main] Opção inválida. Digite 0 ou 1.")
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

# Carregamento de Dados
dataset_path = config.get("dataset_path", "data/imdb_pt_subset.csv")
if not os.path.exists(dataset_path):
    print(f"[main] Arquivo de dataset '{dataset_path}' não encontrado. Verifique 'experiment_settings.yaml'. Encerrando.")
    sys.exit(1)
try:
    df_sample = pd.read_csv(dataset_path)
    print(f"[main] Dataset carregado: {dataset_path} ({len(df_sample)} registros)")
except Exception as e:
    print(f"[main] Erro ao carregar o dataset '{dataset_path}': {e}. Encerrando.")
    sys.exit(1)

# Carregamento da População Inicial
print("\n\n[main] [>] Carregamento da população inicial")
prompts_path = "data/initial_prompts.txt"
if os.path.exists(prompts_path):
    with open(prompts_path, "r", encoding="utf-8") as f:
        initial_prompts = [line.strip() for line in f if line.strip()]
    
    if not initial_prompts:
        print(f"[main] Arquivo de prompts '{prompts_path}' está vazio. Encerrando.")
        sys.exit(1)
    
    print(f"[main] {len(initial_prompts)} prompts carregados do arquivo: {prompts_path}")
else:
    print(f"[main] Arquivo de prompts '{prompts_path}' não encontrado. Encerrando.")
    sys.exit(1)

# Configuração de Caminhos de Saída 
print("\n\n[main] [>] Configurando diretório de saída para o experimento...")
objective_path_name = "emo" if is_multiobjective else "evo"

base_output_dir = os.path.join("logs", objective_path_name, output_model_name, strategy_name)
print(f"[main] Todos os resultados e logs para esta execução serão salvos em: '{base_output_dir}'")
os.makedirs(base_output_dir, exist_ok=True)

config["base_output_dir"] = base_output_dir

output_csv = os.path.join(base_output_dir, "final_results.csv")
output_plot = os.path.join(base_output_dir, "final_pareto_front.png") if is_multiobjective else ""

print(f"[main] Caminhos configurados:\n - CSV: {output_csv}")
if output_plot:
    print(f" - Plot: {output_plot}")

# Execução do Algoritmo
print("\n\n[main] [>] Iniciando execução do algoritmo evolutivo...\n")

if is_multiobjective:
    run_multi_evolution(config, df_sample, initial_prompts, output_csv, output_plot)
else:
    run_mono_evolution(config, df_sample, initial_prompts, output_csv)

print(f"\n[main] Execução finalizada. Resultados disponíveis em:\n - {output_csv}\n")
