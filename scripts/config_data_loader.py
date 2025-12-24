import os
import re
import yaml
import pandas as pd

# Seção: Configuração e Carregamento de Dados

def load_credentials_from_yaml(path_to_yaml="config/credentials.yaml"):
    if not os.path.exists(path_to_yaml):
        print(f"[config_data_loader] ERRO: Arquivo de credenciais '{path_to_yaml}' não encontrado.")
        return None
    try:
        with open(path_to_yaml, "r") as f:
            creds = yaml.safe_load(f)
        if not creds:
            print(f"[config_data_loader] ERRO: Arquivo de credenciais '{path_to_yaml}' está vazio ou malformado.")
            return None
        
        print(f"[config_data_loader] Credenciais carregadas de '{path_to_yaml}'.")
        return creds
    except Exception as e:
        print(f"[config_data_loader] ERRO ao carregar credenciais de '{path_to_yaml}': {e}")
        return None


def load_settings(settings_path="config/experiment_settings.yaml", credentials=None):
    if not os.path.exists(settings_path):
        print(f"[config_data_loader] ERRO: Arquivo de configurações '{settings_path}' não encontrado.")
        return None
    if credentials is None:
        print(f"[config_data_loader] ERRO: Credenciais não fornecidas para resolver placeholders.")
        return None
    try:
        with open(settings_path, "r") as f:
            settings_str = f.read()
        def resolve_placeholder(match):
            placeholder_key = match.group(1)
            value = credentials.get(placeholder_key)
            if value is not None:
                return str(value)
            else:
                print(f"[config_data_loader] [!] ATENÇÃO: Placeholder '{placeholder_key}' não encontrado em credentials.yaml.")
                return f"ERRO_PLACEHOLDER_{placeholder_key}"
        settings_str_resolved = re.sub(r"\$\{(\w+)\}", resolve_placeholder, settings_str)
        settings = yaml.safe_load(settings_str_resolved)
        if not settings:
            print(f"[config_data_loader] ERRO: Configurações em '{settings_path}' estão vazias.")
            return None
        print(f"[config_data_loader] Configurações carregadas e processadas de '{settings_path}'.")
        return settings
    except Exception as e:
        print(f"[config_data_loader] ERRO ao carregar ou processar '{settings_path}': {e}")
        return None


def load_initial_prompts(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        if not prompts:
            print(f"[config_data_loader] Arquivo de prompts '{filepath}' está vazio.")
            return []
        print(f"[config_data_loader] {len(prompts)} prompts carregados de '{filepath}'.")
        return prompts
    except FileNotFoundError:
        print(f"[config_data_loader] Arquivo de prompts '{filepath}' não encontrado.")
        return []


def load_dataset(config):
    task = config.get("task")
    filepath = config.get("dataset_path")

    if not filepath or not os.path.exists(filepath):
        print(f"[config_data_loader] ERRO: Arquivo de dataset '{filepath}' não encontrado.")
        return None

    try:
        if task == 'imdb':
            df = pd.read_csv(filepath)
            print(f"[config_data_loader] Dataset IMDB carregado com {len(df)} registros.")
            return df
        elif task == 'squad':
            df = pd.read_csv(filepath)
            print(f"[config_data_loader] Dataset SQuAD carregado com {len(df)} registros.")
            return df
        else:
            print(f"[config_data_loader] ERRO: Tarefa '{task}' desconhecida para carregamento de dataset.")
            return None

    except Exception as e:
        print(f"[config_data_loader] Erro ao carregar ou processar o dataset '{filepath}': {e}")
        return None

# Seção: Retomada de Execução

def load_population_for_resumption(generation_to_load, base_output_dir, is_multiobjective):
    """
    Carrega a população de uma geração específica para retomar a execução.

    Args:
        generation_to_load (int): O número da geração a ser carregada.
        base_output_dir (str): O diretório base onde os logs das gerações são salvos.

        is_multiobjective (bool): True se for uma execução multi-objetivo, False para mono-objetivo.

    Returns:
        tuple: (list of dict, int) A população carregada e o número da próxima geração,
               ou (None, None) se a população não puder ser carregada.
    """
    if is_multiobjective:
        # Multi-objective saves Pareto fronts in 'per_generation_pareto'
        file_path = os.path.join(base_output_dir, "per_generation_pareto", f"pareto_gen_{generation_to_load}.csv")
    else:
        # Mono-objective saves sorted population in 'generations_detail'
        file_path = os.path.join(base_output_dir, "generations_detail", f"population_sorted_gen_{generation_to_load}.csv")

    if not os.path.exists(file_path):
        print(f"[config_data_loader] ERRO: Arquivo de população para retomada '{file_path}' não encontrado.")
        return None, None

    try:
        df = pd.read_csv(file_path)
        loaded_population = []
        for _, row in df.iterrows():
            individual = {"prompt": row["prompt"]}
            
            individual["acc"] = row.get("acc", 0.0)
            individual["tokens"] = row.get("tokens", 0)

            if is_multiobjective:
                individual["f1"] = row.get("f1", 0.0) # Multi-objective uses 'f1'
                individual["rank"] = row.get("rank", -1)
                individual["crowding_distance"] = row.get("crowding_distance", 0.0)
            else:
                individual["f1"] = row.get("f1_score", 0.0) # Mono-objective uses 'f1_score'
            
            individual["metrics"] = (individual["acc"], individual["f1"], individual["tokens"], "")
            loaded_population.append(individual)
        print(f"[config_data_loader] População da Geração {generation_to_load} carregada com sucesso de '{file_path}'.")
        return loaded_population, generation_to_load + 1
    except Exception as e:
        print(f"[config_data_loader] Erro ao carregar população da Geração {generation_to_load} de '{file_path}': {e}")
        return None, None