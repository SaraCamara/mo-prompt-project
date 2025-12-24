import os
import re
import yaml
import pandas as pd
import logging

# Seção: Configuração e Carregamento de Dados
logger = logging.getLogger(__name__)

def load_credentials_from_yaml(path_to_yaml="config/credentials.yaml"):
    if not os.path.exists(path_to_yaml):
        logger.error(f"Arquivo de credenciais '{path_to_yaml}' não encontrado.")
        return None
    try:
        with open(path_to_yaml, "r") as f:
            creds = yaml.safe_load(f)
        if not creds:
            logger.error(f"Arquivo de credenciais '{path_to_yaml}' está vazio ou malformado.")
            return None
        
        logger.info(f"Credenciais carregadas de '{path_to_yaml}'.")
        return creds
    except Exception as e:
        logger.error(f"Erro ao carregar credenciais de '{path_to_yaml}': {e}")
        return None


def load_settings(settings_path="config/experiment_settings.yaml", credentials=None):
    if not os.path.exists(settings_path):
        logger.error(f"Arquivo de configurações '{settings_path}' não encontrado.")
        return None
    if credentials is None:
        logger.error(f"Credenciais não fornecidas para resolver placeholders.")
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
                logger.warning(f"Placeholder '{placeholder_key}' não encontrado em credentials.yaml.")
                return f"ERRO_PLACEHOLDER_{placeholder_key}"
        settings_str_resolved = re.sub(r"\$\{(\w+)\}", resolve_placeholder, settings_str)
        settings = yaml.safe_load(settings_str_resolved)
        if not settings:
            logger.error(f"Configurações em '{settings_path}' estão vazias.")
            return None
        logger.info(f"Configurações carregadas e processadas de '{settings_path}'.")
        return settings
    except Exception as e:
        logger.error(f"Erro ao carregar ou processar '{settings_path}': {e}")
        return None


def load_initial_prompts(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        if not prompts:
            logger.warning(f"Arquivo de prompts '{filepath}' está vazio.")
            return []
        logger.info(f"{len(prompts)} prompts carregados de '{filepath}'.")
        return prompts
    except FileNotFoundError:
        logger.error(f"Arquivo de prompts '{filepath}' não encontrado.")
        return []


def load_dataset(config):
    task = config.get("task")
    filepath = config.get("dataset_path")

    if not filepath or not os.path.exists(filepath):
        logger.error(f"Arquivo de dataset '{filepath}' não encontrado.")
        return None

    try:
        if task == 'imdb':
            df = pd.read_csv(filepath) # type: ignore
            logger.info(f"Dataset IMDB carregado com {len(df)} registros.")
            return df
        elif task == 'squad':
            df = pd.read_csv(filepath) # type: ignore
            logger.info(f"Dataset SQuAD carregado com {len(df)} registros.")
            return df
        else:
            logger.error(f"Tarefa '{task}' desconhecida para carregamento de dataset.")
            return None

    except Exception as e:
        logger.error(f"Erro ao carregar ou processar o dataset '{filepath}': {e}")
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
        logger.error(f"Arquivo de população para retomada '{file_path}' não encontrado.")
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
        logger.info(f"População da Geração {generation_to_load} carregada com sucesso de '{file_path}'.")
        return loaded_population, generation_to_load + 1
    except Exception as e:
        logger.error(f"Erro ao carregar população da Geração {generation_to_load} de '{file_path}': {e}")
        return None, None