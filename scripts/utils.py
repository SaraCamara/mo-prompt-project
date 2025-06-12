# utils.py
import os
import re
import yaml
import random
import pandas as pd
import requests
import openai
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.tokenize import TreebankWordTokenizer


tokenizer = TreebankWordTokenizer()


# Seção: Configuração e Carregamento de Dados

def load_credentials_from_yaml(path_to_yaml="config/credentials.yaml"):
    if not os.path.exists(path_to_yaml):
        print(f"[utils] ERRO: Arquivo de credenciais '{path_to_yaml}' não encontrado.")
        return None
    try:
        with open(path_to_yaml, "r") as f:
            creds = yaml.safe_load(f)
        if not creds:
            print(f"[utils] ERRO: Arquivo de credenciais '{path_to_yaml}' está vazio ou malformado.")
            return None
        
        print(f"[utils] Credenciais carregadas de '{path_to_yaml}'.")
        return creds
    except Exception as e:
        print(f"[utils] ERRO ao carregar credenciais de '{path_to_yaml}': {e}")
        return None


def load_settings(settings_path="config/experiment_settings.yaml", credentials=None):
    if not os.path.exists(settings_path):
        print(f"[utils] ERRO: Arquivo de configurações '{settings_path}' não encontrado.")
        return None
    if credentials is None:
        print(f"[utils] ERRO: Credenciais não fornecidas para resolver placeholders.")
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
                print(f"[utils] [!] ATENÇÃO: Placeholder '{placeholder_key}' não encontrado em credentials.yaml.")
                return f"ERRO_PLACEHOLDER_{placeholder_key}"
        settings_str_resolved = re.sub(r"\$\{(\w+)\}", resolve_placeholder, settings_str)
        settings = yaml.safe_load(settings_str_resolved)
        if not settings:
            print(f"[utils] ERRO: Configurações em '{settings_path}' estão vazias.")
            return None
        print(f"[utils] Configurações carregadas e processadas de '{settings_path}'.")
        return settings
    except Exception as e:
        print(f"[utils] ERRO ao carregar ou processar '{settings_path}': {e}")
        return None


def load_initial_prompts(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        if not prompts:
            print(f"[utils] Arquivo de prompts '{filepath}' está vazio.")
            return []
        print(f"[utils] {len(prompts)} prompts carregados de '{filepath}'.")
        return prompts
    except FileNotFoundError:
        print(f"[utils] Arquivo de prompts '{filepath}' não encontrado.")
        return []


def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"[utils] Dataset carregado com {len(df)} registros.")
        return df
    except FileNotFoundError:
        print(f"[utils] Dataset '{filepath}' não encontrado.")
        return None
    except Exception as e:
        print(f"[utils] Erro ao carregar dataset '{filepath}': {e}")
        return None


# Seção: Requisições a Modelos

def query_maritalk(full_prompt, model_config):
    model_name = model_config.get("name", "sabiazinho-3")
    api_key = model_config.get("chave_api")
    endpoint_url = model_config.get("endpoint")
    if not api_key or not endpoint_url:
        print(f"[utils] [Maritalk] API Key ou Endpoint não configurado para {model_name}.")
        return "erro_configuracao"
    request_data = {"model": model_name, "messages": [{"role": "user", "content": full_prompt}]}
    try:
        response = requests.post(
            url=endpoint_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=request_data,
            timeout=55
        )
        response.raise_for_status()
        return response.json().get("answer", "").strip().lower()
    except Exception as e:
        print(f"[utils] Erro ao consultar o Maritalk ({model_name}): {e}")
        return "erro_api"


def query_ollama(prompt, model_config):
    model_name = model_config.get("name")
    server_url = model_config.get("endpoint")
    if not model_name or not server_url:
        print(f"[utils] [Ollama] Nome do modelo ou URL do servidor não configurado.")
        return "erro_configuracao"
    if server_url.endswith("/api/generate"):
        chat_server_url = server_url.replace("/api/generate", "/api/chat")
    elif not server_url.endswith("/api/chat"):
        chat_server_url = server_url.rstrip('/') + "/api/chat"
    else:
        chat_server_url = server_url
    try:
        response = requests.post(
            url=chat_server_url,
            json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=55
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip().lower()
    except requests.exceptions.RequestException as e:
        print(f"[utils] Erro ao consultar modelo Ollama '{model_name}': {e}")
        return "erro_api"


# Seção: Operadores Evolutivos

def _call_openai_api(messages, generator_config, temperature=0.8):
    local_api_key = generator_config.get("chave_api")
    local_api_base = generator_config.get("endpoint")
    model_name = generator_config.get("name")
    if not local_api_key:
        print(f"[utils] [OpenAI] ERRO CRÍTICO: Chave API não fornecida.")
        return "erro_configuracao_gerador_sem_chave_api"
    if not model_name:
        print(f"[utils] [OpenAI] Nome do modelo não fornecido.")
        return "erro_configuracao_gerador_sem_modelo"
    original_api_key = openai.api_key
    original_api_base = openai.api_base
    try:
        openai.api_key = local_api_key
        openai.api_base = local_api_base if local_api_base else "https://api.openai.com/v1"
        response = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=temperature)
        return response.choices[0].message["content"].strip()
    except openai.error.AuthenticationError as e:
        print(f"[utils] ERRO DE AUTENTICAÇÃO com a API OpenAI: {e}.")
        return "erro_api_gerador_autenticacao"
    except Exception as e:
        print(f"[utils] Erro inesperado durante a chamada da API OpenAI para '{model_name}': {type(e).__name__} - {e}")
        return "erro_api_gerador_inesperado"
    finally:
        openai.api_key = original_api_key
        openai.api_base = original_api_base


def crossover_and_mutation_ga(pair_of_parent_prompts, config):
    generator_config = config.get("generator")
    if not generator_config:
        print("[utils] Configuração do gerador não encontrada.")
        return [{"prompt": "erro_configuracao_gerador"}]
    if len(pair_of_parent_prompts) != 2:
        print("[utils] crossover_and_mutation_ga espera exatamente dois pais.")
        return [{"prompt": "erro_numero_pais_invalido"}]
    template_generator = generator_config.get("template_generator", {})
    system_instruction = template_generator.get("system")
    user_instruction_crossover = template_generator.get("user_crossover")
    user_instruction_mutation = template_generator.get("user_mutation")
    prompt_a = pair_of_parent_prompts[0]["prompt"]
    prompt_b = pair_of_parent_prompts[1]["prompt"]
    crossover_messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_instruction_crossover.format(prompt_a=prompt_a, prompt_b=prompt_b)}]
    crossover_prompt = _call_openai_api(crossover_messages, generator_config)
    if "erro_" in crossover_prompt:
        return [{"prompt": f"prompt_gerado_com_erro_crossover ({crossover_prompt})"}]
    mutation_messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_instruction_mutation.format(prompt=crossover_prompt)}]
    mutated_prompt = _call_openai_api(mutation_messages, generator_config)
    if "erro_" in mutated_prompt:
        return [{"prompt": f"prompt_gerado_com_erro_mutacao ({mutated_prompt})"}]
    else:
        return [{"prompt": mutated_prompt}]


# Seção: Avaliação de Prompts

def evaluate_prompt_single(prompt_instruction: str, text: str, label: int,
                        evaluator_config: dict, strategy_config: dict,
                        experiment_settings: dict) -> tuple[int, str]:
    strategy_name = strategy_config.get("name", "desconhecida").lower()
    template_str = strategy_config.get("template")
    if not isinstance(prompt_instruction, str): prompt_instruction = str(prompt_instruction)
    if not isinstance(text, str): text = str(text)
    if not template_str:
        print(f"[utils] Template não encontrado para a estratégia '{strategy_name}'.")
        return 1 - label, "erro_template_ausente"
    instruction_suffix = "\nResponda apenas com '1' para resenhas positivas ou '0' para resenhas negativas."
    format_args = {"text": text, "prompt_instruction": prompt_instruction}
    if strategy_name == "cot": pass
    elif strategy_name == "few-shot":
        format_args["prompt_instruction"] += instruction_suffix
        format_args["examples"] = strategy_config.get("examples", "")
    else: # zero-shot e outros
        format_args["prompt_instruction"] += instruction_suffix
    try:
        full_prompt = template_str.format(**format_args)
    except KeyError as e:
        print(f"[utils] ERRO DE FORMATAÇÃO para '{strategy_name}': placeholder {e} ausente.")
        return 1 - label, f"erro_formatacao_template"
    evaluator_type = evaluator_config.get("tipo", "").lower()
    if evaluator_type == "maritalk": response_text = query_maritalk(full_prompt, evaluator_config)
    elif evaluator_type == "ollama": response_text = query_ollama(full_prompt, evaluator_config)
    else:
        print(f"[utils] Tipo de avaliador desconhecido: '{evaluator_type}'")
        response_text = "erro_tipo_avaliador"
    prediction = extract_label(response_text)
    if prediction is None:
        prediction = 1 - label
    return prediction, response_text


def evaluate_prompt(prompt_instruction, dataset, evaluator_config, strategy_config, experiment_settings, output_dir):
    predictions, true_labels = [], dataset["label"].tolist()
    texts = dataset["text"].tolist()
    evaluator_name_sanitized = evaluator_config["name"].replace(":", "_").replace("/", "_")
    strategy_name_sanitized = strategy_config["name"]
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"eval_{evaluator_name_sanitized}_{strategy_name_sanitized}.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f: f.write("prompt_instruction,text,true_label,llm_response,predicted_label\n")
    with open(log_path, "a", encoding="utf-8") as f:
        for text, label in zip(texts, true_labels):
            predicted, response_text = evaluate_prompt_single(prompt_instruction, text, label, evaluator_config, strategy_config, experiment_settings)
            predictions.append(predicted)
            f.write(f"\"{prompt_instruction}\",\"{text.replace('\"', '\"\"')}\",{label},\"{str(response_text).replace('\"', '\"\"')}\",{predicted}\n")
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    tokens = count_tokens(prompt_instruction)
    alert_message = ""
    if precision == 0 and recall == 0 and any(p == 1 for p in predictions):
        alert_message = "Previsões positivas feitas, mas todas incorretas."
    elif not any(p == 1 for p in predictions) and any(l == 1 for l in true_labels):
        alert_message = "Nenhuma previsão positiva feita, mas existiam exemplos positivos."
    return acc, f1, tokens, alert_message


def extract_label(text: str) -> int | None:
    if not isinstance(text, str): return None
    match = re.search(r'\b(0|1)\b', text)
    if match: return int(match.group(1))
    return None


def count_tokens(prompt: str) -> int:
    return len(tokenizer.tokenize(prompt))


# Seção: Algoritmos de Seleção

def roulette_wheel_selection(population, num_parents_to_select):
    if not population:
        print("[utils] [!] População vazia para seleção por roleta.")
        return []
    valid_individuals = [ind for ind in population if "metrics" in ind and ind["metrics"][0] >= 0]
    if not valid_individuals:
        print("[utils] [!] Nenhum indivíduo com fitness válida para seleção.")
        return random.sample(population, min(num_parents_to_select, len(population)))
    fitness_values = [ind["metrics"][0] for ind in valid_individuals]
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return random.sample(valid_individuals, min(num_parents_to_select, len(valid_individuals)))
    probabilities = [f / total_fitness for f in fitness_values]
    num_to_sample = min(num_parents_to_select, len(valid_individuals))
    selected_indices = np.random.choice(len(valid_individuals), size=num_to_sample, p=probabilities, replace=False)
    return [valid_individuals[i] for i in selected_indices]


def tournament_selection_multiobjective(population_with_rank_and_crowding, k_tournament_size, num_to_select):
    selected_parents = []
    population_size = len(population_with_rank_and_crowding)
    if population_size == 0: return []
    if k_tournament_size > population_size: k_tournament_size = population_size
    for _ in range(num_to_select):
        tournament_candidates_indices = random.sample(range(population_size), k_tournament_size)
        tournament_candidates = [population_with_rank_and_crowding[i] for i in tournament_candidates_indices]
        best_candidate = tournament_candidates[0]
        for i in range(1, k_tournament_size):
            candidate = tournament_candidates[i]
            if candidate['rank'] < best_candidate['rank']:
                best_candidate = candidate
            elif candidate['rank'] == best_candidate['rank']:
                if candidate['crowding_distance'] > best_candidate['crowding_distance']:
                    best_candidate = candidate
        selected_parents.append(best_candidate)
    return selected_parents


# Seção: Lógica NSGA-II

def dominates(ind_a_objectives, ind_b_objectives):
    a_is_better_or_equal = (ind_a_objectives["acc"] >= ind_b_objectives["acc"] and ind_a_objectives["tokens"] <= ind_b_objectives["tokens"])
    a_is_strictly_better = (ind_a_objectives["acc"] > ind_b_objectives["acc"] or ind_a_objectives["tokens"] < ind_b_objectives["tokens"])
    return a_is_better_or_equal and a_is_strictly_better


def fast_non_dominated_sort(population_with_objectives):
    fronts = [[]]
    for p_ind in population_with_objectives:
        p_ind['dominated_solutions_indices'] = []
        p_ind['domination_count'] = 0
        for q_idx, q_ind in enumerate(population_with_objectives):
            if p_ind == q_ind: continue
            if dominates(p_ind, q_ind):
                p_ind['dominated_solutions_indices'].append(q_idx)
            elif dominates(q_ind, p_ind):
                p_ind['domination_count'] += 1
        if p_ind['domination_count'] == 0:
            p_ind['rank'] = 0
            fronts[0].append(p_ind)
    current_rank_idx = 0
    while fronts[current_rank_idx]:
        next_front_individuals = []
        for p_ind in fronts[current_rank_idx]:
            for q_idx in p_ind['dominated_solutions_indices']:
                q_ind_ref = population_with_objectives[q_idx]
                q_ind_ref['domination_count'] -= 1
                if q_ind_ref['domination_count'] == 0:
                    q_ind_ref['rank'] = current_rank_idx + 1
                    next_front_individuals.append(q_ind_ref)
        current_rank_idx += 1
        if next_front_individuals:
            fronts.append(next_front_individuals)
        else:
            break
    return fronts


def compute_crowding_distance(front_individuals):
    if not front_individuals: return []
    num_individuals = len(front_individuals)
    for ind in front_individuals: ind['crowding_distance'] = 0.0
    objectives = {'acc': True, 'tokens': False}
    for obj_key, maximize in objectives.items():
        sorted_front = sorted(front_individuals, key=lambda x: x[obj_key])
        sorted_front[0]['crowding_distance'] = float('inf')
        if num_individuals > 1: sorted_front[num_individuals - 1]['crowding_distance'] = float('inf')
        min_obj_val = sorted_front[0][obj_key]
        max_obj_val = sorted_front[num_individuals - 1][obj_key]
        range_obj = max_obj_val - min_obj_val
        if range_obj == 0: continue
        for i in range(1, num_individuals - 1):
            if sorted_front[i]['crowding_distance'] != float('inf'):
                sorted_front[i]['crowding_distance'] += (sorted_front[i+1][obj_key] - sorted_front[i-1][obj_key]) / range_obj
    return front_individuals


# Seção: Funções Auxiliares de Evolução

def evaluate_population(prompts_to_evaluate, dataset, config):
    evaluated_population = []
    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    base_output_dir = config["base_output_dir"]
    eval_log_dir = os.path.join(base_output_dir, "prompt_eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)
    prompt_list = [p["prompt"] if isinstance(p, dict) else p for p in prompts_to_evaluate]
    for p_text in prompt_list:
        metrics = evaluate_prompt(p_text, dataset, evaluator_config, strategy_config, config, eval_log_dir)
        acc, f1, tokens, _ = metrics
        evaluated_population.append({"prompt": p_text, "acc": acc, "f1": f1, "tokens": tokens, "metrics": metrics})
    return evaluated_population


def generate_unique_offspring(current_population, config):
    offspring_prompts = []
    existing_prompts = {ind['prompt'] for ind in current_population}
    population_size = config.get("population_size", 10)
    k_tournament_parents = config.get("k_tournament_parents", 2)
    max_attempts = population_size * 3
    attempts = 0
    while len(offspring_prompts) < population_size and attempts < max_attempts:
        attempts += 1
        if len(current_population) < 2: break
        parent_pair = tournament_selection_multiobjective(current_population, k_tournament_parents, 2)
        child_dict_list = crossover_and_mutation_ga(parent_pair, config)
        if child_dict_list and "prompt" in child_dict_list[0] and "erro_" not in child_dict_list[0]["prompt"]:
            new_prompt = child_dict_list[0]["prompt"]
            if new_prompt not in existing_prompts:
                offspring_prompts.append({"prompt": new_prompt})
                existing_prompts.add(new_prompt)
    if len(offspring_prompts) < population_size:
        print(f"[utils] [!] Apenas {len(offspring_prompts)} filhos únicos foram gerados.")
    return [p["prompt"] for p in offspring_prompts]


def select_survivors_nsgaii(parent_population, offspring_population, population_size):
    combined_population = parent_population + offspring_population
    all_fronts = fast_non_dominated_sort(combined_population)
    next_generation = []
    for front in all_fronts:
        if not front: continue
        compute_crowding_distance(front)
        if len(next_generation) + len(front) <= population_size:
            next_generation.extend(front)
        else:
            num_needed = population_size - len(next_generation)
            front.sort(key=lambda x: x['crowding_distance'], reverse=True)
            next_generation.extend(front[:num_needed])
            break
    return next_generation


# Seção: Persistência e Salvamento de Resultados

def save_generation_results(population, generation, config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    evaluator_name = config.get("evaluators", [{}])[0].get("name", "unknown_model").replace(":", "_").replace("/", "_")
    strategy_name = config.get("strategies", [{}])[0].get("name", "unknown_strategy")
    path = os.path.join(output_dir, f"results_gen_{generation}_{evaluator_name}_{strategy_name}.csv")
    data = []
    for ind in population:
        prompt, metrics = ind.get("prompt"), ind.get("metrics")
        if metrics and len(metrics) >= 4:
            acc, f1, tokens, alert_message = metrics[:4]
        else:
            acc, f1, tokens, alert_message = 0.0, 0.0, 0, "metrics_missing"
        data.append({"generation": generation, "prompt": prompt, "accuracy": acc, "f1_score": f1, "tokens": tokens, "alert": alert_message})
    df = pd.DataFrame(data)
    df = df.sort_values(by=["accuracy", "tokens"], ascending=[False, True])
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"[utils] Resultados detalhados da geração {generation} salvos em {path}")


def save_sorted_population(population, generation, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sorted_log_path = os.path.join(output_dir, f"population_sorted_gen_{generation}.csv")
    data = []
    for ind in population:
        prompt, metrics = ind.get("prompt"), ind.get("metrics")
        if metrics and len(metrics) >= 4:
            acc, f1, tokens, alert_message = ind["metrics"][:4]
        else:
            acc, f1, tokens, alert_message = 0.0, 0.0, 0, "metrics_missing"
        data.append({"generation": generation, "prompt": prompt, "accuracy": acc, "f1_score": f1, "tokens": tokens, "alert": alert_message})
    df = pd.DataFrame(data)
    df = df.sort_values(by=["accuracy", "tokens"], ascending=[False, True])
    df.to_csv(sorted_log_path, index=False, encoding='utf-8')
    print(f"[utils] População ordenada da geração {generation} salva em {sorted_log_path}")


def save_final_results(population, config, output_csv_path): 
    print("[utils] Salvando resultados finais.")
    data = []
    for ind in population:
        prompt, metrics = ind.get("prompt"), ind.get("metrics")
        if metrics and len(metrics) >= 4:
            acc, f1, tokens, alert_message = ind.get("metrics")[:4]
        else:
            acc, f1, tokens, alert_message = 0.0, 0.0, 0, "metrics_final_missing"
        data.append({"prompt": prompt, "accuracy": acc, "f1_score": f1, "tokens": tokens, "alert": alert_message})
    df = pd.DataFrame(data)
    top_k = config.get("top_k", len(df))
    df_top_k = df.head(top_k)
    df_top_k.to_csv(output_csv_path, index=False, encoding='utf-8')


def save_pareto_front_data(front_individuals, csv_path, plot_path):
    if not front_individuals:
        df_empty = pd.DataFrame(columns=["prompt", "acc", "f1", "tokens", "rank", "crowding_distance"])
        df_empty.to_csv(csv_path, index=False)
        plt.figure()
        plt.text(0.5, 0.5, "Fronteira de Pareto Vazia", ha='center', va='center')
        plt.xlabel("Número de Tokens")
        plt.ylabel("Acurácia")
        plt.title("Fronteira de Pareto (Tokens vs Acurácia)")
        plt.savefig(plot_path)
        plt.close()
        return
    data_to_save = []
    for ind in front_individuals:
        data_to_save.append({"prompt": ind.get("prompt", "N/A"), "acc": ind.get("acc", 0.0), "f1": ind.get("f1", 0.0), "tokens": ind.get("tokens", 0), "rank": ind.get("rank", -1), "crowding_distance": ind.get("crowding_distance", 0.0)})
    df = pd.DataFrame(data_to_save)
    df_sorted = df.sort_values(by="acc", ascending=False)
    df_sorted.to_csv(csv_path, index=False, encoding='utf-8')
    plt.figure(figsize=(10, 6))
    plt.scatter(df["tokens"], df["acc"], c='blue', alpha=0.7, edgecolors='w', s=70)
    plt.xlabel("Número de Tokens (Menor é Melhor)")
    plt.ylabel("Acurácia (Maior é Melhor)")
    plt.title("Fronteira de Pareto (Tokens vs Acurácia)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(plot_path)
    plt.close()
    print(f"[utils] Gráfico da fronteira de Pareto salvo em {plot_path}")