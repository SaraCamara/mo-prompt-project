# utils.py
import collections
import json
import subprocess
import os
import re
import yaml
import random
import pandas as pd
import requests
import openai
import concurrent.futures
import matplotlib
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.tokenize import TreebankWordTokenizer


tokenizer = TreebankWordTokenizer()
matplotlib.use('Agg') # Configura o backend do Matplotlib para 'Agg' para evitar problemas de thread com GUI


# Instalação de dependências
def install_requirements():
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("[utils] Dependências instaladas com sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"[utils] Erro ao instalar dependências: {e}")
            sys.exit(1)
    else:
        print(f"[utils] Arquivo {requirements_file} não encontrado.")
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
                print(f"[utils] Opção inválida. Por favor, insira um número entre 0 e {num_options - 1}.")
        except ValueError:
            print("[utils] Entrada inválida. Por favor, insira um número.")

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


def load_dataset(config):
    task = config.get("task")
    filepath = config.get("dataset_path")

    if not filepath or not os.path.exists(filepath):
        print(f"[utils] ERRO: Arquivo de dataset '{filepath}' não encontrado.")
        return None

    try:
        if task == 'imdb':
            df = pd.read_csv(filepath)
            print(f"[utils] Dataset IMDB carregado com {len(df)} registros.")
            return df
        elif task == 'squad':
            df = pd.read_csv(filepath)
            print(f"[utils] Dataset SQuAD carregado com {len(df)} registros.")
            return df
        else:
            print(f"[utils] ERRO: Tarefa '{task}' desconhecida para carregamento de dataset.")
            return None

    except Exception as e:
        print(f"[utils] Erro ao carregar ou processar o dataset '{filepath}': {e}")
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
        # print(f"[utils] [OpenAI] Requisição para '{model_name}': {messages}")
        response = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=temperature)
        # print(f"[utils] [OpenAI] Resposta de '{model_name}': {response.choices[0].message['content'].strip()}")
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
    user_instruction_mutation = template_generator.get("user_mutation", "Mute: {prompt}") # Adicionado valor padrão
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


def mop_crossover_and_mutation_ga(pair_of_parent_prompts, config):
    """
    Realiza Crossover e, condicionalmente, Mutação.
    """
    generator_config = config.get("generator")
    
    # Carrega a taxa de mutação do config, default 1.0 (100%)
    evo_params = config.get("evolution_params", {})
    mutation_rate = evo_params.get("mutation_rate", 1.0) 

    if not generator_config:
        return [{"prompt": "erro_configuracao_gerador"}]

    template_generator = generator_config.get("template_generator", {})
    system_instruction = template_generator.get("system", "Você é um otimizador de prompts.")
    user_instruction_crossover = template_generator.get("user_crossover", "Combine: {prompt_a} e {prompt_b}")
    user_instruction_mutation = template_generator.get("rate_user_mutation", "Mute: {prompt}")

    prompt_a = pair_of_parent_prompts[0]["prompt"] if isinstance(pair_of_parent_prompts[0], dict) else pair_of_parent_prompts[0]
    prompt_b = pair_of_parent_prompts[1]["prompt"] if isinstance(pair_of_parent_prompts[1], dict) else pair_of_parent_prompts[1]

    print(f"[utils] [Crossover/Mutação] Pais: '{prompt_a[:100]}...' e '{prompt_b[:100]}...'")
    # CROSSOVER
    crossover_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_instruction_crossover.format(prompt_a=prompt_a, prompt_b=prompt_b)}
    ]
    
    # Temperatura média para o crossover (ex: 0.5 - 0.7)
    current_prompt = _call_openai_api(crossover_messages, generator_config, temperature=0.6)
    print(f"[utils] [Crossover/Mutação] Crossover gerado: '{current_prompt[:100]}...'")
    
    if "erro_" in current_prompt:
        return [{"prompt": f"erro_crossover ({current_prompt})"}]

    # MUTAÇÃO CONDICIONAL
    # Gera um número aleatório entre 0.0 e 1.0
    # Se for MENOR que a taxa (ex: 0.6), aplica a mutação.
    # Caso contrário, o filho é apenas o resultado do crossover.
    
    if random.random() < mutation_rate:
        # print(f"[utils] Aplicando mutação (Taxa: {mutation_rate})...")
        mutation_messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_instruction_mutation.format(prompt=current_prompt)}
        ]
        # Aumento de temperatura para forçar diversidade se ocorrer a mutação
        print(f"[utils] [Crossover/Mutação] Aplicando mutação (Taxa: {mutation_rate})...")
        mutated_prompt = _call_openai_api(mutation_messages, generator_config, temperature=0.9)
        
        if "erro_" not in mutated_prompt:
            current_prompt = mutated_prompt
        else:
            return [{"prompt": f"erro_mutacao ({mutated_prompt})"}]
    
    # else:
        # print(f"[utils] Mutação pulada pelo sorteio (mantendo resultado do crossover).")

    return [{"prompt": current_prompt}]


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

def evaluate_prompt_single_squad(prompt_instruction: str, context: str, question: str, executor_config: dict, strategy_config: dict) -> str:
    strategy_name = strategy_config.get("name", "desconhecida").lower()
    template_str = strategy_config.get("template")
    
    if not template_str:
        print(f"[utils] Template não encontrado para a estratégia SQuAD.")
        return ""

    format_args = {"prompt_instruction": prompt_instruction, "context": context, "question": question}

    # Adiciona exemplos se a estratégia for few-shot
    if strategy_name == "few-shot":
        format_args["examples"] = strategy_config.get("examples", "")

    
    try:
        # Tenta formatar. Se o template tiver chaves extras ou faltantes, vai dar erro.
        full_prompt = template_str.format(**format_args)
        
        # Lógica de chamada ao LLM, similar à da tarefa IMDB
        evaluator_type = executor_config.get("tipo", "").lower()
        if evaluator_type == "maritalk":
            return query_maritalk(full_prompt, executor_config)
        elif evaluator_type == "ollama":
            return query_ollama(full_prompt, executor_config)
        return "erro_tipo_avaliador_desconhecido"

    except KeyError as e:
        print(f"[utils] ERRO DE FORMATAÇÃO para SQuAD: placeholder '{e.args[0]}' ausente no dicionário de argumentos.")
        # Debug: Imprima o template para ver o que ele está esperando
        print(f"Template esperado: {template_str}") 
        return ""
    except Exception as e:
        print(f"[utils] Erro inesperado ao formatar prompt: {e}")
        return ""




def evaluate_prompt(prompt_instruction, dataset, executor_config, strategy_config, experiment_settings, output_dir):
    task = experiment_settings.get("task")
    if task == 'imdb':
        return evaluate_prompt_imdb(prompt_instruction, dataset, executor_config, strategy_config, experiment_settings, output_dir)
    elif task == 'squad':
        return evaluate_prompt_squad(prompt_instruction, dataset, executor_config, strategy_config, experiment_settings, output_dir)
    else:
        print(f"[utils] ERRO: Tarefa de avaliação '{task}' desconhecida.")
        return 0, 0, 0, "tarefa_desconhecida"

def evaluate_prompt_squad(prompt_instruction, dataset, executor_config, strategy_config, experiment_settings, output_dir):
    total_em = 0
    total_f1 = 0
    total_tokens = count_tokens(prompt_instruction)

    print(f"[utils] [SQuAD] Avaliando prompt_instruction: '{prompt_instruction}'")
    # Configure o número máximo de workers. Ajuste este valor com base nos limites de taxa da API
    MAX_WORKERS = 10
    
    executor_name_sanitized = executor_config["name"].replace(":", "_").replace("/", "_")
    # strategy_name_sanitized = strategy_config["name"] # Já definido acima
    strategy_name_sanitized = strategy_config["name"]
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"eval_{executor_name_sanitized}_{strategy_name_sanitized}.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("prompt_instruction,context,question,correct_answer,llm_response,exact_match,f1_score\n")
    
    # Prepare tasks for parallel execution, maintaining original order
    original_data_points = []
    for index, row in dataset.iterrows():
        original_data_points.append({
            'context': row['context'],
            'question': row['question'],
            'correct_answer': row['correct_answer']
        })

    ordered_results = [None] * len(original_data_points)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(evaluate_prompt_single_squad, prompt_instruction, dp['context'], dp['question'], executor_config, strategy_config): i
            for i, dp in enumerate(original_data_points)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            dp = original_data_points[index]
            try:
                predicted_answer = future.result()
                exact_match = compute_exact(dp['correct_answer'], predicted_answer)
                f1_score_val = compute_f1(dp['correct_answer'], predicted_answer)
                # Log detalhado para os primeiros exemplos
                if index < 3:
                    print(f"[utils] [SQuAD] Exemplo {index}: Q='{dp['question'][:100]}', GT='{dp['correct_answer'][:100]}', Pred='{predicted_answer[:100]}', EM={exact_match}, F1={f1_score_val}")

                ordered_results[index] = (predicted_answer, exact_match, f1_score_val, dp['context'], dp['question'], dp['correct_answer'])
            except Exception as exc:
                print(f"[utils] Erro durante a avaliação de um exemplo SQuAD (índice {index}): {exc}")
                ordered_results[index] = ("erro_processamento_paralelo", 0, 0, dp['context'], dp['question'], dp['correct_answer'])

    # Coleta as métricas e escreve os logs após todas as chamadas de API serem concluídas
    with open(log_path, "a", encoding="utf-8") as f:
        for predicted_answer, exact_match, f1_score_val, context, question, correct_answer in ordered_results:
            total_em += exact_match
            total_f1 += f1_score_val
            f.write(f'"{prompt_instruction}","{context.replace('"', "'''''")}","{question.replace('"', "'''''")}","{correct_answer.replace('"', "'''''")}","{predicted_answer.replace('"', "'''''")}",{exact_match},{f1_score_val}\n')

    avg_em = total_em / len(dataset)
    avg_f1 = total_f1 / len(dataset)
    
    return avg_em, avg_f1, total_tokens, ""

def evaluate_prompt_imdb(prompt_instruction, dataset, evaluator_config, strategy_config, experiment_settings, output_dir):
    predictions = []
    true_labels = dataset["label"].tolist()
    # texts = dataset["text"].tolist() # 'texts' não é usado diretamente após a refatoração
    
    evaluator_name_sanitized = evaluator_config["name"].replace(":", "_").replace("/", "_")
    strategy_name_sanitized = strategy_config["name"]
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"eval_{evaluator_name_sanitized}_{strategy_name_sanitized}.csv")

    total_tokens = count_tokens(prompt_instruction) # Conta tokens para a instrução do prompt uma vez

    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("prompt_instruction,text,true_label,llm_prediction,llm_response\n")

    print(f"[utils] [IMDB] Avaliando prompt_instruction: '{prompt_instruction}'")

    # Use ThreadPoolExecutor para chamadas de API paralelas
    MAX_WORKERS = 10 # Ajuste com base nos limites de taxa da API
    
    # Prepara tarefas para execução paralela, mantendo a ordem original
    original_data_points = []
    for index, row in dataset.iterrows():
        original_data_points.append({
            'text': row['text'],
            'label': row['label']
        })

    ordered_results = [None] * len(original_data_points)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(evaluate_prompt_single, prompt_instruction, dp['text'], dp['label'], evaluator_config, strategy_config, experiment_settings): i
            for i, dp in enumerate(original_data_points)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            dp = original_data_points[index]
            try:
                prediction, response_text = future.result()
                # Log detalhado para os primeiros exemplos
                if index < 3:
                    print(f"[utils] [IMDB] Exemplo {index}: Text='{dp['text'][:100]}', GT={dp['label']}, Pred={prediction}, LLM_Resp='{response_text[:100]}'")
                ordered_results[index] = (prediction, response_text, dp['text'], dp['label'])
            except Exception as exc:
                print(f"[utils] Erro durante a avaliação de um exemplo IMDB (índice {index}): {exc}")
                # Em caso de erro, assume previsão incorreta e registra resposta de erro
                ordered_results[index] = (1 - dp['label'], "erro_processamento_paralelo", dp['text'], dp['label'])

    # Coleta previsões e escreve logs após todas as chamadas de API serem concluídas
    with open(log_path, "a", encoding="utf-8") as f:
        for prediction, response_text, text, label in ordered_results:
            predictions.append(prediction)
            f.write(f'"{prompt_instruction}","{text.replace('"', "'''''")}","{label}","{prediction}","{response_text.replace('"', "'''''")}"\n')

    # Calcula métricas gerais
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='binary') # Assumindo classificação binária (0 ou 1)

    return acc, f1, total_tokens, ""


def extract_label(text: str) -> int | None:
    if not isinstance(text, str): return None
    # Adiciona log para a resposta bruta do LLM antes da extração
    if text and len(text) > 0:
        print(f"[utils] [Extract Label] Resposta bruta do LLM: '{text[:100]}...'")

    match = re.search(r'\b(0|1)\b', text)
    if match: return int(match.group(1))
    return None


def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import string, re

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(a_gold, a_pred):
    # Normaliza ambas as strings para garantir uma comparação consistente
    normalized_gold = normalize_text(a_gold)
    normalized_pred = normalize_text(a_pred)
    gold_toks = normalized_gold.split()
    pred_toks = normalized_pred.split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact(a_gold, a_pred):
    return int(normalize_text(a_gold) == normalize_text(a_pred))



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
    a_is_better_or_equal = (ind_a_objectives["f1"] >= ind_b_objectives["f1"] and ind_a_objectives["tokens"] <= ind_b_objectives["tokens"])
    a_is_strictly_better = (ind_a_objectives["f1"] > ind_b_objectives["f1"] or ind_a_objectives["tokens"] < ind_b_objectives["tokens"])
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
    objectives = {'f1': True, 'tokens': False}
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

def evaluate_population(prompts_to_evaluate, dataset, config, executor_config):
    evaluated_population = []
    evaluator_config = config["evaluators"][0]
    strategy_config = config["strategies"][0]
    base_output_dir = config["base_output_dir"]
    eval_log_dir = os.path.join(base_output_dir, "prompt_eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)
    prompt_list = [p["prompt"] if isinstance(p, dict) else p for p in prompts_to_evaluate]
    print(f"[utils] Iniciando avaliação de {len(prompt_list)} prompts.")
    for p_text in prompt_list:
        metrics = evaluate_prompt(p_text, dataset, executor_config, strategy_config, config, eval_log_dir)
        acc, f1, tokens, _ = metrics
        evaluated_population.append({"prompt": p_text, "acc": acc, "f1": f1, "tokens": tokens, "metrics": metrics})
    return evaluated_population


def generate_unique_offspring(current_population, config, evolution_type="mono"):
    """
    Gera uma nova população de descendentes únicos a partir da população atual,
    utilizando funções de seleção e crossover/mutação apropriadas para o tipo de evolução.

    Args:
        current_population (list): A população atual de indivíduos.
        config (dict): Dicionário de configurações do experimento.
        evolution_type (str): "mono" para mono-objetivo ou "multi" para multi-objetivo.

    Returns:
        list: Uma lista de strings, onde cada string é um prompt de um descendente único.
    """
    offspring_prompts_dicts = []
    existing_prompts = {ind['prompt'] for ind in current_population}
    population_size = config.get("population_size", 10)
    num_parents_for_crossover = 2 # Geralmente 2 pais para crossover
    max_attempts = population_size * 3
    attempts = 0
    
    print(f"[utils] Gerando até {population_size} descendentes únicos ({evolution_type}-objetivo).")

    while len(offspring_prompts_dicts) < population_size and attempts < max_attempts:
        attempts += 1
        if len(current_population) < num_parents_for_crossover:
            print(f"[utils] [Offspring {evolution_type.capitalize()}] População atual menor que {num_parents_for_crossover}, não é possível gerar offspring.")
            break
        
        parent_pair = []
        crossover_mutation_func = None

        if evolution_type == "mono":
            parent_pair = roulette_wheel_selection(current_population, num_parents_for_crossover)
            crossover_mutation_func = crossover_and_mutation_ga
        elif evolution_type == "multi":
            k_tournament_parents = config.get("k_tournament_parents", 2)
            parent_pair = tournament_selection_multiobjective(current_population, k_tournament_parents, num_parents_for_crossover)
            crossover_mutation_func = mop_crossover_and_mutation_ga
        else:
            print(f"[utils] [Offspring] Tipo de evolução desconhecido: {evolution_type}")
            break

        if len(parent_pair) < num_parents_for_crossover:
            continue 
        
        child_dict_list = crossover_mutation_func(parent_pair, config)
        
        if child_dict_list and "prompt" in child_dict_list[0] and "erro_" not in child_dict_list[0]["prompt"]:
            new_prompt = child_dict_list[0]["prompt"]
            if new_prompt not in existing_prompts:
                offspring_prompts_dicts.append({"prompt": new_prompt})
                existing_prompts.add(new_prompt)

    if len(offspring_prompts_dicts) < population_size:
        print(f"[utils] [!] Apenas {len(offspring_prompts_dicts)} filhos únicos foram gerados ({evolution_type}-objetivo).")
    
    return [p["prompt"] for p in offspring_prompts_dicts] # Retorna uma lista de strings


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
        data.append({"generation": generation, "prompt": prompt, "acc": acc, "f1_score": f1, "tokens": tokens, "alert": alert_message})
    df = pd.DataFrame(data)
    df = df.sort_values(by=["f1_score", "tokens"], ascending=[False, True])
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
        data.append({"generation": generation, "prompt": prompt, "acc": acc, "f1_score": f1, "tokens": tokens, "alert": alert_message})
    df = pd.DataFrame(data)
    df = df.sort_values(by=["f1_score", "tokens"], ascending=[False, True])
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
        data.append({"prompt": prompt, "acc": acc, "f1_score": f1, "tokens": tokens, "alert": alert_message})
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
        plt.ylabel("F1 Score")
        plt.title("Fronteira de Pareto (Tokens vs F1 Score)")
        plt.savefig(plot_path)
        plt.close()
        return
    data_to_save = []
    for ind in front_individuals:
        data_to_save.append({"prompt": ind.get("prompt", "N/A"), "acc": ind.get("acc", 0.0), "f1": ind.get("f1", 0.0), "tokens": ind.get("tokens", 0), "rank": ind.get("rank", -1), "crowding_distance": ind.get("crowding_distance", 0.0)})
    df = pd.DataFrame(data_to_save)
    df_sorted = df.sort_values(by="f1", ascending=False)
    df_sorted.to_csv(csv_path, index=False, encoding='utf-8')
    plt.figure(figsize=(10, 6))
    plt.scatter(df["tokens"], df["f1"], c='blue', alpha=0.7, edgecolors='w', s=70)
    plt.xlabel("Número de Tokens (Menor é Melhor)")
    plt.ylabel("F1 Score (Maior é Melhor)")
    plt.title("Fronteira de Pareto (Tokens vs F1 Score)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(plot_path)
    plt.close()
    print(f"[utils] Gráfico da fronteira de Pareto salvo em {plot_path}")

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
        print(f"[utils] ERRO: Arquivo de população para retomada '{file_path}' não encontrado.")
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
        print(f"[utils] População da Geração {generation_to_load} carregada com sucesso de '{file_path}'.")
        return loaded_population, generation_to_load + 1
    except Exception as e:
        print(f"[utils] Erro ao carregar população da Geração {generation_to_load} de '{file_path}': {e}")
        return None, None