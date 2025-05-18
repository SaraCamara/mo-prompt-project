import os
import re
import yaml
import pandas as pd
import requests
import openai
import random
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

# Configuração
def load_credentials_from_yaml(path_to_yaml="config/credentials.yaml"):
    print(f"[utils] Carregando credenciais de '{path_to_yaml}'...")
    if not os.path.exists(path_to_yaml):
        print(f"[utils] [✗] ERRO: Arquivo de credenciais '{path_to_yaml}' não encontrado. ")
        return None
    try:
        with open(path_to_yaml, "r") as f:
            creds = yaml.safe_load(f)
        if not creds:
            print(f"[utils] [✗] ERRO: Arquivo de credenciais '{path_to_yaml}' está vazio ou malformado.")
            return None
        
        print(f"[utils] [✓] Credenciais carregadas de '{path_to_yaml}'. (Lembre-se de adicionar ao .gitignore)")
        return creds
    except Exception as e:
        print(f"[utils] [✗] ERRO ao carregar credenciais de '{path_to_yaml}': {e}")
        return None

def load_settings(settings_path="config/experiment_settings.yaml", credentials=None):
    print(f"[utils] Carregando configurações de '{settings_path}'...")
    if not os.path.exists(settings_path):
        print(f"[utils] [✗] ERRO: Arquivo de configurações '{settings_path}' não encontrado.")
        return None

    if credentials is None:
        print(f"[utils] [✗] ERRO: Dicionário de 'credentials' não fornecido para resolver placeholders em load_settings.")
        return None

    try:
        with open(settings_path, "r") as f:
            settings_str = f.read()

        # Função interna para resolver cada placeholder encontrado
        def resolve_placeholder(match):
            placeholder_key = match.group(1)
            
            value = credentials.get(placeholder_key)
            if value is not None:
                return str(value)
            else:
                print(f"[utils] [!] ATENÇÃO: Placeholder '{placeholder_key}' não encontrado no arquivo credentials.yaml.")
                return f"ERRO_PLACEHOLDER_{placeholder_key}_NAO_ENCONTRADO_EM_CREDENCIALS"

        # Regex para encontrar placeholders como ${VARIAVEL} ou ${VAR_NOME_COMP}
        settings_str_resolved = re.sub(r"\$\{(\w+)\}", resolve_placeholder, settings_str)
        
        settings = yaml.safe_load(settings_str_resolved)
        if not settings:
            print(f"[utils] [✗] ERRO: Configurações em '{settings_path}' resultaram em objeto vazio após carregamento.")
            return None

        print(f"[utils] [✓] Configurações carregadas e processadas de '{settings_path}'.")
        return settings

    except Exception as e:
        print(f"[utils] [✗] ERRO ao carregar ou processar configurações de '{settings_path}': {e}")
        return None


# Carregamento de Dados
def load_initial_prompts(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # Adicionado encoding
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        if not prompts:
            print(f"[utils] [✗] Arquivo de prompts '{filepath}' está vazio.")
            return []
        print(f"[utils] {len(prompts)} prompts carregados de '{filepath}'.")
        return prompts
    except FileNotFoundError:
        print(f"[utils] [✗] Arquivo de prompts '{filepath}' não encontrado.")
        return []

def load_dataset(filepath):
    print(f"[utils] Carregando dataset de {filepath}")
    try:
        df = pd.read_csv(filepath)
        print(f"[utils] Dataset carregado com {len(df)} registros.")
        return df
    except FileNotFoundError:
        print(f"[utils] [✗] Dataset '{filepath}' não encontrado.")
        return None
    except Exception as e:
        print(f"[utils] [✗] Erro ao carregar dataset '{filepath}': {e}")
        return None


# Requisições a Modelos
def query_maritalk(full_prompt, model_config):
    model_name = model_config.get("name", "sabiazinho-3")
    api_key = model_config.get("chave_api")
    endpoint_url = model_config.get("endpoint")

    if not api_key or not endpoint_url:
        print(f"[utils] [✗] [Maritalk] API Key ou Endpoint não configurado para {model_name}.")
        return "erro_configuracao"

    # print(f"[utils] Consultando Maritalk ({model_name}) com prompt:\n{full_prompt[:100]}...")
    request_data = {
        "model": model_name, # Usa o nome do modelo do config
        "messages": [{"role": "user", "content": full_prompt}]
    }
    try:
        response = requests.post(
            url=endpoint_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("answer", "").strip().lower()
    except Exception as e:
        print(f"[utils] [✗] Erro ao consultar o Maritalk ({model_name}): {e}")
        return "erro_api"

def query_ollama(prompt, model_config):
    model_name = model_config.get("name")
    server_url = model_config.get("endpoint")

    if not model_name or not server_url:
        print(f"[utils] [✗] [Ollama] Nome do modelo ou URL do servidor não configurado.")
        return "erro_configuracao"
    
    if server_url.endswith("/api/generate"):
        chat_server_url = server_url.replace("/api/generate", "/api/chat")
    elif not server_url.endswith("/api/chat"): 
        chat_server_url = server_url.rstrip('/') + "/api/chat"
    else:
        chat_server_url = server_url


    # print(f"[utils] Consultando modelo '{model_name}' via Ollama ({chat_server_url})...")
    try:
        response = requests.post(
            url=chat_server_url,
            json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=15 # Aumentado um pouco o timeout
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip().lower()
    except requests.exceptions.RequestException as e:
        print(f"[utils] [✗] Erro ao consultar modelo Ollama '{model_name}': {e}")
        return "erro_api"


# Evolução com LLM (ex: GPT-4o)
def _call_openai_api(messages, generator_config, temperature=0.7):
    # Obtém os valores da configuração do gerador (que já devem ter sido resolvidos)
    local_api_key = generator_config.get("chave_api")
    local_api_base = generator_config.get("endpoint") # "endpoint" no YAML é o api_base para esta versão
    model_name = generator_config.get("name")

    # --- DEBUGGING CRUCIAL ---
    # Verifique se os valores estão chegando corretamente aqui.
    # Se local_api_key for None ou o placeholder, o problema está no carregamento/resolução das configs.
    print(f"[DEBUG _call_openai_api] Chave API recebida da config: {'*' * (len(local_api_key) - 4) + local_api_key[-4:] if local_api_key and len(local_api_key) > 4 else 'INDEFINIDA OU MUITO CURTA'}")
    print(f"[DEBUG _call_openai_api] Base API recebida da config: {local_api_base}")
    print(f"[DEBUG _call_openai_api] Modelo recebido da config: {model_name}")
    # --- FIM DEBUGGING ---

    if not local_api_key:
        print(f"[utils] [✗] [OpenAI] ERRO CRÍTICO: Chave API (chave_api) não fornecida ou não resolvida na 'generator_config'.")
        return "erro_configuracao_gerador_sem_chave_api"
    if not model_name:
        print(f"[utils] [✗] [OpenAI] Nome do modelo (name) não fornecido na 'generator_config'.")
        return "erro_configuracao_gerador_sem_modelo"

    # Salvar o estado global ATUAL do módulo openai antes de alterá-lo.
    # Isto é importante para "limpar" após a chamada e não afetar outras partes do código
    # que possam usar o módulo openai com configurações diferentes (ou nenhuma, esperando env vars).
    original_api_key = openai.api_key
    original_api_base = openai.api_base
    # Se você usa openai.organization, salve e restaure também:
    # original_organization = openai.organization

    try:
        # Define as configurações da API no módulo openai PARA ESTA CHAMADA
        openai.api_key = local_api_key
        if local_api_base: # Só define api_base se um valor foi fornecido no config
            openai.api_base = local_api_base
        else:
            # Se nenhum api_base for fornecido, e você quiser garantir o padrão da OpenAI,
            # você pode explicitamente resetá-lo para o default da biblioteca, mas geralmente
            # a biblioteca já usa o default se openai.api_base for None ou não definido.
            # Para 0.28.0, se openai.api_base não for setado ou for None, ele usa o default.
            # Se você setou para algo antes e quer voltar ao default, pode ser necessário
            # setar para None ou o valor default explícito da OpenAI.
            openai.api_base = "https://api.openai.com/v1" # Default para 0.28.0


        # DEBUG: Verifique o estado do módulo openai ANTES da chamada
        # print(f"[DEBUG _call_openai_api] openai.api_key ANTES da chamada: {openai.api_key}")
        # print(f"[DEBUG _call_openai_api] openai.api_base ANTES da chamada: {openai.api_base}")

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature
            # Você pode adicionar outros parâmetros como 'request_timeout' se necessário
        )
        # Para SDK 0.28.0, o acesso ao conteúdo da mensagem é assim:
        return response.choices[0].message["content"].strip()

    except openai.error.AuthenticationError as e: # Erro de autenticação específico do SDK 0.28.0
        print(f"[utils] [✗] ERRO DE AUTENTICAÇÃO com a API OpenAI (SDK 0.28.0): {e}. "
            f"Verifique se sua chave API (iniciando com '{local_api_key[:4]}' se válida) está correta e ativa.")
        return "erro_api_gerador_autenticacao"
    except openai.error.APIConnectionError as e:
        print(f"[utils] [✗] Erro de conexão com a API OpenAI (SDK 0.28.0) em '{openai.api_base}': {e}")
        return "erro_api_gerador_conexao"
    except openai.error.RateLimitError as e:
        print(f"[utils] [✗] Erro de limite de taxa (RateLimitError) da API OpenAI (SDK 0.28.0): {e}")
        return "erro_api_gerador_limite_taxa"
    except openai.error.InvalidRequestError as e: # Erro comum para requests malformados ou modelos inválidos
        print(f"[utils] [✗] Erro de requisição inválida para API OpenAI (SDK 0.28.0) para o modelo '{model_name}': {e}")
        return f"erro_api_gerador_requisicao_invalida"
    except openai.error.APIError as e: # Outros erros da API (e.g., erros 5xx do servidor OpenAI)
        http_status = e.http_status if hasattr(e, 'http_status') else 'N/A'
        http_body = e.http_body if hasattr(e, 'http_body') else 'N/A'
        print(f"[utils] [✗] Erro genérico da API OpenAI (SDK 0.28.0): Status={http_status}, Resposta={http_body}")
        return f"erro_api_gerador_status_{http_status}"
    except Exception as e: # Captura quaisquer outros erros inesperados
        print(f"[utils] [✗] Erro inesperado durante a chamada da API OpenAI (SDK 0.28.0) para o modelo '{model_name}': {type(e).__name__} - {e}")
        # import traceback # Para depuração mais profunda, se necessário
        # traceback.print_exc()
        return "erro_api_gerador_inesperado"
    finally:
        # **CRUCIAL: Restaura o estado global original do módulo openai**
        # Isso evita que esta chamada afete outras partes do seu código ou bibliotecas
        # que possam depender de uma configuração global diferente de `openai.api_key`.
        openai.api_key = original_api_key
        openai.api_base = original_api_base
        # openai.organization = original_organization # Se você usou

def crossover_and_mutation_ga(pair_of_parent_prompts, config):
    """
    Realiza a operação Evo: Crossover entre dois pais seguido de mutação no filho gerado.
    Retorna uma lista contendo um único dicionário de prompt filho.
    """
    generator_config = config.get("generator")
    if not generator_config:
        print("[utils] [✗] Configuração do gerador não encontrada.")
        return [{"prompt": "erro_configuracao_gerador"}]

    if len(pair_of_parent_prompts) != 2:
        print("[utils] [✗] crossover_and_mutation_ga espera exatamente dois pais.")
        # Poderia retornar um erro ou um prompt de erro, dependendo de como o chamador lida.
        return [{"prompt": "erro_numero_pais_invalido"}]


    template_generator = generator_config.get("template_generator", {})
    system_instruction = template_generator.get("system")
    user_instruction_crossover = template_generator.get("user_crossover")
    user_instruction_mutation = template_generator.get("user_mutation")

    prompt_a_dict = pair_of_parent_prompts[0]
    prompt_b_dict = pair_of_parent_prompts[1]

    prompt_a = prompt_a_dict["prompt"] if isinstance(prompt_a_dict, dict) else prompt_a_dict
    prompt_b = prompt_b_dict["prompt"] if isinstance(prompt_b_dict, dict) else prompt_b_dict

    # Crossover
    crossover_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_instruction_crossover.format(prompt_a=prompt_a, prompt_b=prompt_b)}
    ]
    crossover_prompt = _call_openai_api(crossover_messages, generator_config)
    if "erro_" in crossover_prompt: # Verifica erro na geração
        return [{"prompt": f"prompt_gerado_com_erro_crossover ({crossover_prompt})"}]

    # Mutação
    mutation_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_instruction_mutation.format(prompt=crossover_prompt)}
    ]
    mutated_prompt = _call_openai_api(mutation_messages, generator_config)
    if "erro_" in mutated_prompt:
        return [{"prompt": f"prompt_gerado_com_erro_mutacao ({mutated_prompt})"}]
    else:
        return [{"prompt": mutated_prompt}]


# Avaliação
def evaluate_prompt_single(prompt_instruction: str, text: str, label: int,
                        evaluator_config: dict, strategy_config: dict,
                        experiment_settings: dict) -> tuple[int, str]:
    """
    Avalia um único prompt para um dado texto e label, usando a estratégia e avaliador configurados.
    Retorna (predição, texto_da_resposta_do_llm).
    """
    strategy_name = strategy_config.get("name", "desconhecida").lower() # Normaliza para minúsculas
    template_str = strategy_config.get("template")

    if not isinstance(prompt_instruction, str): # Garante que a instrução do prompt é uma string
        prompt_instruction = str(prompt_instruction)
    if not isinstance(text, str): # Garante que o texto de entrada é uma string
        text = str(text)

    if not template_str:
        print(f"[utils] [✗] Template não encontrado para a estratégia '{strategy_name}'.")
        # Considera predição incorreta e retorna mensagem de erro
        return 1 - label, "erro_template_ausente_na_configuracao_da_estrategia"

    # Instrução para o LLM sobre o formato da resposta
    instruction_suffix = "\nResponda apenas com a palavra 'positivo' ou 'negativo'."
    
    # Argumentos base para a formatação do template
    format_args = {
        "text": text,
        "prompt_instruction": prompt_instruction # Será sobrescrito se o sufixo for adicionado
    }

    if strategy_name == "cot":
        # Para CoT, o prompt_instruction original é usado.
        # O template CoT no YAML já inclui a instrução de formato de resposta no final.
        # Placeholders esperados no template CoT: {prompt_instruction}, {text}
        # `format_args` já os contém.
        pass # Nenhuma modificação adicional em format_args é necessária para CoT.
    
    elif strategy_name == "few-shot":
        # Adiciona o sufixo à instrução principal.
        format_args["prompt_instruction"] = prompt_instruction + instruction_suffix
        # Adiciona examples. strategy_config.get("examples", "") garante que é uma string.
        format_args["examples"] = strategy_config.get("examples", "")
        # Placeholders esperados no template few-shot: {prompt_instruction}, {examples}, {text}
        
    elif strategy_name == "zero-shot":
        # Adiciona o sufixo à instrução principal.
        format_args["prompt_instruction"] = prompt_instruction + instruction_suffix
        # Placeholders esperados no template zero-shot: {prompt_instruction}, {text}
        # Se o template zero-shot *não* contiver {examples}, ele será ignorado por .format(**format_args).
        # Se o template zero-shot *contiver* {examples}, e `examples` não estiver em format_args, dará KeyError.
        # Para garantir, podemos adicionar um `examples` vazio se necessário, ou garantir que o template não o tenha.
        # Como `examples` não é adicionado a `format_args` aqui, o template zero-shot NÃO DEVE ter `{examples}`.
        
    else:
        # Fallback para estratégias desconhecidas ou não explicitamente tratadas
        print(f"[utils] [!] Estratégia '{strategy_name}' não reconhecida explicitamente. "
            "Aplicando formatação genérica (instrução com sufixo e todos os campos possíveis).")
        format_args["prompt_instruction"] = prompt_instruction + instruction_suffix
        format_args["examples"] = strategy_config.get("examples", "") # Inclui para o caso genérico

    # Monta o prompt final
    try:
        # Descomente para depuração intensiva:
        # print(f"\n[DEBUG] Strategy: {strategy_name}")
        # print(f"[DEBUG] Template String:\n```\n{template_str}\n```")
        # print(f"[DEBUG] Arguments for .format():")
        # for k, v in format_args.items():
        #     print(f"  - {k}: \"{str(v)[:100]}{'...' if len(str(v)) > 100 else ''}\"")

        full_prompt = template_str.format(**format_args)
        
        print(f"[DEBUG] Full Prompt Generated for '{strategy_name}':\n------BEGIN PROMPT------\n{full_prompt}\n-------END PROMPT-------")

    except KeyError as e:
        print(f"[utils] [✗] ERRO DE FORMATAÇÃO para estratégia '{strategy_name}': "
            f"O placeholder {e} é esperado no template, mas não foi fornecido nos argumentos de formatação.")
        print(f"   Template problemático:\n```\n{template_str}\n```")
        print(f"   Argumentos fornecidos para .format(): {list(format_args.keys())}")
        return 1 - label, f"erro_formatacao_template_chave_ausente_{e}"
    except Exception as e_gen:
        print(f"[utils] [✗] ERRO GENÉRICO DE FORMATAÇÃO para estratégia '{strategy_name}': {e_gen}")
        return 1 - label, "erro_formatacao_template_desconhecido"

    # --- Chamada ao LLM avaliador ---
    response_text = "erro_interno_chamada_llm" # Default em caso de falha na chamada
    evaluator_type = evaluator_config.get("tipo", "desconhecido").lower()

    # print(f"[DEBUG] Enviando para avaliador tipo '{evaluator_type}':\n{full_prompt[:200]}...")

    if evaluator_type == "maritalk":
        response_text = query_maritalk(full_prompt, evaluator_config)
    elif evaluator_type == "ollama":
        response_text = query_ollama(full_prompt, evaluator_config)
    # Adicionar outros tipos de avaliador aqui (ex: openai)
    # elif evaluator_type == "openai":
    #    response_text = query_openai(full_prompt, evaluator_config) # Você precisaria criar query_openai
    else:
        print(f"[utils] [✗] Tipo de avaliador desconhecido ou não implementado: '{evaluator_type}'")
        response_text = "erro_tipo_avaliador_nao_configurado"

    # --- Extração da predição ---
    prediction = extract_label(response_text)
    if prediction is None:
        # Se extract_label falhar (incluindo erros de API retornados como texto ou config)
        # ou se o LLM não responder no formato esperado.
        prediction = 1 - label # Considera como predição incorreta
        
    return prediction, response_text

def evaluate_prompt(prompt_instruction, dataset, evaluator_config, strategy_config, experiment_settings):
    predictions = []
    true_labels = dataset["label"].tolist()
    texts = dataset["text"].tolist()

    evaluator_name_sanitized = evaluator_config["name"].replace(":", "_").replace("/", "_")
    strategy_name_sanitized = strategy_config["name"]
    log_dir = "logs/prompt_eval_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"eval_{evaluator_name_sanitized}_{strategy_name_sanitized}.csv")

    # Cabeçalho do log se o arquivo não existir
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("prompt_instruction,text,true_label,llm_response,predicted_label\n")

    with open(log_path, "a", encoding="utf-8") as log_file:
        for text, label in zip(texts, true_labels):
            predicted, response_text = evaluate_prompt_single(
                prompt_instruction, text, label,
                evaluator_config, strategy_config, experiment_settings
            )
            predictions.append(predicted)
            log_file.write(f"\"{prompt_instruction}\",\"{text.replace('\"', '\"\"')}\",{label},\"{str(response_text).replace('\"', '\"\"')}\",{predicted}\n")


    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    tokens = count_tokens(prompt_instruction)

    alert_message = ""
    if precision == 0 and recall == 0 and any(p == 1 for p in predictions): # Se previu positivo mas errou todos
        alert_message = "Previsões positivas feitas, mas todas incorretas."
    elif not any(p == 1 for p in predictions) and any(l == 1 for l in true_labels): # Não previu positivo, mas havia positivos
        alert_message = "Nenhuma previsão positiva feita, mas existiam exemplos positivos."

    return acc, f1, tokens, alert_message

def extract_label(text: str) -> int | None:
    text_lower = text.lower().strip()
    # Verifica frases mais explícitas primeiro
    if "classificação: positivo" in text_lower or text_lower == "positivo":
        return 1
    elif "classificação: negativo" in text_lower or text_lower == "negativo":
        return 0

    # Falsos positivos (ex: "isso não é positivo" -> seria pego como positivo)
    # A instrução no prompt é a melhor defesa.
    if "positivo" in text_lower: # Pode precisar de mais lógica se "não positivo" for uma resposta
        return 1
    elif "negativo" in text_lower:
        return 0
    return None # Se nenhuma etiqueta clara for encontrada

def count_tokens(prompt: str) -> int:
    return len(tokenizer.tokenize(prompt))


#  Funções Monoobjetivo 
def roulette_wheel_selection(population, num_parents_to_select):
    """
    Seleciona pais da população usando o método da roleta.
    A fitness é baseada na acurácia (metrics[0]).
    """
    selected_parents = []

    if not population:
        print("[utils] [!] População vazia para seleção por roleta.")
        return selected_parents

    # Extrai fitness (acurácia) e garante que são não-negativas
    # individuals_with_fitness = [
    #     (ind, ind["metrics"][0]) for ind in population if "metrics" in ind and ind["metrics"] and ind["metrics"][0] >= 0
    # ]

    fitness_values = []
    valid_individuals = []
    for ind in population:
        if "metrics" in ind and isinstance(ind["metrics"], (list, tuple)) and len(ind["metrics"]) > 0:
            fitness = ind["metrics"][0] # Acurácia
            if fitness >= 0: # Acurácia deve ser >= 0
                fitness_values.append(fitness)
                valid_individuals.append(ind)
            else:
                print(f"[utils] [!] Indivíduo com fitness negativa ignorado na roleta: {fitness}")
        else:
            print(f"[utils] [!] Indivíduo sem métrica de fitness válida ignorado na roleta: {ind.get('prompt')}")

    if not valid_individuals:
        print("[utils] [!] Nenhum indivíduo com fitness válida para seleção por roleta.")
        # Fallback: selecionar aleatoriamente da população original se não houver fitness válida.
        # Ou simplesmente retornar lista vazia. Para este caso, vamos selecionar aleatoriamente se possível.
        if population and len(population) >= num_parents_to_select:
            return random.sample(population, num_parents_to_select)
        return selected_parents # Retorna vazio

    total_fitness = sum(fitness_values)

    if total_fitness == 0:
        # Se todas as fitness válidas são 0, seleciona aleatoriamente entre os indivíduos válidos
        # print("[utils] [!] Fitness total é 0. Selecionando pais aleatoriamente entre os válidos.")
        if len(valid_individuals) >= num_parents_to_select:
            return random.sample(valid_individuals, num_parents_to_select)
        else: # Se não houver indivíduos suficientes para selecionar
            return valid_individuals # Retorna o que tiver

    probabilities = [f / total_fitness for f in fitness_values]

    # Para selecionar pais, geralmente é sem reposição para um par, mas com reposição entre pares.
    # Se num_parents_to_select é 2 (para um filho), e você chama isso N vezes, está correto.
    
    if not probabilities or len(valid_individuals) == 0 : # Checagem de segurança
        return []

    # Se num_parents_to_select = 2, e eles DEVEM ser diferentes:
    if num_parents_to_select == 2 and len(valid_individuals) >= 2:
        indices = np.random.choice(len(valid_individuals), size=num_parents_to_select, p=probabilities, replace=False)

        idx1 = np.random.choice(len(valid_individuals), p=probabilities)
        selected_parents.append(valid_individuals[idx1])

        if num_parents_to_select > 1: # Se precisar de mais um pai
            # A roleta padrão amostra com reposição.
            idx2 = np.random.choice(len(valid_individuals), p=probabilities)
            # Para garantir que pai1 != pai2, se possível:
            max_retries = 10 # Evitar loop infinito se a população for muito pequena e homogênea
            count = 0
            while idx2 == idx1 and len(valid_individuals) > 1 and count < max_retries:
                idx2 = np.random.choice(len(valid_individuals), p=probabilities)
                count += 1
            selected_parents.append(valid_individuals[idx2])

    elif len(valid_individuals) > 0: # Para outros casos de num_parents_to_select
        indices = np.random.choice(len(valid_individuals), size=num_parents_to_select, p=probabilities, replace=True)
        selected_parents = [valid_individuals[i] for i in indices]

    return selected_parents

def selection_accuracy(population, k, num_selected):
    # print(f"[utils] Seleção por torneio (k={k}, selecionados={num_selected})...")
    selected_parents = [] # Renomeado para clareza
    if not population: # Guarda contra população vazia
        print("[utils] [!] População vazia para seleção por torneio.")
        return []
    if len(population) < k: # Guarda se a população for menor que o tamanho do torneio
        print(f"[utils] [!] População (tam: {len(population)}) menor que k ({k}). Usando todos os indivíduos como candidatos.")
        k = len(population)

    while len(selected_parents) < num_selected:
        if not population: break # Segurança adicional
        
        candidates = random.sample(population, k)
        # Ordena por acurácia (metrics[0]) descendente, tokens (metrics[2]) ascendente
        candidates.sort(key=lambda x: (x["metrics"][0], -x["metrics"][2]), reverse=True)
        selected_parents.append(candidates[0]) # Adiciona o melhor do torneio
    return selected_parents

def select_population(parents, offspring, config):
    # print("[utils] Selecionando próxima geração ")
    combined = parents + offspring 
    selected = sorted(
        combined,
        key=lambda x: (x["metrics"][0] if "metrics" in x and x["metrics"] else -1,
                    -(x["metrics"][2] if "metrics" in x and x["metrics"] and len(x["metrics"]) > 2 else float('inf'))),
        reverse=True
    )[:config.get("population_size", 10)]
    print(f"[utils] População selecionada com {len(selected)} indivíduos.")
    print(f"[utils] População selecionada:  {selected}.")
    return selected


# Funções Multiobjetivo
def dominates(ind_a_objectives, ind_b_objectives):
    """
    Verifica se o indivíduo A domina o indivíduo B.
    Objetivos: 'acc' (maximizar), 'tokens' (minimizar).
    ind_a_objectives e ind_b_objectives são dicionários como {'acc': x, 'tokens': y}
    """
    a_is_better_or_equal = (ind_a_objectives["acc"] >= ind_b_objectives["acc"] and
                            ind_a_objectives["tokens"] <= ind_b_objectives["tokens"])
    a_is_strictly_better = (ind_a_objectives["acc"] > ind_b_objectives["acc"] or
                            ind_a_objectives["tokens"] < ind_b_objectives["tokens"])
    return a_is_better_or_equal and a_is_strictly_better

def compute_pareto_front(evaluated_pop):
    print("[utils] Calculando fronte de Pareto...")
    return [a for a in evaluated_pop if not any(dominates(b, a) for b in evaluated_pop if b != a)]

def fast_non_dominated_sort(population_with_objectives):
    """
    Realiza a classificação não-dominada rápida na população.
    Cada indivíduo em population_with_objectives é um dicionário
    que deve conter as chaves dos objetivos (ex: 'acc', 'tokens') e 'prompt'.
    Adiciona 'rank' (número da fronteira, 0 é a melhor) a cada indivíduo.
    Retorna uma lista de fronteiras (fronts), onde cada fronteira é uma lista de indivíduos.
    """
    # print("[utils] Executando Fast Non-Dominated Sort...")
    fronts = [[]] # Lista de fronteiras, fronts[0] é a F_1

    for p_idx, p_ind in enumerate(population_with_objectives):
        # Adicionar atributos para o sort se não existirem (ou resetar)
        p_ind['dominated_solutions_indices'] = [] # Índices das soluções que p domina
        p_ind['domination_count'] = 0          # Número de soluções que dominam p
        # p_ind['rank'] será atribuído

        for q_idx, q_ind in enumerate(population_with_objectives):
            if p_idx == q_idx:
                continue
            
            # Usar apenas os valores dos objetivos para a função 'dominates'
            if dominates(p_ind, q_ind): # p domina q
                p_ind['dominated_solutions_indices'].append(q_idx)
            elif dominates(q_ind, p_ind): # q domina p
                p_ind['domination_count'] += 1
        
        if p_ind['domination_count'] == 0:
            p_ind['rank'] = 0 # Rank 0 para a primeira fronteira
            fronts[0].append(p_ind)

    current_rank_idx = 0
    while fronts[current_rank_idx]: # Enquanto a fronteira atual não estiver vazia
        next_front_individuals = []
        for p_ind in fronts[current_rank_idx]: # Para cada solução p na fronteira F_i
            for q_idx in p_ind['dominated_solutions_indices']: # Para cada solução q dominada por p
                q_ind_ref = population_with_objectives[q_idx]
                q_ind_ref['domination_count'] -= 1
                if q_ind_ref['domination_count'] == 0:
                    q_ind_ref['rank'] = current_rank_idx + 1
                    next_front_individuals.append(q_ind_ref)
        current_rank_idx += 1
        if next_front_individuals:
            fronts.append(next_front_individuals)
        else:
            break # Não há mais indivíduos para as próximas fronteiras
            
    # print(f"[utils] Non-Dominated Sort concluído. Encontradas {len(fronts)} fronteiras.")
    return fronts

def compute_crowding_distance(front_individuals):
    """
    Calcula a distância de crowding para indivíduos DENTRO DE UMA MESMA FRONTEIRA.
    Assume que front_individuals é uma lista de dicionários, cada um com 'acc' e 'tokens'.
    Adiciona a chave 'crowding_distance' a cada indivíduo.
    """
    if not front_individuals:
        return []

    num_individuals = len(front_individuals)
    for ind in front_individuals:
        ind['crowding_distance'] = 0.0

    # Objetivos a serem considerados. Lembre-se: 'acc' é maximizado, 'tokens' é minimizado.
    # Para o cálculo da distância, a ordenação deve ser consistente.
    # Se o objetivo é minimizado, ordena-se ascendente. Se maximizado, descendente.
    # Ou, mais simples, sempre ordena ascendente e inverte o valor do objetivo de maximização.
    
    objectives = {'acc': True, 'tokens': False} # True se maior é melhor, False se menor é melhor

    for obj_key, maximize in objectives.items():
        # Ordena os indivíduos da fronteira pelo valor do objetivo atual
        # Se for para maximizar, ordena descendente para que o primeiro seja o "melhor" (maior valor)
        # Se for para minimizar, ordena ascendente para que o primeiro seja o "melhor" (menor valor)
        # Para o cálculo da distância, a ordenação é geralmente ascendente no valor do objetivo.
        # Se um objetivo é de maximização (como 'acc'), podemos ordenar por -acc (ascendente).
        
        # Para simplificar: sempre ordenamos de forma que o "melhor" venha primeiro
        # No cálculo da distância, a ordenação é pelo valor bruto do objetivo.
        # A direção (max/min) importa para a função 'dominates', não tanto aqui, desde que consistente.
        
        sorted_front = sorted(front_individuals, key=lambda x: x[obj_key])
        
        # Atribui distância infinita às soluções extremas para este objetivo
        sorted_front[0]['crowding_distance'] = float('inf')
        if num_individuals > 1: # Só faz sentido se houver mais de um
            sorted_front[num_individuals - 1]['crowding_distance'] = float('inf')

        # Se todos os valores do objetivo são iguais, não contribui para a distância
        min_obj_val = sorted_front[0][obj_key]
        max_obj_val = sorted_front[num_individuals - 1][obj_key]
        range_obj = max_obj_val - min_obj_val

        if range_obj == 0: # Todos os valores são iguais para este objetivo
            continue

        # Calcula a distância para as soluções intermediárias
        for i in range(1, num_individuals - 1):
            # Evita adicionar inf se já for inf por ser extremo em outro objetivo
            if sorted_front[i]['crowding_distance'] != float('inf'):
                sorted_front[i]['crowding_distance'] += \
                    (sorted_front[i+1][obj_key] - sorted_front[i-1][obj_key]) / range_obj
    
    return front_individuals # Retorna os indivíduos com a 'crowding_distance' atualizada

def tournament_selection_multiobjective(population_with_rank_and_crowding, k_tournament_size, num_to_select):
    """
    Seleção por torneio para pais baseada no rank de Pareto e crowding distance (estilo NSGA-II).
    Seleciona 'num_to_select' pais.
    """
    selected_parents = []
    population_size = len(population_with_rank_and_crowding)

    if population_size == 0: return []
    if k_tournament_size > population_size: k_tournament_size = population_size # Ajusta k se for maior que a pop

    for _ in range(num_to_select):
        tournament_candidates_indices = random.sample(range(population_size), k_tournament_size)
        tournament_candidates = [population_with_rank_and_crowding[i] for i in tournament_candidates_indices]
        
        best_candidate = tournament_candidates[0]
        for i in range(1, k_tournament_size):
            candidate = tournament_candidates[i]
            # Compara rank (menor é melhor)
            if candidate['rank'] < best_candidate['rank']:
                best_candidate = candidate
            elif candidate['rank'] == best_candidate['rank']:
                # Se o rank é o mesmo, compara crowding distance (maior é melhor)
                if candidate['crowding_distance'] > best_candidate['crowding_distance']:
                    best_candidate = candidate
        selected_parents.append(best_candidate) # Adiciona o vencedor do torneio
        
    return selected_parents


# Persistência
def save_generation_results(population, generation, config):
    # Alinhado com save_sorted_population para incluir mais detalhes
    base_log_path = config.get("generation_log_dir", "logs/generations_detail") 
    os.makedirs(base_log_path, exist_ok=True)
    
    # Usa o nome do modelo e estratégia do config para nomear o arquivo de log
    evaluator_name = config.get("evaluators", [{}])[0].get("name", "unknown_model").replace(":", "_").replace("/", "_")
    strategy_name = config.get("strategies", [{}])[0].get("name", "unknown_strategy")
    
    path = os.path.join(base_log_path, f"results_gen_{generation}_{evaluator_name}_{strategy_name}.csv")

    data = []
    for ind in population:
        prompt = ind["prompt"]
        # Assegura que 'metrics' existe e tem o formato esperado
        if "metrics" in ind and isinstance(ind["metrics"], (list, tuple)) and len(ind["metrics"]) >= 4:
            acc, f1, tokens, alert_message = ind["metrics"][:4]
        else: # Fallback se metrics estiver ausente ou malformado
            acc, f1, tokens, alert_message = 0.0, 0.0, 0, "metrics_data_missing_or_malformed"

        data.append({
            "generation": generation,
            "prompt": prompt,
            "accuracy": acc,
            "f1_score": f1,
            "tokens": tokens,
            "alert": alert_message
        })

    df = pd.DataFrame(data)
    # Ordena pela acurácia (descendente) e depois por tokens (ascendente) 
    df = df.sort_values(by=["accuracy", "tokens"], ascending=[False, True])
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"[utils] Resultados detalhados da geração {generation} salvos em {path}")

def save_sorted_population(population, generation, base_log_path="logs/generations"):
    os.makedirs(base_log_path, exist_ok=True)
    sorted_log_path = os.path.join(base_log_path, f"population_sorted_gen_{generation}.csv")

    data = []
    for ind in population:
        prompt = ind["prompt"]
        # Assumindo que metrics é (acc, f1, tokens, alert_message)
        acc, f1, tokens, alert_message = ind["metrics"][:4]
        data.append({
            "generation": generation,
            "prompt": prompt,
            "accuracy": acc,
            "f1_score": f1,
            "tokens": tokens,
            "alert": alert_message
        })

    df = pd.DataFrame(data)
    # Ordena pela acurácia (descendente) e depois por tokens (ascendente) como critério de desempate
    df = df.sort_values(by=["accuracy", "tokens"], ascending=[False, True])
    df.to_csv(sorted_log_path, index=False, encoding='utf-8')
    print(f"[utils] População ordenada da geração {generation} salva em {sorted_log_path}")

def save_final_results(population, config, output_csv_path): # output_csv_path vindo do main.py
    print("[utils] Salvando resultados finais...")
    
    data = []
    for ind in population: # Aqui, population é a lista dos top N finais
        prompt = ind["prompt"]
        if "metrics" in ind and isinstance(ind["metrics"], (list, tuple)) and len(ind["metrics"]) >= 4:
            acc, f1, tokens, alert_message = ind["metrics"][:4]
            data.append({
                "prompt": prompt,
                "accuracy": acc,
                "f1_score": f1,
                "tokens": tokens,
                "alert": alert_message
            })
        else:
            data.append({
                "prompt": prompt,
                "accuracy": 0.0, "f1_score": 0.0, "tokens": 0, "alert": "metrics_final_missing"
            })

    df = pd.DataFrame(data)
    # Se quiser garantir a ordenação aqui também, embora já deva vir ordenada:
    # df = df.sort_values(by=["accuracy", "tokens"], ascending=[False, True])
    
    # Salva os top_k definidos no YAML (ou todos se top_k for maior que a população)
    top_k = config.get("top_k", len(df))
    df_top_k = df.head(top_k)

    df_top_k.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"[utils] {len(df_top_k)} melhores resultados finais salvos em {output_csv_path}")

def save_pareto_front_data(front_individuals, csv_path, plot_path): # Renomeado para clareza
    """Salva os dados da fronteira de Pareto em CSV e gera um gráfico."""
    if not front_individuals:
        print("[utils] [!] Tentativa de salvar fronteira de Pareto vazia.")
        # Criar um CSV vazio com cabeçalhos ou simplesmente não fazer nada
        df_empty = pd.DataFrame(columns=["prompt", "acc", "f1", "tokens", "rank", "crowding_distance"]) # Exemplo
        df_empty.to_csv(csv_path, index=False)
        # Para o plot, pode-se criar um plot vazio ou uma mensagem
        plt.figure()
        plt.text(0.5, 0.5, "Fronteira de Pareto Vazia", ha='center', va='center')
        plt.xlabel("Número de Tokens")
        plt.ylabel("Acurácia")
        plt.title("Fronte de Pareto (Tokens x Acurácia)")
        plt.savefig(plot_path)
        plt.close()
        return

    # Monta DataFrame apenas com os campos relevantes para o CSV/Plot
    data_to_save = []
    for ind in front_individuals:
        data_to_save.append({
            "prompt": ind.get("prompt", "N/A"),
            "acc": ind.get("acc", 0.0),
            "f1": ind.get("f1", 0.0), # Adicionando f1 se existir
            "tokens": ind.get("tokens", 0),
            "rank": ind.get("rank", -1), # Se rank estiver disponível
            "crowding_distance": ind.get("crowding_distance", 0.0) # Se crowding_distance estiver disponível
        })
    df = pd.DataFrame(data_to_save)
    
    # Ordenar para o CSV para consistência, se desejado (ex: por acurácia)
    df_sorted = df.sort_values(by="acc", ascending=False)
    df_sorted.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"[utils] Resultados da fronte de Pareto salvos em {csv_path}")

    # Plotar apenas 'acc' vs 'tokens'
    plt.figure(figsize=(10, 6))
    plt.scatter(df["tokens"], df["acc"], c='blue', alpha=0.7, edgecolors='w', s=70)
    plt.xlabel("Número de Tokens (Menor é Melhor)")
    plt.ylabel("Acurácia (Maior é Melhor)")
    plt.title("Fronteira de Pareto (Tokens vs Acurácia)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Opcional: anotar alguns pontos no gráfico
    # for i, row in df.iterrows():
    #     plt.annotate(f"P{i}", (row["tokens"], row["acc"]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        
    plt.savefig(plot_path)
    plt.close()
    print(f"[utils] Gráfico da fronteira de Pareto salvo em {plot_path}")
