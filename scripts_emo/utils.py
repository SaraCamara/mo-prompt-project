import os
import yaml
import pandas as pd
import requests
import openai
import re
from sklearn.metrics import accuracy_score, f1_score
from nltk.tokenize import TreebankWordTokenizer

### === Carregamento de Configurações ===

def load_credentials(path):
    with open(path, "r") as f:
        creds = yaml.safe_load(f)

    openai.api_key = creds["openai_api_key"]
    openai.api_base = creds["openai_api_base"]

    return creds

def load_settings(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

### === Prompt Construction ===

def build_prompt(prompt_instruction, text, strategy):
    if strategy == "zero-shot":
        return (
            f"{prompt_instruction}\n\n"
            f"Texto: \"{text}\"\n"
            f"Resposta:"
        )

    elif strategy == "few-shot":
        examples = (
            "Texto: \"Esta versão de Anna Christie está em alemão. Greta Garbo interpreta Anna Christie, mas todos os outros personagens têm atores diferentes da versão em inglês. Ambos foram filmados para trás porque Garbo teve um número de seguidores na Alemanha. É uma boa história e um imperdível para os fãs de Garbo.\"\nClassificação: positivo\n"
            "Texto: \"Não é apenas um filme mal feito com baixo orçamento, mas a trama em si é apenas estúpida !!! Um homem místico que come mulheres?(E pela aparência, não virgem) ridículo !!!Se você não tem nada melhor para fazer (como dormir), deve assistir isso. Okay, certo.\"\nClassificação: negativo\n"
        )
        return (
            f"{prompt_instruction}\n\n"
            f"{examples}"
            f"Texto: \"{text}\"\nClassificação:"
        )

    elif strategy == "cot":
        return (
            f"{prompt_instruction}\n\n"
            f"Texto: \"{text}\"\n\n"
            f"Explicação: Analise o sentimento do seguinte texto. Primeiro explique por que é positivo ou negativo, depois informe a classificação final."
        )

### === Avaliação de Resposta ===

def extract_label(response_text):
    response = response_text.lower()

    # Regras principais
    if "positivo" in response:
        return 1
    elif "negativo" in response:
        return 0

    # Regras alternativas
    if "1" in response:
        return 1
    elif "0" in response:
        return 0

    return None

tokenizer = TreebankWordTokenizer()

def count_tokens(prompt: str) -> int:
    return len(tokenizer.tokenize(prompt))

def evaluate_prompt(prompt_instruction, model, strategy, df):
    preds = []
    gold = df["label"].tolist()
    
    creds_path = os.path.expanduser("config/credentials.yaml")
    with open(creds_path) as f:
        creds = yaml.safe_load(f)

    settings_path = os.path.expanduser("config/experiment_settings.yaml")
    with open(settings_path) as f:
        settings = yaml.safe_load(f)

    for text, true_label in zip(df["text"], gold):
        full_prompt = build_prompt(prompt_instruction, text, strategy)

        if model == "maritalk":
            response = query_maritalk(full_prompt)
        else:
            response = query_ollama(full_prompt, model, settings["ollama_server_url"])

        label = extract_label(response)
        if label is None:
            label = 1 - true_label
            print(f"Resposta inesperada: '{response}' → Inferido: {label}")

        preds.append(label)

    num_tokens = count_tokens(prompt_instruction)
    return accuracy_score(gold, preds), f1_score(gold, preds), num_tokens


### === Consulta à Maritalk ===

def query_maritalk(full_prompt):
    request_data = {
        "model": "sabiazinho-3",
        "messages": [{"role": "user", "content": full_prompt}]
    }

    try:
        creds_path = os.path.expanduser("config/credentials.yaml")
        with open(creds_path) as f:
            creds = yaml.safe_load(f)

        response = requests.post(
            url=creds["sabia_url"],
            headers={
                "Authorization": f"Bearer {creds['sabia_api_key']}",
                "Content-Type": "application/json"
            },
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("answer", "").strip().lower()
    except Exception as e:
        print(f"Erro na requisição para Maritalk: {e}")
        return "erro"

### === Consulta usando Ollama ===
def query_ollama(prompt, model_name, server_url):
    try:
        response = requests.post(
            url=server_url.replace("/generate", "/chat"),  # corrige endpoint
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        raw_response = data.get("message", {}).get("content", "").strip().lower()
        if not raw_response:
            print(f"Resposta vazia do modelo '{model_name}' para o prompt.")
        return raw_response
    except requests.exceptions.RequestException as e:
        print(f"[✗] Erro ao consultar modelo '{model_name}' via Ollama: {e}")
        return "erro"


### === Geração de novo prompt com gpt-4o-mini ===

def generate_prompt_with_gpt4o(top_prompts_with_scores):
    prompt_text = "\n".join([f"{i+1}. {p} (F1: {s:.2f})" for i, (p, s) in enumerate(top_prompts_with_scores)])
    system_instruction = "Você é um otimizador de prompts para classificação de sentimentos."
    user_instruction = (
        "Com base nos prompts abaixo e seus desempenhos, gere um novo prompt em português "
        "para a tarefa de classificação de sentimento (positivo ou negativo). Gere apenas o prompt:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"{user_instruction}\n\n{prompt_text}"}
            ],
            temperature=0.7,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"Erro ao gerar prompt com gpt-4o-mini Mini: {e}")
        return "Prompt gerado com erro"
