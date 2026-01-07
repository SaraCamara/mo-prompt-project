import requests
import openai
import re
import logging

# Seção: Requisições a Modelos
logger = logging.getLogger(__name__)

def query_maritalk(full_prompt, model_config):
    model_name = model_config.get("name", "sabiazinho-3")
    api_key = model_config.get("chave_api")
    endpoint_url = model_config.get("endpoint")
    if not api_key or not endpoint_url:
        logger.error(f"[Maritalk] API Key ou Endpoint não configurado para {model_name}.")
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
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro ao consultar o Maritalk ({model_name}): {e}")
        return "erro_api"


def query_ollama(prompt, model_config):
    model_name = model_config.get("name")
    server_url = model_config.get("endpoint")
    if not model_name or not server_url:
        logger.error(f"[Ollama] Nome do modelo ou URL do servidor não configurado.")
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
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip().lower()
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro ao consultar modelo Ollama '{model_name}': {e}")
        return "erro_api"


def _call_openai_api(messages, generator_config, temperature=0.8):
    local_api_key = generator_config.get("chave_api")
    local_api_base = generator_config.get("endpoint")
    model_name = generator_config.get("name")
    if not local_api_key:
        logger.critical("ERRO CRÍTICO: Chave API não fornecida.")
        return "erro_configuracao_gerador_sem_chave_api"
    if not model_name:
        logger.error("Nome do modelo não fornecido.")
        return "erro_configuracao_gerador_sem_modelo"
    original_api_key = openai.api_key
    original_api_base = openai.api_base
    try:
        openai.api_key = local_api_key
        openai.api_base = local_api_base if local_api_base else "https://api.openai.com/v1"
        logger.debug(f"[OpenAI] Requisição para '{model_name}': {messages}")
        response = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=temperature)
        logger.debug(f"[OpenAI] Resposta de '{model_name}': {response.choices[0].message['content'].strip()}")
        return response.choices[0].message["content"].strip()
    except openai.error.AuthenticationError as e:
        logger.error(f"ERRO DE AUTENTICAÇÃO com a API OpenAI: {e}.")
        return "erro_api_gerador_autenticacao"
    except Exception as e:
        logger.error(f"Erro inesperado durante a chamada da API OpenAI para '{model_name}': {type(e).__name__} - {e}")
        return "erro_api_gerador_inesperado"
    finally:
        openai.api_key = original_api_key
        openai.api_base = original_api_base