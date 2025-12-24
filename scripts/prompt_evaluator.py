import os
import concurrent.futures
from scripts.llm_clients import query_maritalk, query_ollary
from scripts.evaluation_metrics import extract_label, compute_exact, compute_f1, count_tokens, calculate_imdb_metrics, calculate_squad_metrics

# Seção: Avaliação de Prompts

def evaluate_prompt_single(prompt_instruction: str, text: str, label: int,
                        evaluator_config: dict, strategy_config: dict,
                        experiment_settings: dict) -> tuple[int, str]:
    strategy_name = strategy_config.get("name", "desconhecida").lower()
    template_str = strategy_config.get("template")
    if not isinstance(prompt_instruction, str): prompt_instruction = str(prompt_instruction)
    if not isinstance(text, str): text = str(text)
    if not template_str:
        print(f"[prompt_evaluator] Template não encontrado para a estratégia '{strategy_name}'.")
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
        print(f"[prompt_evaluator] ERRO DE FORMATAÇÃO para '{strategy_name}': placeholder {e} ausente.")
        return 1 - label, f"erro_formatacao_template"
    evaluator_type = evaluator_config.get("tipo", "").lower()
    if evaluator_type == "maritalk": response_text = query_maritalk(full_prompt, evaluator_config)
    elif evaluator_type == "ollama": response_text = query_ollama(full_prompt, evaluator_config)
    else:
        print(f"[prompt_evaluator] Tipo de avaliador desconhecido: '{evaluator_type}'")
        response_text = "erro_tipo_avaliador"
    prediction = extract_label(response_text)
    if prediction is None:
        prediction = 1 - label
    return prediction, response_text

def evaluate_prompt_single_squad(prompt_instruction: str, context: str, question: str, executor_config: dict, strategy_config: dict) -> str:
    strategy_name = strategy_config.get("name", "desconhecida").lower()
    template_str = strategy_config.get("template")
    
    if not template_str:
        print(f"[prompt_evaluator] Template não encontrado para a estratégia SQuAD.")
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
        print(f"[prompt_evaluator] ERRO DE FORMATAÇÃO para SQuAD: placeholder '{e.args[0]}' ausente no dicionário de argumentos.")
        # Debug: Imprima o template para ver o que ele está esperando
        print(f"Template esperado: {template_str}") 
        return ""
    except Exception as e:
        print(f"[prompt_evaluator] Erro inesperado ao formatar prompt: {e}")
        return ""


def evaluate_prompt(prompt_instruction, dataset, executor_config, strategy_config, experiment_settings, output_dir):
    task = experiment_settings.get("task")
    if task == 'imdb':
        return evaluate_prompt_imdb(prompt_instruction, dataset, executor_config, strategy_config, experiment_settings, output_dir)
    elif task == 'squad':
        return evaluate_prompt_squad(prompt_instruction, dataset, executor_config, strategy_config, experiment_settings, output_dir)
    else:
        print(f"[prompt_evaluator] ERRO: Tarefa de avaliação '{task}' desconhecida.")
        return 0, 0, 0, "tarefa_desconhecida"

def evaluate_prompt_squad(prompt_instruction, dataset, executor_config, strategy_config, experiment_settings, output_dir):
    total_em = 0
    total_f1 = 0
    total_tokens = count_tokens(prompt_instruction)

    print(f"[prompt_evaluator] [SQuAD] Avaliando prompt_instruction: '{prompt_instruction}'")
    # Configure o número máximo de workers. Ajuste este valor com base nos limites de taxa da API
    MAX_WORKERS = 10
    
    executor_name_sanitized = executor_config["name"].replace(":", "_").replace("/", "_")
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
                    print(f"[prompt_evaluator] [SQuAD] Exemplo {index}: Q='{dp['question'][:100]}', GT='{dp['correct_answer'][:100]}', Pred='{predicted_answer[:100]}', EM={exact_match}, F1={f1_score_val}")

                ordered_results[index] = (predicted_answer, exact_match, f1_score_val, dp['context'], dp['question'], dp['correct_answer'])
            except Exception as exc:
                print(f"[prompt_evaluator] Erro durante a avaliação de um exemplo SQuAD (índice {index}): {exc}")
                ordered_results[index] = ("erro_processamento_paralelo", 0, 0, dp['context'], dp['question'], dp['correct_answer'])

    # Coleta as métricas e escreve os logs após todas as chamadas de API serem concluídas
    with open(log_path, "a", encoding="utf-8") as f:
        for predicted_answer, exact_match, f1_score_val, context, question, correct_answer in ordered_results:
            total_em += exact_match
            total_f1 += f1_score_val
            f.write(f'"{prompt_instruction}","{context.replace('"', "'''''")}","{question.replace('"', "'''''")}","{correct_answer.replace('"', "'''''")}","{predicted_answer.replace('"', "'''''")}",{exact_match},{f1_score_val}\n')

    avg_em, avg_f1 = calculate_squad_metrics(total_em, total_f1, len(dataset))
    
    return avg_em, avg_f1, total_tokens, ""

def evaluate_prompt_imdb(prompt_instruction, dataset, evaluator_config, strategy_config, experiment_settings, output_dir):
    predictions = []
    true_labels = dataset["label"].tolist()
    
    evaluator_name_sanitized = evaluator_config["name"].replace(":", "_").replace("/", "_")
    strategy_name_sanitized = strategy_config["name"]
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"eval_{evaluator_name_sanitized}_{strategy_name_sanitized}.csv")

    total_tokens = count_tokens(prompt_instruction) # Conta tokens para a instrução do prompt uma vez

    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("prompt_instruction,text,true_label,llm_prediction,llm_response\n")

    print(f"[prompt_evaluator] [IMDB] Avaliando prompt_instruction: '{prompt_instruction}'")

    MAX_WORKERS = 10 # Ajuste com base nos limites de taxa da API
    
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
                if index < 3:
                    print(f"[prompt_evaluator] [IMDB] Exemplo {index}: Text='{dp['text'][:100]}', GT={dp['label']}, Pred={prediction}, LLM_Resp='{response_text[:100]}'")
                ordered_results[index] = (prediction, response_text, dp['text'], dp['label'])
            except Exception as exc:
                print(f"[prompt_evaluator] Erro durante a avaliação de um exemplo IMDB (índice {index}): {exc}")
                ordered_results[index] = (1 - dp['label'], "erro_processamento_paralelo", dp['text'], dp['label'])

    with open(log_path, "a", encoding="utf-8") as f:
        for prediction, response_text, text, label in ordered_results:
            predictions.append(prediction)
            f.write(f'"{prompt_instruction}","{text.replace('"', "'''''")}","{label}","{prediction}","{response_text.replace('"', "'''''")}"\n')

    acc, f1 = calculate_imdb_metrics(true_labels, predictions)

    return acc, f1, total_tokens, ""