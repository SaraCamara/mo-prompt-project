import sys
import os
import yaml
import random
import pandas as pd

from utils import (
    load_credentials,
    load_settings,
    build_prompt,
    evaluate_prompt,
    generate_prompt_with_gpt4o
)


def run_scenario(model, strategy, initial_prompts, df, top_k, num_epochs, log_path):
    results = []

    print(f"=== Iniciando cenário: {model} + {strategy} ===")
    for prompt in initial_prompts:
        acc, f1, tokens = evaluate_prompt(prompt, model, strategy, df)
        results.append((prompt, acc, f1, tokens))

    os.makedirs(log_path, exist_ok=True)
    log_file = f"{log_path}/results_{model}_{strategy}.csv"

    for epoch in range(num_epochs):
        print(f"\n→ Geração {epoch+1}/{num_epochs}")
        # Ordena por F1 decrescente e tokens crescente
        results.sort(key=lambda x: (-x[2], x[3], random.random()))
        top_prompts = results[:top_k]

        try:
            new_prompt = generate_prompt_with_gpt4o([(p, f1) for p, _, f1, _ in top_prompts])
            print(f"Novo prompt gerado: {new_prompt}")
        except Exception as e:
            print(f"Erro ao gerar prompt com GPT-4o Mini: {e}")
            break

        acc, f1, tokens = evaluate_prompt(new_prompt, model, strategy, df)
        print(f"Desempenho → Accuracy: {acc:.3f}, F1: {f1:.3f}")
        results.append((new_prompt, acc, f1, tokens))
        pd.DataFrame(results, columns=["prompt", "accuracy", "f1", "num_tokens"]).to_csv(log_file, index=False)

    return results

def main():
    if len(sys.argv) < 3:
        print("Uso: python scripts_emo/inferencia.py <modelo> <caminho_dataset_csv>")
        sys.exit(1)

    modelo_especifico = sys.argv[1].lower()
    caminho_csv = sys.argv[2]

    if not os.path.exists(caminho_csv):
        print(f"[✗] Arquivo CSV {caminho_csv} não encontrado.")
        sys.exit(1)

    # Carrega configurações
    creds = load_credentials("config/credentials.yaml")
    settings = load_settings("config/experiment_settings.yaml")

    # Lê dataset
    df = pd.read_csv(caminho_csv).dropna(subset=["text", "label"])
    df = df[df["label"].isin([0, 1])]

    # Lê prompts iniciais
    with open("data/initial_prompts.txt", "r", encoding="utf-8") as f:
        initial_prompts = [line.strip() for line in f if line.strip()]

    final_results = []
    modelos = [modelo_especifico] if modelo_especifico != "all" else settings["models"]

    for model in modelos:
        for strategy in settings["strategies"]:
            scenario_results = run_scenario(
                model,
                strategy,
                initial_prompts,
                df,
                settings["top_k"],
                settings["num_epochs"],
                log_path="logs"
            )
            for prompt, acc, f1, tokens in scenario_results:
                final_results.append({
                    "model": model,
                    "strategy": strategy,
                    "prompt": prompt,
                    "accuracy": acc,
                    "f1": f1, 
                    "num_tokens": tokens
                })

    pd.DataFrame(final_results).to_csv(settings["results_log_path"], index=False)
    print("\n✓ Resultados finais salvos em:", settings["results_log_path"])

if __name__ == "__main__":
    main()
