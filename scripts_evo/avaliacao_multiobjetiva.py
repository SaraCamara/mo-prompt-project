import argparse
import os
import pandas as pd
import json
from nltk.tokenize import TreebankWordTokenizer
import matplotlib.pyplot as plt

# Tokenizador 
tokenizer = TreebankWordTokenizer()

def count_tokens(prompt: str) -> int:
    return len(tokenizer.tokenize(prompt))

def is_dominated(sol, others):
    for _, other in others.iterrows():
        if (other["accuracy"] >= sol["accuracy"] and other["num_tokens"] <= sol["num_tokens"]) and \
           (other["accuracy"] > sol["accuracy"] or other["num_tokens"] < sol["num_tokens"]):
            return True
    return False

def find_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["num_tokens"] = df["prompt"].apply(count_tokens)
    pareto_df = df[~df.apply(lambda row: is_dominated(row, df), axis=1)]
    return pareto_df.sort_values(by=["accuracy", "num_tokens"], ascending=[False, True])

def main(input_dir, output_dir, show_plot):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Pasta '{input_dir}' não encontrada.")

    csv_files = [f for f in os.listdir(input_dir) if f.startswith("final_results_") and f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("Nenhum arquivo 'final_results_*.csv' encontrado.")

    os.makedirs(output_dir, exist_ok=True)

    # Carrega todos os arquivos encontrados
    dfs = [pd.read_csv(os.path.join(input_dir, f)) for f in csv_files]
    df_all = pd.concat(dfs, ignore_index=True)

    # Gera fronteira de Pareto
    pareto_df = find_pareto_front(df_all)

    # Salva CSV e JSON
    pareto_csv = os.path.join(output_dir, "pareto_optimal_solutions.csv")
    pareto_json = os.path.join(output_dir, "pareto_optimal_prompts.json")
    pareto_df.to_csv(pareto_csv, index=False)
    with open(pareto_json, "w", encoding="utf-8") as f:
        json.dump(
            pareto_df[["model", "strategy", "accuracy", "num_tokens", "prompt"]].to_dict(orient="records"),
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"[✓] Soluções Pareto-ótimas salvas em:\n- CSV:  {pareto_csv}\n- JSON: {pareto_json}")

    # Visualização
    if show_plot:
        df_all["num_tokens"] = df_all["prompt"].apply(count_tokens) 
        plt.figure(figsize=(8, 6))
        plt.scatter(df_all["num_tokens"], df_all["accuracy"], alpha=0.4, label="Todas as soluções")
        plt.scatter(pareto_df["num_tokens"], pareto_df["accuracy"], color="red", label="Pareto-ótimas")
        plt.xlabel("Número de tokens no prompt")
        plt.ylabel("Acurácia")
        plt.title("Fronteira de Pareto: Acurácia vs Tokens")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação multiobjetiva de prompts com fronteira de Pareto.")
    parser.add_argument("--input_dir", default="logs", help="Diretório com arquivos final_results_*.csv")
    parser.add_argument("--output_dir", default="logs", help="Diretório de saída para CSV e JSON")
    parser.add_argument("--plot", action="store_true", help="Exibir gráfico da fronteira de Pareto")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.plot)
