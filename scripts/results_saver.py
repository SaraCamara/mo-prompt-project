import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import logging

matplotlib.use('Agg') # Configura o backend do Matplotlib para 'Agg' para evitar problemas de thread com GUI

# Seção: Persistência e Salvamento de Resultados
logger = logging.getLogger(__name__)

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
    logger.info(f"Resultados detalhados da geração {generation} salvos em {path}")

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
    logger.info(f"População ordenada da geração {generation} salva em {sorted_log_path}")


def save_final_results(population, config, output_csv_path): 
    logger.info("Salvando resultados finais.")
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
    logger.info(f"Gráfico da fronteira de Pareto salvo em {plot_path}")