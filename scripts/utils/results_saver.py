import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import logging

matplotlib.use('Agg') 

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
    top_k = config.get("evolution_params", {}).get("top_k", len(df))
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
    plt.xlabel("Número de Tokens")
    plt.ylabel("F1 Score")
    plt.title("Fronteira de Pareto (Tokens vs F1 Score)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Gráfico da fronteira de Pareto salvo em {plot_path}")


def save_evolution_summary_markdown(config, metrics, output_dir):
    """Gera um resumo em markdown da execução evolutiva.
    
    Args:
        config: Dicionário de configuração do experimento
        metrics: Dicionário com métricas da execução
        output_dir: Diretório base de saída
    """
    import datetime
    
    summary_path = os.path.join(output_dir, "evolution_summary.md")
    
    # Extrai informações
    task = config.get("task", "unknown")
    objective = config.get("objective", "unknown")
    evaluator = config.get("evaluators", [{}])[0]
    model_name = evaluator.get("name", "unknown")
    model_id = evaluator.get("model", "unknown")
    strategy = config.get("strategies", [{}])[0]
    strategy_name = strategy.get("name", "unknown")
    
    evolution_params = config.get("evolution_params", {})
    
    # Métricas de tempo
    start_time = metrics.get("start_time", "N/A")
    end_time = metrics.get("end_time", "N/A")
    total_duration = metrics.get("total_duration_seconds", 0)
    
    # Formato de tempo legível
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    # Métricas finais
    generations_completed = metrics.get("generations_completed", 0)
    final_best_f1 = metrics.get("final_best_f1", 0.0)
    final_best_acc = metrics.get("final_best_acc", 0.0)
    final_best_tokens = metrics.get("final_best_tokens", 0)
    final_population_size = metrics.get("final_population_size", 0)
    pareto_front_size = metrics.get("pareto_front_size", None)
    
    # Métricas por geração (se disponível)
    generation_times = metrics.get("generation_times", [])
    
    # Constrói o markdown
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Resumo da Evolução de Prompts\n\n")
        
        # Informações gerais
        f.write("## Configuração do Experimento\n\n")
        f.write(f"- **Tarefa**: {task}\n")
        f.write(f"- **Modo de Otimização**: {objective}\n")
        f.write(f"- **Modelo Avaliador**: {model_name} (`{model_id}`)\n")
        f.write(f"- **Estratégia**: {strategy_name}\n")
        f.write(f"- **Data de Execução**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Parâmetros evolutivos
        f.write("## Parâmetros Evolutivos\n\n")
        f.write("```yaml\n")
        f.write(f"population_size: {evolution_params.get('population_size', 'N/A')}\n")
        f.write(f"max_generations: {evolution_params.get('max_generations', 'N/A')}\n")
        f.write(f"mutation_rate: {evolution_params.get('mutation_rate', 'N/A')}\n")
        f.write(f"k_tournament_parents: {evolution_params.get('k_tournament_parents', 'N/A')}\n")
        f.write(f"stagnation_limit: {evolution_params.get('stagnation_limit', 'N/A')}\n")
        f.write(f"top_k: {evolution_params.get('top_k', 'N/A')}\n")
        f.write("```\n\n")
        
        # Métricas de tempo
        f.write("## Métricas de Tempo\n\n")
        f.write(f"- **Início**: {start_time}\n")
        f.write(f"- **Fim**: {end_time}\n")
        f.write(f"- **Duração Total**: {duration_str} ({total_duration:.2f}s)\n")
        f.write(f"- **Gerações Completadas**: {generations_completed}\n")
        
        if generation_times:
            avg_time = sum(generation_times) / len(generation_times)
            min_time = min(generation_times)
            max_time = max(generation_times)
            f.write(f"- **Tempo Médio por Geração**: {avg_time:.2f}s\n")
            f.write(f"- **Tempo Mínimo por Geração**: {min_time:.2f}s\n")
            f.write(f"- **Tempo Máximo por Geração**: {max_time:.2f}s\n")
        f.write("\n")
        
        # Resultados finais
        f.write("## Resultados Finais\n\n")
        f.write(f"- **Melhor F1 Score**: {final_best_f1:.4f}\n")
        f.write(f"- **Melhor Accuracy**: {final_best_acc:.4f}\n")
        f.write(f"- **Tokens do Melhor Prompt**: {final_best_tokens}\n")
        f.write(f"- **Tamanho da População Final**: {final_population_size}\n")
        
        if pareto_front_size is not None:
            f.write(f"- **Tamanho da Fronteira de Pareto**: {pareto_front_size}\n")
        f.write("\n")
        
        # Tempos por geração (tabela)
        if generation_times:
            f.write("## Tempo por Geração\n\n")
            f.write("| Geração | Tempo (s) |\n")
            f.write("|---------|----------|\n")
            for i, t in enumerate(generation_times):
                f.write(f"| {i} | {t:.2f} |\n")
            f.write("\n")
        
        # Arquivos de saída
        f.write("## Arquivos de Saída\n\n")
        f.write(f"- **Diretório Base**: `{output_dir}`\n")
        f.write(f"- **Resultados Finais**: `{os.path.join(output_dir, 'final_results.csv')}`\n")
        
        if objective == "multiobjetivo":
            f.write(f"- **Gráfico Pareto**: `{os.path.join(output_dir, 'final_pareto_front.png')}`\n")
            f.write(f"- **Pareto por Geração**: `{os.path.join(output_dir, 'per_generation_pareto/')}`\n")
        else:
            f.write(f"- **Detalhes por Geração**: `{os.path.join(output_dir, 'generations_detail/')}`\n")
        
    logger.info(f"Resumo da evolução salvo em {summary_path}")
