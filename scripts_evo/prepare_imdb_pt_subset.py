# prepare_imdb_pt_subset.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # Diretórios
    data_dir = "data"
    plot_dir = "logs/plots"
    
    ensure_dir(data_dir)
    ensure_dir(plot_dir)

    # Carregar o dataset
    print("Carregando o dataset 'maritaca-ai/imdb_pt'...")
    dataset = load_dataset("maritaca-ai/imdb_pt")
    df = pd.DataFrame(dataset['train'])

    # Exibir amostras iniciais
    print("\nAmostras iniciais do dataset:")
    print(df.head(15))

    # Contagem das classes
    class_counts = df['label'].value_counts()
    print("\nDistribuição das classes:")
    print(class_counts)

    # Gráfico da distribuição das classes
    plt.figure(figsize=(6,4))
    plt.bar(class_counts.index, class_counts.values, tick_label=['Negativo', 'Positivo'])
    plt.xlabel("Classe")
    plt.ylabel("Número de Amostras")
    plt.title("Distribuição das Classes no Dataset IMDB-PT")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/imdb_class_distribution.png")
    plt.close()

    # Subconjunto balanceado
    df_subset = df.groupby('label', group_keys=False).apply(lambda x: x.sample(50, random_state=42))
    print("\nAmostras selecionadas para o subconjunto de testes:")
    print(df_subset.head())

    class_counts_subset = df_subset['label'].value_counts()
    print("\nDistribuição das classes no subconjunto:")
    print(class_counts_subset)

    # Gráfico do subconjunto
    plt.figure(figsize=(6,4))
    plt.bar(class_counts_subset.index, class_counts_subset.values, tick_label=['Negativo', 'Positivo'])
    plt.xlabel("Classe")
    plt.ylabel("Número de Amostras")
    plt.title("Distribuição das Classes no Subconjunto IMDB-PT")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/imdb_subset_distribution.png")
    plt.close()

    # Salvar CSV
    output_path = f"{data_dir}/imdb_pt_subset.csv"
    df_subset.to_csv(output_path, index=False)
    print(f"\nSubconjunto salvo em: {output_path}")

if __name__ == "__main__":
    main()
