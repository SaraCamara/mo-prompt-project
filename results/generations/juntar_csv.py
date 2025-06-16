import pandas as pd
import os

# Defina o caminho para a pasta que contém os seus arquivos CSV.
# Se a pasta "per_generation_pareto" não estiver no mesmo diretório que o script,
# você precisará fornecer o caminho completo.
caminho_pasta = 'logs/emo/sabiazinho/few-shot/per_generation_pareto'

# Lista para armazenar os DataFrames de cada arquivo
dataframes = []

# Itera sobre os arquivos no diretório especificado
for nome_arquivo in sorted(os.listdir(caminho_pasta)):
    # Verifica se o arquivo é um dos arquivos pareto que você quer processar
    if nome_arquivo.startswith('pareto_gen_') and nome_arquivo.endswith('.csv'):
        # Constrói o caminho completo para o arquivo
        caminho_arquivo = os.path.join(caminho_pasta, nome_arquivo)

        # Lê o arquivo CSV para um DataFrame do pandas
        df = pd.read_csv(caminho_arquivo)

        # Extrai o número da geração a partir do nome do arquivo.
        # Isso remove o prefixo 'pareto_gen_' e o sufixo '.csv'.
        numero_geracao = nome_arquivo.replace('pareto_gen_', '').replace('.csv', '')

        # Adiciona a nova coluna 'generation' no início do DataFrame
        df.insert(0, 'generation', numero_geracao)

        # Adiciona o DataFrame modificado à nossa lista
        dataframes.append(df)

# Concatena todos os DataFrames da lista em um único DataFrame
df_final = pd.concat(dataframes, ignore_index=True)

# Salva o DataFrame resultante em um novo arquivo CSV
df_final.to_csv('pareto_gens.csv', index=False)

print("Arquivo 'pareto_gens.csv' criado com sucesso!")