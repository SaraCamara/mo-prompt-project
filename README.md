# mo-prompt-project
---

### Projeto de otimização de prompts com abordagem multiobjetivo e evolutiva

Este projeto investiga a otimização evolutiva de prompts aplicados a tarefas de classificação de sentimento em português, utilizando modelos de linguagem natural (LLMs) e diferentes estratégias de prompting. A função de otimização multiobjetiva aplicada busca maximização da acurácia e minimização da quantidade de tokens utilizados.

## Estrutura de Pastas

- `config/`: configurações de experimentos e chaves de API
- `data/`: dataset e prompts iniciais
- `logs/`: resultados intermediários
- `results/`: notebooks avaliações dos resultados
- `scripts/`: scripts principais
- `.gitignore`: arquivos ignorados
- `README.md`: descrição do projeto
- `requirements.txt`: dependências

### Resumo Visual da Estrutura

```
mo-prompt-project/
├── config/
│   ├── credentials.yaml
│   └── experiment_settings.yaml
├── data/
│   ├── imdb_pt_subset.csv
│   └── initial_prompts.txt
├── logs/
│   ├── mop/
│   │   ├── {model}/
│   │   │   └── {strategy}/
│   │   ├── generations/
│   ├── evo/
│   │   ├── {model}/
│   │   │   └── {strategy}/
├── results/
│   └── notebooks.ipynb
├── scripts/
│   ├── main.py
│   ├── mono_evolution.py
│   ├── multi_evolution.py  
│   ├── utils.py
│   ├── config_data_loader.py
│   ├── llm_clients.py
│   ├── evolutionary_operators.py
│   ├── evaluation_metrics.py
│   ├── prompt_evaluator.py
│   ├── selection_algorithms.py
│   ├── nsga2_algorithms.py
│   ├── population_manager.py
│   └── results_saver.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Hiperparâmetros

| Nome              | Valor |
|-------------------|--------------|
| `top_k`           | 10           |
| `max_generations` | 10           |
| `population_size` | 10           |
| `stagnation_limit`| 3            |
| `k_tournament_parents` | 2            |


---

## Reprodutibilidade

### Replicabilidade:
Instruções para o estudo ser reproduzido utilizando os mesmos métodos e obtendo resultados semelhantes.

* Arquivo `credential.yaml` não presente no projeto publicado segue o seguinte template em `config/`:

```python
# Configuração da API do GPT-4o Mini
openai_api_key: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
openai_api_base: "https://api.openai.com/v1"

# Configuração da API do Sabiá
sabia_api_key: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
sabia_url: "https://chat.maritaca.ai/api/chat/inference"
```

#### 1. Instalação de dependências e execução

```bash
python -m venv .venv
source .venv/bin/activate
```

```bash
python scripts/main.py
```

#### 2. Seleção de algoritmo, modelo e estratégia

```bash
[>] Selecione estratégia de otimização: 
0) Monoobjetivo
1) Multiobjetivo

[>] Selecione modelo avaliador:
0) gemma:2b
1) deepseek
2) sabiazinho

[>] Selecione estratégia de prompt:
0) Zero-shot
1) Few-Shot
2) CoT
```
---

**Initial Prompts :**

- `GPT-4o Mini` 

```
Instrução:

Você precisa analisar algumas críticas em um banco de dados sobre filmes.
Gere 10 prompts para de classificação de sentimentos dessas críticas, conforme os exemplos:

”Determine se a avaliação do filme é positiva ou negativa.”,
”Classifique a seguinte avaliação como boa ou ruim.”,
”Analise o sentimento desta frase e decida se é positivo ou negativo.”
```
---

Instruções para consistência dos resultados em diferentes contextos ou situações. Para aplicação do estudo em outros contextos, tarefas, datasets, modelos de LLM, etc devem se atentar as alterações necessárias em:

* Dataset: repositório `data/` e dataset_path em `experimentals_setting.yaml`.
* Prompts iniciais: repositório `data/` .
* Template de prompts: repositório `config/` em `experimentals_setting.yaml`.
* Modelos LLM usados: repositório `config/` em `experimentals_setting.yaml`, `credentials.yaml` e, repositório `scripts/llm_clients.py` e `scripts/evolutionary_operators.py`.

---

## Experimentos
### Tarefa:
Classificação binária de sentimento sobre resenhas de filmes (positivo ou negativo) e Extração de Respostas (Question Answering).

#### Testes realizados:

```mermaid
graph LR;

subgraph Estratégias de Otimização
    Mono-objetiva
    Multi-objetiva
    end

subgraph Modelos de avaliação  
    gemma:2b
    sabiazinho-3
    end

subgraph Estratégias de prompting
    zero-shot
    few-shot
    end

Mono-objetiva & Multi-objetiva ------>  gemma:2b & sabiazinho3 ------> zero-shot & few-shot; 
```

---

**Modelo de Evolução:**

- `GPT-4o Mini` (via API) — utilizado como operador de mutação e geração de novos prompts

---

### 1. Mono objetivo - EvoPrompt

a. Modelo `gemma:2b`

    `logs/evo/gemma_2b/zero-shot/final_results.csv`
    `logs/evo/gemma_2b/few-shot/final_results.csv`

b. Modelo `sabiazinho-3`

    `logs/evo/sabiazinho_3/zero-shot/final_results.csv`
    `logs/evo/sabiazinho_3/few-shot/final_results.csv`

### 2. Multi objetivo - MOPrompt

a. Modelo `gemma:2b`

    `logs/mop/gemma_2b/zero-shot/final_results.csv`
    `logs/mop/gemma_2b/few-shot/final_results.csv`

b. Modelo `sabiazinho-3`

    `logs/mop/sabiazinho_3/zero-shot/final_results.csv`
    `logs/mop/sabiazinho_3/few-shot/final_results.csv`
    `logs/mop/sabiazinho_3/few-shot_top5/final_results.csv`

---

## Análise dos resultados

    `results/artigo_analise_resultados.ipynb`

---