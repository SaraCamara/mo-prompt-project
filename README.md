#  MO-Prompt-Project

**Otimização Evolutiva de Prompts com Abordagem Multiobjetivo**

Este projeto investiga a otimização evolutiva de prompts aplicados a tarefas de classificação de sentimento e extração de respostas em português, utilizando modelos de linguagem natural (LLMs) e diferentes estratégias de prompting.

##  Objetivo

A função de otimização multiobjetiva busca:
-  **Maximizar** acurácia (F1-score)
-  **Minimizar** quantidade de tokens utilizados

Suporta dois modos de otimização:
- **Multiobjetivo (MOP)**: Usa NSGA-II para otimizar F1 vs Tokens simultaneamente
- **Monoobjetivo (EVO)**: Otimiza apenas F1

##  Estrutura do Projeto

```
mo-prompt-project/
├── README.md                          # Este arquivo
├── CHANGELOG.md                       # Histórico de mudanças
├── requirements.txt                   # Dependências Python
│
├── config/                            # Configurações
│   ├── credentials.yaml               # Chaves de API
│   └── experiment_settings.yaml       # Parâmetros dos experimentos
│
├── data/                              # Dados e prompts
│   ├── imdb_pt_subset.csv             # Dataset IMDB português
│   ├── squad_*.json                   # Dataset SQuAD português
│   └── initial_prompts_*.txt          # Prompts iniciais
│
├── scripts/                           # Código-fonte (estrutura organizada)
│   ├── main.py                        # ← Entrada principal
│   ├── core/                          # Algoritmos evolutivos
│   ├── llm/                           # Clientes LLM e avaliação
│   ├── ollama/                        # Setup e monitoramento Ollama
│   ├── utils/                         # Utilidades
│   └── demo/                          # Demos
│
├── logs/                              # Resultados e outputs
│   └── {task}/{mode}/{model}/{strategy}/
│
├── docs/                              # Documentação
│   ├── README.md                      # Índice de docs
│   ├── SETUP/                         # Guias de setup
│   ├── FEATURES/                      # Funcionalidades
│   ├── GUIDES/                        # Tutoriais
│   └── ARCHITECTURE/                  # Detalhes técnicos
│
└── results/                           # Notebooks de análise
    ├── analise_resultados.ipynb
    ├── artigo_analise_resultados.ipynb
    └── comparativo_multi_evo.ipynb
```

##  Organização de Scripts

```
scripts/
├── main.py                    # ← CLI interativo (entrada)
│
├── core/                      # Algoritmos evolutivos
│   ├── multi_evolution.py
│   ├── mono_evolution.py
│   ├── evolutionary_operators.py
│   ├── nsga2_algorithms.py
│   ├── population_manager.py
│   └── selection_algorithms.py
│
├── llm/                       # LLM e avaliação
│   ├── llm_clients.py
│   ├── prompt_evaluator.py
│   └── evaluation_metrics.py
│
├── ollama/                    # Ollama local
│   ├── setup.py
│   ├── watch.py
│   ├── calculate_optimal_workers.py
│   ├── configure_parallel_ollama.sh
│   └── ollama-local.sh
│
├── utils/                     # Utilidades
│   ├── config_data_loader.py
│   ├── results_saver.py
│   ├── execution_tracker.py
│   ├── cli_interface.py
│   ├── logger_config.py
│   └── helpers.py
│
└── demo/
    └── demo_progress_bar.py
```

##  Quick Start

### 1. Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuração (primeira vez)

```bash
python -m scripts.ollama.setup --auto
```

### 3. Monitorar (em outro terminal)

```bash
python -m scripts.ollama.watch
```

### 4. Executar

```bash
python scripts/main.py
```

## ⚙️ Configuração Requerida

### `config/credentials.yaml`

```yaml
openai_api_key: "sk-..."
openai_api_base: "https://api.openai.com/v1"

maritalk_api_key: "..."
maritalk_endpoint: "https://chat.maritaca.ai/api/chat/inference"
```

### `config/experiment_settings.yaml`

Contém modelos, parâmetros evolutivos, estratégias e datasets.

##  Parâmetros Padrão

| Parâmetro | Valor |
|-----------|-------|
| population_size | 10 |
| max_generations | 10 |
| mutation_rate | 0.8 |
| k_tournament | 2 |
| stagnation_limit | 3 |

##  Documentação

- **[docs/README.md](docs/README.md)** - Índice completo
- **[docs/SETUP/](docs/SETUP/)** - Guias de configuração
- **[docs/FEATURES/](docs/FEATURES/)** - Resume, Progress Display
- **[docs/GUIDES/](docs/GUIDES/)** - Tutoriais práticos
- **[CHANGELOG.md](CHANGELOG.md)** - Histórico de mudanças

##  Retomar Execução

```bash
python scripts/main.py
```

O sistema detecta execuções anteriores e oferece retomar da última geração.

##  Resultados

Salvos em: `logs/{task}/{mode}/{model}/{strategy}/`

```
├── final_results.csv
├── final_pareto_front.png
├── evolution_summary.md
├── execution_metadata.json
├── timing_log.csv
└── per_generation_pareto/
```

##  Troubleshooting

- **Timeouts Ollama**: [docs/SETUP/ollama-optimization.md](docs/SETUP/ollama-optimization.md)
- **Workers**: [docs/SETUP/max-workers.md](docs/SETUP/max-workers.md)
- **GPU**: [docs/SETUP/gpu-parallelism.md](docs/SETUP/gpu-parallelism.md)

---

**Última atualização**: Janeiro 2026
