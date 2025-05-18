# emo-prompt-project
---
[WIP]

### Evolutionary-Multiobjective-Optimization Prompt Project

Este projeto investiga a otimização evolutiva de prompts aplicados a **tarefas de classificação de sentimento em português**, utilizando **modelos de linguagem natural (LLMs)** e diferentes estratégias de prompting. Os primeiros testes realizados são inspirados na estrutura do [EvoPrompt (ICLR 2024)](https://arxiv.org/pdf/2309.08532), o projeto realiza a evolução de prompts de forma automatizada, com geração orientada por LLMs e avaliação multi-modelo.

## Estrutura de Pastas
- `config/`: configurações de experimentos e chaves de API
- `data/`: dataset e prompts iniciais
- `logs/`: resultados intermediários e finais
- `scripts/`: scripts principais
- `requirements.txt`: dependências

### Resumo Visual da Estrutura

```
emo-prompt-project/
├── config/
│   ├── credentials.yaml
│   └── experiment_settings.yaml
├── data/
│   ├── imdb_pt_subset.csv
│   └── initial_prompts.txt
├── logs/
│   ├── results_{model}_{strategy}.csv
│   └── final_results_{model}.csv
├── scripts/
│   ├── main.py
│   ├── mono_evolution.py
│   ├── multi_evolution.py  
│   └── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Como Executar

### 1. Instalação de Dependências

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execução do Algoritmo Monoobjetivo (Acurácia)

```bash
python scripts/main_evo.py
```

### 3. Execução do Algoritmo Multiobjetivo (Acurácia + Tokens)

```bash
python scripts/main_emo.py
```

---

## Hiperparâmetros

| Nome              | Valor |
|-------------------|-------|
| `generations`     | 10    |
| `population_size` | 10    |
| `dev_sample`      | 50    |
| `top_k`           | 3     |

> O processo de evolução é realizado por 1 ciclo no teste atual. Pode ser expandido conforme performance.

---

## Visualização dos Resultados

- Visualizar fronteiras de Pareto.
- Comparar estratégias de prompting.
- Analisar acurácia × custo computacional (tokens).

---


## Testes Iniciais

### Tarefa:

**Classificação binária de sentimento** sobre resenhas de filmes (positivo ou negativo).

### Estratégias de Prompting:
- `zero-shot`
- `few-shot`
- `chain-of-thought (CoT)`

### Modelos de Avaliação:
- [`maritalk`](https://www.maritaca.ai/) (modelo Sabiazinho-3 via API)
- `deepseek` (mock/local)
- `llama` (mock/local)

### Modelo de Evolução:
- [`GPT-4o Mini`](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence) — utilizado como operador de mutação e geração de novos prompts

---

## Experimentos: EvoPrompt × Maritalk

### Teste 1a: Estratégia `zero-shot`
- Modelo: **Sabiazinho-3 (Maritalk)**
- Estratégia: Instrução direta para classificação binária
- Prompt base: 10 variações em português

**Results**:  
→ `logs/results_maritalk_zero-shot.csv`

---

### Teste 1b: Estratégia `few-shot`
- Modelo: **Sabiazinho-3 (Maritalk)**
- Estratégia: 2 exemplos + instrução

**Results**:  
→ `logs/results_maritalk_few-shot.csv`

---

### Teste 1c: Estratégia `CoT` (Chain-of-Thought)
- Modelo: **Sabiazinho-3 (Maritalk)**
- Estratégia: Instrução + raciocínio passo a passo

**Results**:  
→ `logs/results_maritalk_cot.csv`

---

> Os próximos experimentos com `llama` e `deepseek` seguirão a mesma estrutura e protocolo.

---

##  Referências

- Guo et al., 2024. *Connecting Large Language Models with Evolutionary Algorithms* (ICLR)
- Baumann & Kramer, 2024. *EMO-Prompts: Evolutionary Multi-Objective Prompt Optimization*