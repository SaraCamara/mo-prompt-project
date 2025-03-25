# emo-prompt-project

### Evolutionary-Multiobjective-Optimization Prompt Project

Este projeto investiga a otimização evolutiva de prompts aplicados a **tarefas de classificação de sentimento em português**, utilizando modelos de linguagem natural e diferentes estratégias de prompting. Os primeiros testes realizados são inspirados na estrutura do [EvoPrompt (ICLR 2024)](https://arxiv.org/pdf/2309.08532), o projeto realiza a evolução de prompts de forma automatizada, com geração orientada por LLMs e avaliação multi-modelo.

---

## Estrutura do Projeto


emo-prompt-project/
├── config/
│ ├── credentials.yaml                   # <--- Chaves de API (não versionado)
│ └── experiment_settings.yaml           # <--- Modelos, estratégias e paths
├── dataset/
│ ├── imdb_notebook.ipynb
│ └── imdb_pt_subset.csv                 # <--- Subconjunto do IMDB pt (Hugging Face)
├── notebooks/
│ └── logs/
│       └── results_model_strategy.csv    # <--- Resultados parciais por cenário
│ └── evo_prompt_pt_classification.ipynb
├── .gitignore
└── README.md


---

![Python 3.x](https://img.shields.io/badge/python-3.x-green.svg)

Notebook em Jupyter/IPython sobre otimização evolutiva multiobjetivo de prompts com LLMs.

> **Notas:**
> - Escrito em Python 3.x  
> - Veja [`requirements.txt`](./requirements.txt) para dependências

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

## Hiperparâmetros

| Nome              | Valor |
|-------------------|-------|
| `generations`     | 1     | -> 10 
| `population_size` | 10    |
| `dev_sample`      | 50    |
| `top_k`           | 3     |

> O processo de evolução é realizado por 1 ciclo no teste atual. Pode ser expandido conforme performance.

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

[WIP]