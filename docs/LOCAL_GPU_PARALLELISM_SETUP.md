# Configuração de Paralelismo em GPU Local (Ollama)

Este documento descreve as alterações realizadas para habilitar e otimizar o processamento paralelo de inferências usando o Ollama local, visando resolver problemas de timeout e aumentar o throughput dos experimentos evolutivos.

## 1. O Problema Original

O projeto utiliza algoritmos evolutivos que avaliam populações de prompts. Isso gera "rajadas" de requisições simultâneas (ex: 10 threads).
Originalmente, o Ollama processa requisições de forma **sequencial** por padrão.

*   **Sintoma**: Erros `Read timed out` frequentes.
*   **Causa**: Com 10 workers disparando requisições e o Ollama processando 1 por 1, a 10ª requisição aguardava ~30-50s na fila, estourando o timeout padrão.

## 2. Solução Implementada

Habilitamos o paralelismo nativo do Ollama e adaptamos os clientes Python para serem resilientes e sincronizados com a capacidade do servidor.

### A. Configuração do Servidor (`scripts/ollama-local.sh`)

Criamos um script dedicado para iniciar o Ollama com variáveis de ambiente específicas para concorrência:

*   **`OLLAMA_NUM_PARALLEL=4`**: Permite que o servidor processe até 4 requisições de inferência simultaneamente para o mesmo modelo carregado na VRAM.
*   **`OLLAMA_MAX_LOADED_MODELS=2`**: Mantém até 2 modelos na memória (útil se alternar entre avaliadores).
*   **`OLLAMA_KEEP_ALIVE=10m`**: Evita descarregar o modelo da VRAM entre gerações do algoritmo genético.

### B. Adaptação do Código Python

#### 1. Cliente HTTP (`scripts/llm_clients.py`)
*   **Retry com Backoff**: Implementado `max_retries=3` com espera exponencial (10s, 20s, 30s) para lidar com picos de carga.
*   **Timeout Estendido**: Timeout padrão aumentado de 50s para 120s (configurável via YAML).

#### 2. Avaliador (`scripts/prompt_evaluator.py`)
*   **Ajuste de Workers**: O número de threads (`MAX_WORKERS`) foi ajustado dinamicamente:
    *   **Ollama Local**: 4 workers (para casar com `OLLAMA_NUM_PARALLEL`). Evita filas desnecessárias no lado do cliente.****
    *   **APIs Nuvem**: 10 workers (mantido para alta vazão em serviços externos).

## 3. Análise de Performance

### Benchmark Inicial (RTX 5060 Ti)
Teste realizado com 4 requisições simultâneas ao `deepseek-r1:7b`:

| Modo | Tempo Total | Speedup Aproximado |
|------|-------------|--------------------|
| **Sequencial** (Original) | ~56 segundos | 1x |
| **Paralelo** (4 slots) | ~30 segundos | **~1.9x** |

### Insights e Gargalos

1.  **Compute Bound vs Memory Bound**:
    *   O paralelismo do Ollama (`OLLAMA_NUM_PARALLEL`) compartilha os pesos do modelo (VRAM) mas divide os núcleos de processamento (Compute).
    *   O speedup não é linear (4x) porque a inferência de LLMs é intensiva em computação. Com 4 streams, cada token gerado demora mais individualmente, mas o throughput total aumenta.

2.  **Modelos "Thinking" (DeepSeek R1)**:
    *   O modelo `deepseek-r1` gera longas cadeias de pensamento (`<thinking>`) antes da resposta final. Isso consome mais tempo de GPU por requisição, aumentando a latência e a probabilidade de congestionamento se o número de workers exceder os slots paralelos.

3.  **VRAM**:
    *   O modelo `deepseek-r1:7b` (q4_k_m) ocupa ~4.7GB.
    *   O contexto (`num_ctx`) multiplicado por 4 slots consome VRAM adicional (KV Cache).
    *   **Atual**: Com 4 slots, o uso de VRAM é seguro em uma placa de 16GB, mas cuidado ao aumentar o contexto.

### Possíveis Melhorias Futuras

1.  **Orquestração de Containers (Docker/K8s)**:
    *   Se houver múltiplas GPUs disponíveis, pode-se usar Docker Compose ou Kubernetes para subir uma instância do Ollama por GPU e usar um Load Balancer (Nginx) para distribuir as requisições.

2.  **Otimização de Batch**:
    *   Testar empiricamente valores de `OLLAMA_NUM_PARALLEL` entre 2 e 8. Às vezes, 2 ou 3 paralelos entregam latência melhor por token do que 4, dependendo da saturação dos CUDA cores.

3.  **Filas Assimétricas**:
    *   Implementar uma fila de prioridade no Python (`population_manager.py`) para enviar os prompts mais promissores primeiro ou processar em lotes menores para evitar timeouts em avaliações muito longas.
