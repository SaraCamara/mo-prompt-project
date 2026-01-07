#!/bin/bash

# === [0] Configuração de Ambiente Local ===
# Use uma forma mais robusta para obter o diretório do script
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Define caminhos para o executável Ollama e o diretório de runtime
OLLAMA_BIN="$CURRENT_DIR/bin/ollama_exec"
RUNTIME_HOME="$CURRENT_DIR/bin/ollama_runtime_home"

# Define onde o Ollama vai salvar os modelos e seu host
# Garante que estas variáveis sejam exportadas ANTES de qualquer comando Ollama
export OLLAMA_MODELS="$RUNTIME_HOME/models"
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_LLM_GPU=1 # Tenta carregar todas as camadas possíveis na GPU

echo "DEBUG: CURRENT_DIR = $CURRENT_DIR"
echo "DEBUG: OLLAMA_BIN = $OLLAMA_BIN"
echo "DEBUG: RUNTIME_HOME = $RUNTIME_HOME"
echo "DEBUG: OLLAMA_MODELS = $OLLAMA_MODELS"
echo "DEBUG: OLLAMA_HOST = $OLLAMA_HOST"

echo "=== [1] Verificando ambiente ==="

# Verifica se o executável local existe
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ Erro: O arquivo '$OLLAMA_BIN' não foi encontrado."
    exit 1
fi

# Cria a pasta de modelos se não existir
mkdir -p "$OLLAMA_MODELS" # Agora este mkdir usará o caminho correto

echo "=== [2] Gerenciando Ollama Local ==="

# 1. PARAR PROCESSOS ANTIGOS
echo "➡ Parando instâncias antigas..."
pkill -f "ollama_exec serve" 2>/dev/null
# Espera 2 segundos para o sistema operacional liberar a porta TCP (evita address in use)
sleep 2

# 2. INICIAR O SERVIDOR
echo "➡ Iniciando ./ollama_exec..."
echo "   Logs em: $CURRENT_DIR/ollama_manual.log"
echo "   Modelos em: $OLLAMA_MODELS"

# Executa em background
nohup env OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" serve > "$CURRENT_DIR/ollama_manual.log" 2>&1 &
OLLAMA_PID=$!

# 3. AGUARDAR O SERVIDOR ESTAR PRONTO (NOVO BLOCO DE VERIFICAÇÃO)
echo "➡ Aguardando o servidor Ollama ficar pronto..."
MAX_RETRIES=15
COUNT=0

# Loop: tenta conectar a cada 1 segundo. Se conseguir, sai do loop.
until curl -s http://localhost:11434/ > /dev/null; do
    sleep 1
    COUNT=$((COUNT+1))
    echo -n "."
    
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo ""
        echo "❌ Erro: Ollama não iniciou após 15 segundos."
        echo "   Verifique o log: cat $CURRENT_DIR/ollama_manual.log"
        tail -n 5 "$CURRENT_DIR/ollama_manual.log" # Mostra as últimas 5 linhas do log
        kill $OLLAMA_PID 2>/dev/null # Garante que o processo Ollama seja encerrado
        exit 1
    fi
done
echo ""
echo "✅ Ollama conectado e respondendo!"

# 4. BAIXAR MODELOS (AGORA QUE O SERVIDOR ESTÁ RODANDO)
models=("qwen2.5:7b" "deepseek-r1:7b" "llama3.1") # Reordenado para melhor fluxo

for model in "${models[@]}"; do
    # Tratamento especial para deepseek-r1:7b devido a problemas anteriores, tentar re-baixá-lo
    if [ "$model" == "deepseek-r1:7b" ]; then
        echo "➡ Verificando e re-baixando modelo $model para garantir integridade/compatibilidade..."
        "$OLLAMA_BIN" rm "$model" 2>/dev/null # Remove silenciosamente se existir
        "$OLLAMA_BIN" pull "$model"
    elif [ "$model" == "llama3.1" ]; then
        echo "➡ Baixando modelo $model..."
        "$OLLAMA_BIN" pull "$model" || {
            echo "❌ Erro ao baixar $model. Pode ser necessário atualizar o Ollama."
            echo "   Por favor, baixe a versão mais recente do Ollama em: https://ollama.com/download"
        }
    elif ! "$OLLAMA_BIN" list | grep -q "$model"; then # Para outros modelos, baixar apenas se não estiverem presentes
        echo "➡ Baixando modelo $model..."
        "$OLLAMA_BIN" pull "$model"
    else
        echo "➡ Modelo $model já existe."
    fi
done

echo "=== [3] Iniciando Open WebUI (Docker) ==="

# Nota: Para rodar docker sem sudo, seu usuário deve estar no grupo 'docker'.
docker rm -f open-webui 2>/dev/null

docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v open-webui:/app/backend/data \
  --name open-webui --restart always \
  ghcr.io/open-webui/open-webui:main

echo ""
echo "✅ Configuração concluída!"
echo "   WebUI: http://localhost:3000"
echo "   API Ollama: http://localhost:11434"
