#!/bin/bash

# === Ollama Local Server com Paralelismo ===
# Este script configura e inicia o Ollama nativo com suporte a requisições paralelas

set -e

# === [0] Configuração de Ambiente ===
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_FILE="$CURRENT_DIR/ollama_server.log"

# Configurações do servidor
export OLLAMA_HOST="0.0.0.0:11434"

# === CONFIGURAÇÕES DE PARALELISMO ===
# Permite processar múltiplas requisições simultaneamente (sem duplicar modelo na VRAM)
export OLLAMA_NUM_PARALLEL=4          # Número de requisições processadas em paralelo
export OLLAMA_MAX_LOADED_MODELS=2     # Máximo de modelos carregados na VRAM simultaneamente
export OLLAMA_KEEP_ALIVE="10m"        # Mantém modelo na VRAM por 10 min após última requisição

# === CONFIGURAÇÕES DE GPU ===
export OLLAMA_GPU_OVERHEAD="512MiB"   # Reserva de VRAM para o sistema
# export OLLAMA_NUM_GPU=999           # Descomente para forçar todas as camadas na GPU

echo "=========================================="
echo "  Ollama Server - Configuração Paralela"
echo "=========================================="
echo ""
echo "Configurações:"
echo "  • Host: $OLLAMA_HOST"
echo "  • Requisições paralelas: $OLLAMA_NUM_PARALLEL"
echo "  • Modelos simultâneos: $OLLAMA_MAX_LOADED_MODELS"
echo "  • Keep-alive: $OLLAMA_KEEP_ALIVE"
echo "  • Log: $LOG_FILE"
echo ""

# === [1] Verificar instalação ===
if ! command -v ollama &> /dev/null; then
    echo " Erro: Ollama não está instalado."
    echo "   Instale com: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

OLLAMA_VERSION=$(ollama --version 2>/dev/null | grep -oP '[\d.]+' | head -1)
echo " Ollama encontrado: versão $OLLAMA_VERSION"

# === [2] Parar instâncias anteriores ===
echo ""
echo "➡ Parando instâncias anteriores..."

# Para o serviço systemd se estiver rodando
sudo systemctl stop ollama 2>/dev/null || true

# Para processos manuais
pkill -f "ollama serve" 2>/dev/null || true

# Aguarda liberação da porta
sleep 2

# Verifica se a porta está livre
if lsof -i :11434 &>/dev/null; then
    echo "  Porta 11434 ainda em uso. Tentando liberar..."
    sudo fuser -k 11434/tcp 2>/dev/null || true
    sleep 1
fi

# === [3] Iniciar servidor ===
echo "➡ Iniciando Ollama com paralelismo..."

# Inicia em background com as variáveis de ambiente
nohup ollama serve > "$LOG_FILE" 2>&1 &
OLLAMA_PID=$!

echo "   PID: $OLLAMA_PID"

# === [4] Aguardar servidor ficar pronto ===
echo -n "➡ Aguardando servidor"
MAX_RETRIES=20
COUNT=0

until curl -s http://localhost:11434/ > /dev/null 2>&1; do
    sleep 1
    COUNT=$((COUNT+1))
    echo -n "."
    
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo ""
        echo " Erro: Ollama não iniciou após $MAX_RETRIES segundos."
        echo "   Últimas linhas do log:"
        tail -n 10 "$LOG_FILE"
        exit 1
    fi
done

echo ""
echo " Servidor Ollama pronto!"

# === [5] Verificar/Baixar modelos ===
echo ""
echo "➡ Verificando modelos..."

MODELS=("deepseek-r1:7b" "qwen2.5:7b")

for model in "${MODELS[@]}"; do
    if ollama list 2>/dev/null | grep -q "$model"; then
        echo "   ✓ $model (já instalado)"
    else
        echo "   ⬇ Baixando $model..."
        ollama pull "$model"
    fi
done

# === [6] Pré-carregar modelo principal ===
echo ""
echo "➡ Pré-carregando modelo principal na VRAM..."

# Faz uma requisição simples para carregar o modelo
curl -s http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1:7b",
  "messages": [{"role": "user", "content": "oi"}],
  "stream": false
}' > /dev/null 2>&1 &

# === [7] Iniciar Open WebUI (opcional) ===
echo ""
read -p "➡ Iniciar Open WebUI? [s/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo "➡ Iniciando Open WebUI..."
    docker rm -f open-webui 2>/dev/null || true
    
    docker run -d -p 3000:8080 \
        --add-host=host.docker.internal:host-gateway \
        -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
        -v open-webui:/app/backend/data \
        --name open-webui \
        --restart unless-stopped \
        ghcr.io/open-webui/open-webui:main
    
    echo " WebUI: http://localhost:3000"
fi

# === [8] Status final ===
echo ""
echo "=========================================="
echo "   Ollama rodando com paralelismo!"
echo "=========================================="
echo ""
echo "Endpoints:"
echo "  • API: http://localhost:11434"
echo "  • Chat: http://localhost:11434/api/chat"
echo ""
echo "Testar paralelismo:"
echo "  curl http://localhost:11434/api/chat -d '{\"model\":\"deepseek-r1:7b\",\"messages\":[{\"role\":\"user\",\"content\":\"oi\"}],\"stream\":false}'"
echo ""
echo "Ver logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Parar servidor:"
echo "  pkill -f 'ollama serve'"
echo ""
