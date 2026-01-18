#!/bin/bash

# === Configure Ollama Systemd Service for Parallelism ===
# This script updates your Ollama systemd service to enable parallel processing

set -e

echo "=========================================="
echo "  Ollama Parallel Configuration Setup"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "  This script requires sudo privileges."
    echo "   Running with sudo..."
    exec sudo bash "$0" "$@"
fi

# Configuration
SERVICE_FILE="/etc/systemd/system/ollama.service"
BACKUP_FILE="/etc/systemd/system/ollama.service.backup.$(date +%Y%m%d_%H%M%S)"

# Parallel settings - adjust based on your VRAM
OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-6}
OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS:-2}
OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-15m}
OLLAMA_GPU_OVERHEAD=${OLLAMA_GPU_OVERHEAD:-1024MiB}

echo "Configuration:"
echo "  • Parallel requests: $OLLAMA_NUM_PARALLEL"
echo "  • Max loaded models: $OLLAMA_MAX_LOADED_MODELS"
echo "  • Keep alive: $OLLAMA_KEEP_ALIVE"
echo "  • GPU overhead: $OLLAMA_GPU_OVERHEAD"
echo ""

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo " Error: Ollama service file not found at $SERVICE_FILE"
    echo "   Is Ollama installed as a service?"
    exit 1
fi

echo "✓ Found Ollama service file"

# Backup original file
echo "➡ Creating backup at: $BACKUP_FILE"
cp "$SERVICE_FILE" "$BACKUP_FILE"
echo "✓ Backup created"

# Check if Environment variables already exist
if grep -q "Environment=\"OLLAMA_NUM_PARALLEL" "$SERVICE_FILE"; then
    echo ""
    echo "  OLLAMA environment variables already present in service file."
    echo "   Current configuration:"
    grep "Environment=\"OLLAMA" "$SERVICE_FILE" || true
    echo ""
    read -p "Do you want to update them? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. No changes made."
        exit 0
    fi
    
    # Remove old environment variables
    sed -i '/Environment="OLLAMA_/d' "$SERVICE_FILE"
fi

# Add environment variables to [Service] section
echo ""
echo "➡ Adding parallel configuration to service file..."

# Find the [Service] section and add environment variables
awk -v num_parallel="$OLLAMA_NUM_PARALLEL" \
    -v max_models="$OLLAMA_MAX_LOADED_MODELS" \
    -v keep_alive="$OLLAMA_KEEP_ALIVE" \
    -v gpu_overhead="$OLLAMA_GPU_OVERHEAD" '
/^\[Service\]/ {
    print $0
    print "# Parallel processing configuration"
    print "Environment=\"OLLAMA_NUM_PARALLEL=" num_parallel "\""
    print "Environment=\"OLLAMA_MAX_LOADED_MODELS=" max_models "\""
    print "Environment=\"OLLAMA_KEEP_ALIVE=" keep_alive "\""
    print "Environment=\"OLLAMA_GPU_OVERHEAD=" gpu_overhead "\""
    next
}
{print}
' "$SERVICE_FILE" > "${SERVICE_FILE}.tmp" && mv "${SERVICE_FILE}.tmp" "$SERVICE_FILE"

echo "✓ Configuration added to service file"

# Show the updated configuration
echo ""
echo "Updated service configuration:"
echo "─────────────────────────────"
grep -A 5 "\[Service\]" "$SERVICE_FILE" | head -n 6
echo ""

# Reload systemd and restart service
echo "➡ Reloading systemd daemon..."
systemctl daemon-reload
echo "✓ Daemon reloaded"

echo ""
echo "➡ Restarting Ollama service..."
systemctl restart ollama
echo "✓ Service restarted"

# Wait for service to start
sleep 2

# Check service status
echo ""
echo "➡ Checking service status..."
if systemctl is-active --quiet ollama; then
    echo "✓ Ollama service is running"
    
    # Verify configuration from systemd
    echo ""
    echo "Verifying configuration from systemd service:"
    systemctl show ollama --property=Environment | grep -o 'OLLAMA_[^=]*=[^ ]*' | while read -r var; do
        echo "  ✓ $var"
    done
    
    # Also check service file directly
    if grep -q "Environment=\"OLLAMA_NUM_PARALLEL" "$SERVICE_FILE"; then
        echo ""
        echo "✓ Environment variables confirmed in service file"
    else
        echo ""
        echo "  Warning: Could not verify environment variables in service file"
    fi
else
    echo " Service failed to start!"
    echo "   Check logs with: sudo journalctl -u ollama -n 50"
    exit 1
fi

echo ""
echo "=========================================="
echo "   Configuration Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Verify configuration:"
echo "     python -m scripts.setup --check"
echo "     python -m scripts.watch"
echo ""
echo "  2. Run performance test:"
echo "     python scripts/test_parallel_ollama.py"
echo ""
echo "  3. Start running experiments:"
echo "     python scripts/main.py"
echo ""
echo "To revert changes:"
echo "  sudo cp $BACKUP_FILE $SERVICE_FILE"
echo "  sudo systemctl daemon-reload && sudo systemctl restart ollama"
echo ""
