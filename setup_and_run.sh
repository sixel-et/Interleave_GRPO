#!/bin/bash
# setup_and_run.sh
# 
# Setup for Interleave GRPO training on Runpod with single A40
# 
# Usage:
#   bash setup_and_run.sh

set -e

echo "========================================"
echo "Interleave GRPO Setup - Single A40"
echo "========================================"
echo "Time: $(date)"
echo ""

WORKSPACE="/workspace"
REPO_NAME="Interleave_GRPO"
VENV_DIR="$WORKSPACE/venv"
MODEL="meta-llama/Llama-3.2-3B-Instruct"

# ============================================================================
# INSTALL SYSTEM TOOLS
# ============================================================================

echo ">>> Installing system tools..."
apt-get update -qq && apt-get install -y -qq tmux nano > /dev/null
echo "✓ Installed tmux, nano"

# ============================================================================
# CACHE DIRECTORIES
# ============================================================================

echo ">>> Setting up cache directories..."
export HF_HOME="$WORKSPACE/.cache/huggingface"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"
echo "✓ Caches on /workspace"

# ============================================================================
# VIRTUAL ENVIRONMENT
# ============================================================================

echo ""
echo ">>> Setting up venv at $VENV_DIR..."

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    if python -c "import trl" 2>/dev/null; then
        echo "✓ Existing venv with packages found"
        SKIP_INSTALL=1
    else
        SKIP_INSTALL=0
    fi
else
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "✓ Created new venv"
    SKIP_INSTALL=0
fi

echo 'source /workspace/venv/bin/activate' >> ~/.bashrc
echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

if [ "$SKIP_INSTALL" != "1" ]; then
    echo ""
    echo ">>> Installing dependencies..."
    pip install --upgrade pip -q
    pip install torch transformers accelerate trl datasets huggingface-hub bitsandbytes -q
    echo "✓ Dependencies installed"
fi

echo ""
echo ">>> Versions:"
pip show trl transformers | grep -E "^(Name|Version)"

# ============================================================================
# VERIFY GPU
# ============================================================================

echo ""
echo ">>> GPU check..."
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ============================================================================
# HUGGINGFACE AUTH
# ============================================================================

echo ""
echo ">>> HuggingFace auth..."

if [[ -z "$HF_TOKEN" ]] && [[ -f "$WORKSPACE/.cache/huggingface/token" ]]; then
    export HF_TOKEN=$(cat "$WORKSPACE/.cache/huggingface/token")
    echo "  Loaded from network volume"
fi

if [[ -n "$HF_TOKEN" ]]; then
    echo "✓ HF_TOKEN set"
    mkdir -p ~/.cache/huggingface
    echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
    echo -n "$HF_TOKEN" > "$WORKSPACE/.cache/huggingface/token"
elif huggingface-cli whoami &>/dev/null; then
    echo "✓ Already logged in"
else
    echo "Not logged in. Run: huggingface-cli login"
fi

# ============================================================================
# DONE
# ============================================================================

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To train:"
echo "  cd $WORKSPACE/$REPO_NAME"
echo "  python interleave_grpo.py"
echo ""