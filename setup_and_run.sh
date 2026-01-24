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

# Add to bashrc for future sessions
grep -q "source /workspace/venv/bin/activate" ~/.bashrc || echo 'source /workspace/venv/bin/activate' >> ~/.bashrc
grep -q "export HF_HOME=" ~/.bashrc || echo 'export HF_HOME="/workspace/.cache/huggingface"' >> ~/.bashrc

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

if [ "$SKIP_INSTALL" != "1" ]; then
    echo ""
    echo ">>> Installing dependencies..."
    pip install --upgrade pip -q
    pip install torch transformers accelerate trl datasets huggingface-hub bitsandbytes wandb -q
    echo "✓ Dependencies installed"
fi

echo ""
echo ">>> Versions:"
pip show trl transformers wandb | grep -E "^(Name|Version)"

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
    echo "⚠ Not logged in. Run: huggingface-cli login"
fi

# ============================================================================
# WANDB AUTH
# ============================================================================

echo ""
echo ">>> WandB auth..."

if [[ -z "$WANDB_API_KEY" ]] && [[ -f "$WORKSPACE/.wandb_api_key" ]]; then
    export WANDB_API_KEY=$(cat "$WORKSPACE/.wandb_api_key")
    echo "  Loaded from network volume"
fi

if [[ -n "$WANDB_API_KEY" ]]; then
    echo "✓ WANDB_API_KEY set"
    # Save to network volume for persistence
    echo -n "$WANDB_API_KEY" > "$WORKSPACE/.wandb_api_key"
elif wandb status 2>/dev/null | grep -q "Logged in"; then
    echo "✓ Already logged in"
    # Save key to network volume if logged in
    if [[ -f ~/.netrc ]] && [[ ! -f "$WORKSPACE/.wandb_api_key" ]]; then
        grep -A2 "api.wandb.ai" ~/.netrc | grep password | awk '{print $2}' > "$WORKSPACE/.wandb_api_key" 2>/dev/null || true
    fi
else
    echo "⚠ Not logged in. Run: wandb login"
    echo "  Or: echo 'YOUR_KEY' > /workspace/.wandb_api_key"
fi

# Add wandb key export to bashrc
grep -q "WANDB_API_KEY" ~/.bashrc || echo '[ -f "/workspace/.wandb_api_key" ] && export WANDB_API_KEY=$(cat /workspace/.wandb_api_key)' >> ~/.bashrc

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
echo "First time auth (if needed):"
echo "  huggingface-cli login"
echo "  wandb login"
echo ""