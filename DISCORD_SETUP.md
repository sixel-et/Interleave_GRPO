# Discord Bot Integration

Connect your trained models (or frontier models) to Discord using llmcord + vLLM.

## Architecture

```
Discord <--> llmcord <--> OpenAI-compatible API <--> Your Model
                              |
                         vLLM server (local)
                         or OpenAI/OpenRouter (frontier)
```

## Quick Start

### 1. Serve your model with vLLM

```bash
# Install vLLM
pip install vllm

# Serve your checkpoint (OpenAI-compatible API on port 8000)
vllm serve ./outputs/checkpoint-3900 \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name interleave-model
```

Test it:
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "interleave-model",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

### 2. Set up llmcord

```bash
# Clone llmcord
git clone https://github.com/jakobdylanc/llmcord.git
cd llmcord

# Install dependencies
pip install -r requirements.txt

# Copy example config
cp config-example.yaml config.yaml
```

### 3. Create Discord bot

1. Go to https://discord.com/developers/applications
2. Click "New Application" → name it
3. Go to "Bot" tab → click "Add Bot"
4. Enable "MESSAGE CONTENT INTENT"
5. Copy the bot token
6. Go to "OAuth2" → "URL Generator"
   - Select "bot" scope
   - Select permissions: Send Messages, Read Message History, etc.
7. Copy URL, open in browser, add bot to your server

### 4. Configure llmcord

Edit `config.yaml`:

```yaml
# Discord bot token
bot_token: "YOUR_DISCORD_BOT_TOKEN"

# LLM providers
providers:
  # Your local model via vLLM
  local:
    base_url: "http://localhost:8000/v1"
    api_key: "not-needed"
  
  # Frontier models via OpenRouter (optional)
  openrouter:
    base_url: "https://openrouter.ai/api/v1"
    api_key: "YOUR_OPENROUTER_KEY"
  
  # Direct OpenAI (optional)
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "YOUR_OPENAI_KEY"

# Available models
models:
  # Your trained model
  local/interleave-model:
    max_tokens: 1024
    temperature: 0.7
  
  # Frontier models via OpenRouter
  openrouter/anthropic/claude-3.5-sonnet:
    max_tokens: 4096
  
  openrouter/openai/gpt-4o:
    max_tokens: 4096

# Default model (what bot uses when @mentioned)
default_model: "local/interleave-model"

# Permissions (optional)
permissions:
  users:
    allowed_ids: []  # Empty = allow all
    blocked_ids: []
```

### 5. Run

```bash
# Terminal 1: vLLM server
vllm serve ./outputs/checkpoint-3900 --port 8000 --served-model-name interleave-model

# Terminal 2: Discord bot
cd llmcord
python llmcord.py
```

In Discord: `@YourBot Hello!`

## Switching Models

Admins can switch models with `/model`:
```
/model local/interleave-model      # Your trained model
/model openrouter/anthropic/claude-3.5-sonnet  # Claude
/model openrouter/openai/gpt-4o    # GPT-4
```

## Multiple Local Models

Run multiple vLLM instances on different ports:

```bash
# Baseline model
vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8001 --served-model-name baseline

# Trained model  
vllm serve ./outputs/checkpoint-3900 --port 8002 --served-model-name trained
```

Config:
```yaml
providers:
  baseline:
    base_url: "http://localhost:8001/v1"
  trained:
    base_url: "http://localhost:8002/v1"

models:
  baseline/baseline: {}
  trained/trained: {}
```

## Alternative: Ollama (simpler, slower)

If you convert your model to GGUF format:

```bash
# Install ollama
curl -fsSL https://ollama.com/install.sh | sh

# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./your-model.gguf
TEMPLATE """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
EOF

# Create and run
ollama create interleave -f Modelfile
ollama serve
```

Then use any Ollama-compatible Discord bot, or configure llmcord:
```yaml
providers:
  ollama:
    base_url: "http://localhost:11434/v1"
```

## Troubleshooting

**Bot doesn't respond:**
- Check MESSAGE CONTENT INTENT is enabled in Discord developer portal
- Verify bot has permissions in the channel
- Check vLLM server is running: `curl http://localhost:8000/v1/models`

**Slow responses:**
- vLLM needs GPU. Check `nvidia-smi` shows utilization
- Reduce `max_tokens` in config
- Use smaller model or quantization

**Out of memory:**
- Use `--gpu-memory-utilization 0.8` flag with vLLM
- Try quantization: `--quantization awq` or `--quantization gptq`
