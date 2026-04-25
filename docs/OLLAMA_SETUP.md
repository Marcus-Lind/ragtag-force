# Ollama Setup Guide (Alternative LLM Backend)

RAG-Tag Force defaults to the **Anthropic API** (Claude Haiku 3.5), but can also
use **Ollama** for fully local, offline inference.

## 1. Install Ollama

### Windows
Download from [ollama.com/download](https://ollama.com/download/windows) and run the installer.

### macOS
```bash
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## 2. Pull the Model

```bash
ollama pull llama3.1:8b
```

This downloads ~4.7 GB. Requires at least 8 GB RAM (16 GB recommended).

## 3. Start the Server

```bash
ollama serve
```

Ollama runs at `http://localhost:11434` by default. Verify:

```bash
curl http://localhost:11434/api/tags
```

You should see `llama3.1:8b` in the model list.

## 4. Configure RAG-Tag Force

Edit your `.env` file to use Ollama instead of Anthropic:

```env
# Comment out Anthropic settings
# ANTHROPIC_API_KEY=sk-ant-...
# ANTHROPIC_MODEL=claude-haiku-4-20250506

# Uncomment Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

> **Note**: The current codebase is wired for Anthropic. Switching to Ollama
> requires modifying `src/llm/client.py` to use the Ollama Python SDK instead.
> This is left as a future enhancement.

## 5. Verify

```bash
curl http://localhost:11434/api/generate -d '{"model":"llama3.1:8b","prompt":"Hello","stream":false}'
```

You should get a JSON response with a generated completion.
