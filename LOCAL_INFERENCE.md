# Local CPU Inference with Ollama

Future option: run the entire pipeline locally on a CPU-only node using Ollama, eliminating all cloud API dependencies and rate limits. Trade-off is speed — expect 4-5 hour runs overnight.

## Why

- Groq free tier has tight TPM limits (12K for 70B models)
- No GPU needed — Ollama + llama.cpp runs on CPU
- No API keys, no rate limits, no cost
- Acceptable for a daily cron job with overnight window

## Setup

```bash
# Install Ollama (on the target node)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a small model that works well on CPU
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M

# Ollama exposes an OpenAI-compatible API at localhost:11434/v1
```

## Implementation (Minimal)

Since the codebase already uses the `openai` Python package, adding Ollama is ~20 lines:

1. Add to `summarize_and_rank.py`:
   ```python
   OLLAMA_BASE_URL = "http://localhost:11434/v1"
   CULL_MODEL_LOCAL = "phi3.5:3.8b-mini-instruct-q4_K_M"
   SUMMARIZE_MODEL_LOCAL = "phi3.5:3.8b-mini-instruct-q4_K_M"
   RANK_MODEL_LOCAL = "phi3.5:3.8b-mini-instruct-q4_K_M"
   ```

2. Add `ollama` case to `_get_client()`:
   ```python
   elif provider == "ollama":
       _clients[provider] = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
   ```

3. Skip rate-limit sleep in `_call_llm()` when provider is `ollama`.

4. Add `--provider ollama` flag to `run_pipeline.py`.

## Expected Performance (CPU-only, 3.8B Q4_K_M model)

| Step | Calls | Time per call | Total |
|------|-------|---------------|-------|
| Cull (batches of 50) | ~28 | ~5-7 min | ~2-3 hours |
| Summarize (batches of 5) | ~12 | ~7-10 min | ~1.5-2 hours |
| Rank (1 call) | 1 | ~10-20 min | ~15 min |
| **Total** | **~41** | | **~4-5 hours** |

Prefill (processing input tokens) is faster than generation on CPU: ~50-100 tokens/sec for a 3B model vs ~5 tokens/sec for output generation.

## Model Options (CPU-friendly, Q4_K_M quantization)

| Model | Size | RAM | Notes |
|-------|------|-----|-------|
| phi3.5:3.8b-mini-instruct | 3.8B | ~2.5 GB | Best balance of speed/quality for structured output |
| llama3.2:3b | 3B | ~2 GB | Faster, slightly less capable |
| mistral:7b | 7B | ~4.5 GB | Better quality, 2x slower |
| qwen2.5:7b | 7B | ~4.5 GB | Strong at structured JSON |

## Caveats

- **JSON reliability**: 3B models occasionally produce malformed JSON. The existing fallback/retry logic in `_call_llm()` handles this, but expect more parse errors than with 70B cloud models.
- **Quality**: Paper selection judgment from a 3B model is weaker than 70B. Mitigated by tournament structure (multiple chances to surface good papers).
- **First run**: Ollama downloads the model on first `ollama pull` (~2 GB for a 3B Q4_K_M model).
