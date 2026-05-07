# Local Inference

Local inference is now a first-class option, not a future plan. The pipeline runs on `gemma4:e2b` via [Ollama](https://ollama.com) when invoked with `--provider ollama`, and that's what the daily GitHub Actions workflow uses.

## Run locally

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:e2b      # ~7.2 GB
ollama serve &
uv run python run_pipeline.py --provider ollama
```

Override the model via env var:

```bash
OLLAMA_MODEL=gemma4:e4b uv run python run_pipeline.py --provider ollama
```

## Mix providers

Cull + summarize on Ollama (free, slow), rank on Groq or Gemini (better judgment):

```bash
uv run python run_pipeline.py --provider ollama --rank-provider gemini
```

## What changes vs. Groq

`run_pipeline.py` increases batch sizes when `--provider ollama` is set (cull batches 5 → 25, full-text summarize chunks 1 → 3). There's no TPM cap, so fewer/larger calls are cheaper on a CPU runner. `_call_llm()` also skips its rate-limit sleep when the provider is Ollama.

## Caveats

- **Speed.** CPU inference is much slower than Groq. The GH Actions workflow has a 5.5h timeout to accommodate this.
- **JSON reliability.** Small models occasionally produce malformed JSON; the existing fallback in `_call_llm()` handles parse errors.
- **Quality.** A 2.3B-effective model produces weaker paper-selection judgment than a 70B model. Mitigated by the tournament structure and (optionally) running rank on a stronger remote model via `--rank-provider`.
