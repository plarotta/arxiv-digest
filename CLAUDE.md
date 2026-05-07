# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Daily top-15 ML paper digest from arxiv. Fetches papers via RSS, selects candidates via tournament-style abstract culling, downloads full paper PDFs, summarizes using full text, and ranks to produce a self-contained static HTML page for GitHub Pages.

## Commands

```bash
uv add <package>                         # Add dependencies (managed via uv, never bare pip)
uv run python run_pipeline.py            # Full pipeline via Groq (requires GROQ_API_KEY)
uv run python run_pipeline.py --provider ollama  # Local inference via Ollama (no API keys)
uv run python run_pipeline.py --skip-pdf # Abstract-only mode (no PDF downloads)
uv run python run_pipeline.py --dry-run  # Fetch only, no LLM calls
uv run python run_pipeline.py --from-cache .cache/2026-03-28_fetched.json  # Reuse cached fetch
uv run python run_pipeline.py --top 10 --shortlist 40 --output docs/ --categories cs.LG cs.CL
uv run python run_pipeline.py --rank-provider gemini  # Use Gemini Pro for final ranking
uv run python fetch_arxiv.py             # Fetch-only test (prints first 5 papers)
uv run python pdf_extraction.py 2401.00001  # Test PDF download + text extraction for one paper
uv run python generate_site.py           # Generate preview with dummy data → site/index.html
```

No test suite or linter is configured. Dependencies are in `pyproject.toml` (the `requirements.txt` is legacy and not used).

## Architecture

Pipeline orchestrated by `run_pipeline.py` (6 steps, or 5 with `--skip-pdf`):

1. **fetch_arxiv.py** — Pulls RSS feeds for 5 categories (cs.LG, cs.CL, cs.AI, cs.CV, cs.RO), filters out replacements (~37% of RSS volume) by parsing `Announce Type` from abstract metadata, strips RSS prefix from abstracts, and deduplicates cross-listed papers by arxiv_id. Returns `ArxivPaper` dataclass instances.
2. **summarize_and_rank.py** `cull_abstracts_tournament()` — Tournament-style culling: shuffles papers, splits into batches. `run_pipeline.py` picks batch size by provider — **5 for Groq** (stays under the 6k TPM cap), **25 for Ollama** (no TPM, fewer calls = faster on CPU). LLM picks top survivors from each batch based on raw abstracts (no summarization). For ~500 papers this is ~100 calls on Groq, ~20 on Ollama → ~60 survivors.
3. **pdf_extraction.py** `download_and_extract()` — Downloads PDFs for ~60 survivors (4s delay between requests), extracts text via PyMuPDF. Strips images, tables, captions, and references before truncating to ~25k chars. Skipped with `--skip-pdf`.
4. **summarize_and_rank.py** `summarize_with_full_text()` — Summarizes survivors using full text. `run_pipeline.py` uses chunk size **1 on Groq** (per-call TPM ceiling) and **3 on Ollama**. Falls back to abstract-based summary if PDF extraction failed. With `--skip-pdf`, uses `summarize_all_abstracts()` in batches of 20 instead.
5. **summarize_and_rank.py** `rank_and_select()` — Final ranking selects top 15 from ~60 summarized candidates. 1 API call. `--rank-provider` defaults to `--provider`; can be overridden to `gemini` for Gemini Pro.
6. **generate_site.py** — Renders Jinja2 template (inline `HTML_TEMPLATE` string) to a self-contained HTML file. Writes both `index.html` and a dated archive copy to `{output}/archive/{date}.html`.

Data flow: `ArxivPaper` → dict → (tournament cull) → dict → (PDF text) → `PaperSummary` → dict → HTML

Intermediate JSON cached in `.cache/`: `_fetched.json`, `_culled.json`, `_fulltext_summaries.json`, `_selected.json`. PDFs cached in `.cache/pdfs/`.

## Key Details

- **LLM Providers:** Three are wired up — `groq` (default, cloud), `ollama` (local, what GH Actions uses), and `gemini` (only valid for `--rank-provider`). All accessed through the `openai` Python package with different base URLs.
- Models in `summarize_and_rank.py`:
  - Groq: `CULL_MODEL` = `llama-3.1-8b-instant`, `SUMMARIZE_MODEL` and `RANK_MODEL_GROQ` = `llama-3.3-70b-versatile`.
  - Gemini: `RANK_MODEL_GEMINI` = `gemini-2.5-pro`.
  - Ollama: `OLLAMA_MODEL` (default `gemma4:e2b`, env-overridable) — same model used for cull, summarize, and rank.
  - `_model_for(provider, task)` picks the right one.
- Env vars: `GROQ_API_KEY` (only required when `--provider groq` or `--rank-provider groq`), `GEMINI_API_KEY` or `GOOGLE_API_KEY` (only for `--rank-provider gemini`), `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`), `OLLAMA_MODEL` (default `gemma4:e2b`).
- `_call_llm()` skips its post-call rate-limit sleep when `provider == "ollama"`. The 60s post-call sleep on full-text summarize batches is also skipped for Ollama.
- PDF text extraction strips images, table rows, figure/table captions, and the references section before truncation (`MAX_CHARS` in `pdf_extraction.py`).
- `--from-cache` flag allows rerunning the pipeline from a cached `_fetched.json` (useful on weekends when arxiv RSS is empty).
- All LLM calls go through `_call_llm()` in `summarize_and_rank.py`, which handles markdown-fence stripping, rate-limit retries with exponential backoff (up to 5 retries), and a 2s sleep between calls (skipped for Ollama). All LLM responses are expected to be raw JSON (no markdown fences).
- `preview.py` contains sample paper data and a hardcoded output path from a different environment; use `generate_site.py` directly for local previews.
- The real entrypoint is `run_pipeline.py`.
- The GitHub Action (`.github/workflows/daily-digest.yml`) installs Ollama on the runner, caches `~/.ollama/models`, pulls `gemma4:e2b`, and runs the pipeline with `--provider ollama` daily at 9 PM UTC. Job timeout is 5.5h since CPU inference is slow.
