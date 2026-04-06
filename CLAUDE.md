# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Daily top-15 ML paper digest from arxiv. Fetches papers via RSS, selects candidates via tournament-style abstract culling, downloads full paper PDFs, summarizes using full text, and ranks to produce a self-contained static HTML page for GitHub Pages.

## Commands

```bash
uv add <package>                         # Add dependencies (managed via uv, never bare pip)
uv run python run_pipeline.py            # Full pipeline (requires GROQ_API_KEY)
uv run python run_pipeline.py --skip-pdf # Abstract-only mode (no PDF downloads)
uv run python run_pipeline.py --dry-run  # Fetch only, no LLM calls
uv run python run_pipeline.py --from-cache .cache/2026-03-28_fetched.json  # Reuse cached fetch
uv run python run_pipeline.py --top 10 --shortlist 40 --output docs/ --categories cs.LG cs.CL
uv run python run_pipeline.py --rank-provider gemini  # Use Gemini Pro for final ranking
uv run python fetch_arxiv.py             # Fetch-only test (prints first 5 papers)
uv run python pdf_extraction.py 2401.00001  # Test PDF download + text extraction for one paper
uv run python generate_site.py           # Generate preview with dummy data → site/index.html
```

No test suite or linter is configured.

## Architecture

Pipeline orchestrated by `run_pipeline.py` (6 steps, or 5 with `--skip-pdf`):

1. **fetch_arxiv.py** — Pulls RSS feeds for 5 categories (cs.LG, cs.CL, cs.AI, cs.CV, cs.RO), filters out replacements (~37% of RSS volume) by parsing `Announce Type` from abstract metadata, strips RSS prefix from abstracts, and deduplicates cross-listed papers by arxiv_id. Returns `ArxivPaper` dataclass instances.
2. **summarize_and_rank.py** `cull_abstracts_tournament()` — Tournament-style culling: shuffles papers, splits into batches of 50, LLM picks top ~6 from each batch based on raw abstracts (no summarization). Uses `llama-3.1-8b-instant` for higher TPM limits. ~10 API calls for 500 papers → ~60 survivors.
3. **pdf_extraction.py** `download_and_extract()` — Downloads PDFs for ~60 survivors (4s delay between requests), extracts text via PyMuPDF. Strips images, tables, captions, and references before truncating to ~25k chars. Skipped with `--skip-pdf`.
4. **summarize_and_rank.py** `summarize_with_full_text()` — Summarizes survivors using full text in batches of 5. Falls back to abstract-based summary if PDF extraction failed. ~12 API calls. With `--skip-pdf`, uses `summarize_all_abstracts()` in batches of 20 instead.
5. **summarize_and_rank.py** `rank_and_select()` — Final ranking selects top 15 from ~60 summarized candidates. 1 API call. Supports `--rank-provider gemini` for Gemini Pro.
6. **generate_site.py** — Renders Jinja2 template (inline `HTML_TEMPLATE` string) to a self-contained HTML file. Writes both `index.html` and a dated archive copy.

Data flow: `ArxivPaper` → dict → (tournament cull) → dict → (PDF text) → `PaperSummary` → dict → HTML

Intermediate JSON cached in `.cache/`: `_fetched.json`, `_culled.json`, `_fulltext_summaries.json`, `_selected.json`. PDFs cached in `.cache/pdfs/`.

## Key Details

- **LLM Providers:** Groq (primary, via OpenAI-compatible API) and Gemini (optional, for final ranking only). Both accessed through the `openai` Python package with different base URLs.
- Models are set in `summarize_and_rank.py`: `CULL_MODEL` (`llama-3.1-8b-instant` — small model for high-volume culling), `SUMMARIZE_MODEL` and `RANK_MODEL_GROQ` (`llama-3.3-70b-versatile`), `RANK_MODEL_GEMINI` (`gemini-2.5-pro`).
- Env vars: `GROQ_API_KEY` (required), `GEMINI_API_KEY` or `GOOGLE_API_KEY` (optional, only for `--rank-provider gemini`).
- PDF text extraction strips images, table rows, figure/table captions, and the references section before truncation (`MAX_CHARS` in `pdf_extraction.py`).
- `--from-cache` flag allows rerunning the pipeline from a cached `_fetched.json` (useful on weekends when arxiv RSS is empty).
- All LLM calls go through `_call_llm()` in `summarize_and_rank.py`, which handles markdown-fence stripping, rate-limit retries with exponential backoff (up to 5 retries), and a 2s sleep between calls. All LLM responses are expected to be raw JSON (no markdown fences).
- `preview.py` contains sample paper data and a hardcoded output path from a different environment; use `generate_site.py` directly for local previews.
- `main.py` is a stub placeholder — the real entrypoint is `run_pipeline.py`.
- The GitHub Action (`.github/workflows/daily-digest.yml`) runs at 9 PM UTC daily, outputs to `docs/`, and auto-commits.
