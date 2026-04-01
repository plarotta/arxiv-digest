# ML Digest

Daily top-15 ML paper digest from arxiv. Fetches new submissions, selects candidates via tournament-style abstract culling, summarizes them using full paper text (not just abstracts), ranks them by significance, and generates a self-contained static HTML page.

## How It Works

The pipeline uses a tournament cull + full-text approach to select the best papers from each day's arxiv submissions:

```
Step 1: arxiv RSS → fetch ~500 papers
Step 2: Tournament cull → pick top ~60 from raw abstracts (5 API calls)
Step 3: Download PDFs for 60 survivors
Step 4: Full-text summarize 60 papers (12 API calls)
Step 5: Final rank → top 15 (1 API call)
Step 6: Generate static HTML
```

### Step 2 — Tournament Abstract Cull

Papers are shuffled (to avoid positional bias), split into batches of ~100, and each batch is sent to the LLM with raw abstracts. The model picks the top ~12 from each batch based on novelty, empirical strength, and subfield diversity. No summarization happens here — just selection. This replaces the old approach of LLM-summarizing all 500 papers (which wasted 25+ API calls on papers that were immediately discarded).

### Steps 3-4 — Full-Text Summarization

PDFs are downloaded from arxiv for the ~60 survivors (4-second delay between requests, cached in `.cache/pdfs/`). Text is extracted via PyMuPDF with cleanup (strips images, table rows, figure/table captions, references section, truncates to 25k chars). Papers are then summarized in batches of 5 using the full text. Papers with failed PDF extraction fall back to abstract-based summarization.

### Step 5 — Final Ranking

All ~60 summaries are sent to the LLM in a single call to select and rank the top 15. Supports an optional `--rank-provider gemini` flag to use Gemini 2.5 Pro for this step (higher-quality judgment).

## Setup

```bash
uv init
uv add feedparser jinja2 pymupdf openai tqdm
```

Set your API key:

```bash
export GROQ_API_KEY="your-key-here"

# Optional: for --rank-provider gemini
export GEMINI_API_KEY="your-key-here"
```

## Usage

```bash
# Full pipeline
uv run python run_pipeline.py

# Custom options
uv run python run_pipeline.py --top 10 --output docs/
uv run python run_pipeline.py --categories cs.LG cs.CL
uv run python run_pipeline.py --shortlist 40 --top 10

# Use Gemini Pro for final ranking (requires GEMINI_API_KEY)
uv run python run_pipeline.py --rank-provider gemini

# Skip PDF downloads (abstract-only mode, faster and cheaper)
uv run python run_pipeline.py --skip-pdf

# Fetch only, no LLM calls
uv run python run_pipeline.py --dry-run

# Reuse cached fetch data (useful on weekends when arxiv RSS is empty)
uv run python run_pipeline.py --from-cache .cache/2026-03-28_fetched.json
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `site` | Output directory for generated HTML |
| `--top` | `15` | Number of papers in the final selection |
| `--shortlist` | `60` | Number of candidates to keep after tournament cull |
| `--categories` | all 5 | arxiv categories to fetch (space-separated) |
| `--skip-pdf` | off | Skip PDF download/extraction; summarize from abstracts only |
| `--dry-run` | off | Fetch papers only, skip all LLM calls |
| `--from-cache` | none | Path to a cached `_fetched.json` file to use instead of RSS |
| `--cache-dir` | `.cache` | Directory for intermediate JSON and PDF files |
| `--rank-provider` | `groq` | LLM provider for final ranking (`groq` or `gemini`) |

## Models

| Task | Provider | Model | Notes |
|------|----------|-------|-------|
| Tournament cull | Groq | Llama 3.3 70B | Picks top candidates from raw abstracts |
| Summarization | Groq | Llama 3.3 70B | Full-text and abstract-based summaries |
| Final ranking | Groq (default) | Llama 3.3 70B | Can use `--rank-provider gemini` for Gemini 2.5 Pro |

Model constants are defined at the top of `summarize_and_rank.py`.

## Deployment (GitHub Pages)

1. Set `--output docs/` so the site lands in `/docs`
2. Push to GitHub
3. Enable Pages from `/docs` on the `main` branch
4. Set up a daily GitHub Action (see `.github/workflows/daily-digest.yml`)
5. Add `GROQ_API_KEY` to repo Settings > Secrets > Actions

The included workflow runs at 9 PM UTC daily (after arxiv updates ~8 PM UTC), generates the digest into `docs/`, and auto-commits.

## Caching

Intermediate data is saved in `.cache/` for debugging and reuse:

| File | Contents |
|------|----------|
| `{date}_fetched.json` | Raw paper metadata from RSS |
| `{date}_culled.json` | Papers surviving tournament cull (~60) |
| `{date}_fulltext_summaries.json` | Summaries using full PDF text |
| `{date}_selected.json` | Final top 15 with rankings and rationales |
| `pdfs/{arxiv_id}.pdf` | Downloaded PDFs (reused across runs) |

## Output Structure

```
site/
├── index.html              # Today's digest (overwritten daily)
└── archive/
    ├── 2025-03-27.html
    ├── 2025-03-28.html
    └── ...
```

## Project Structure

```
arxiv-digest/
├── run_pipeline.py          # CLI entrypoint — orchestrates the 6-step pipeline
├── fetch_arxiv.py           # RSS fetching and deduplication
├── summarize_and_rank.py    # Tournament cull, summarization, and ranking (Groq + Gemini)
├── pdf_extraction.py        # PDF download, text extraction, and cleanup
├── generate_site.py         # Jinja2 HTML generation (template is inline)
├── preview.py               # Sample data for previewing the site design
├── requirements.txt         # pip dependencies (legacy)
├── pyproject.toml           # uv project config
└── .github/workflows/
    └── daily-digest.yml     # Daily GitHub Actions workflow
```
