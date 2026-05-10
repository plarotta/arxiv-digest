"""
summarize_and_rank.py — Summarize arxiv papers and rank them using LLMs.

Pipeline:
  1. Tournament cull: pick top candidates from raw abstracts (no summarization)
  2. Full-text summarize: summarize only the survivors using full PDF text
  3. Final rank: select top N from the summarized candidates

Providers:
  - Groq (default): Llama 3.3 70B via OpenAI-compatible API
  - Gemini (optional): Gemini 2.5 Pro for final ranking only
"""

import json
import os
import random
import re
import time
from openai import OpenAI, APIStatusError, APIConnectionError
from dataclasses import dataclass, asdict
from typing import Optional
from tqdm import tqdm

# ── Provider config ──────────────────────────────────────────
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

CULL_MODEL = "llama-3.1-8b-instant"  # Small model for high-volume cull (higher TPM limits)
SUMMARIZE_MODEL = "llama-3.3-70b-versatile"
RANK_MODEL_GROQ = "llama-3.3-70b-versatile"
RANK_MODEL_GEMINI = "gemini-2.5-pro"

# Local Ollama model — single small model used for cull, summarize, and rank.
# Default sized for GitHub Actions free runners (4-core CPU, 16GB RAM, 14GB SSD).
# Override with $OLLAMA_MODEL (e.g. "gemma4:e2b" if SSD-constrained).
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")

RATE_LIMIT_SLEEP = 2  # seconds between API calls

_clients: dict[str, OpenAI] = {}


def _get_client(provider: str = "groq") -> OpenAI:
    """Get or create an OpenAI-compatible client for the given provider."""
    if provider not in _clients:
        if provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
            _clients[provider] = OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)
        elif provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
            _clients[provider] = OpenAI(base_url=GEMINI_BASE_URL, api_key=api_key)
        elif provider == "ollama":
            _clients[provider] = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    return _clients[provider]


def _parse_json_list(text: str) -> list:
    """Parse an LLM response that should be a list. Accepts a raw JSON array,
    or an object wrapping a list under a common key (results/selections/papers/etc.),
    or any object whose first list-valued field is the payload. Returns [] on failure."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("results", "selections", "papers", "items", "summaries", "rankings", "data"):
            if isinstance(data.get(k), list):
                return data[k]
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def _model_for(provider: str, task: str) -> str:
    """Pick the right model for (provider, task) where task is 'cull', 'summarize', or 'rank'."""
    if provider == "ollama":
        return OLLAMA_MODEL
    if provider == "gemini":
        return RANK_MODEL_GEMINI
    # groq
    return CULL_MODEL if task == "cull" else SUMMARIZE_MODEL


def _call_llm(
    system: str,
    user: str,
    model: str = SUMMARIZE_MODEL,
    provider: str = "groq",
    max_tokens: int = 4096,
    max_retries: int = 5,
    post_call_sleep: float | None = None,
) -> str:
    """Make an LLM API call and return the text response.
    Handles markdown-fence stripping, rate-limit retries with backoff."""
    client = _get_client(provider)
    # Local Ollama has no rate limits or TPM caps — skip post-call sleep entirely.
    if provider == "ollama":
        post_call_sleep = 0
    # Force JSON output via grammar-constrained decoding. Small models like
    # gemma4 ignore "respond with JSON" instructions on long contexts; JSON
    # mode makes non-JSON output impossible at the decoder level.
    kwargs = {}
    if provider == "ollama":
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
                **kwargs,
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            time.sleep(post_call_sleep if post_call_sleep is not None else RATE_LIMIT_SLEEP)
            return text
        except APIConnectionError:
                wait = min(10 * (2 ** attempt), 120)
                tqdm.write(f"  Connection error (attempt {attempt + 1}/{max_retries}), waiting {wait:.0f}s...")
                time.sleep(wait)
        except APIStatusError as e:
            if e.status_code in (413, 429):
                # Extract retry delay from error if available
                retry_match = re.search(r"retry in (\d+\.?\d*)", str(e), re.IGNORECASE)
                wait = float(retry_match.group(1)) if retry_match else min(30 * (2 ** attempt), 120)
                tqdm.write(f"  Rate limited (attempt {attempt + 1}/{max_retries}), waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


# ── Data model ───────────────────────────────────────────────

@dataclass
class PaperSummary:
    arxiv_id: str
    title: str
    authors: list[str]
    categories: list[str]
    link: str
    summary: str          # 2-4 sentence summary
    contributions: list[str]  # Key contributions as bullet points
    relevance_tags: list[str] # e.g. ["architectures", "training", "rl", "nlp"]
    affiliations: list[str] = None  # e.g. ["MIT", "Google DeepMind"]
    abstract: str = ""        # original arxiv abstract
    rank: Optional[int] = None
    rank_rationale: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Prompts ──────────────────────────────────────────────────

CULL_SYSTEM = """You are a senior ML researcher scanning today's new arxiv submissions.
From the following batch of papers, select exactly {k} that are most likely to represent
significant, novel, or impactful work. Judge based on:
- Genuine methodological novelty over incremental improvements
- Claims of strong empirical results or clear SOTA advances
- New architectures, training paradigms, or theoretical insights
- Potential to influence future research directions
- Diversity across subfields (don't pick all NLP papers if there's great vision/RL work too)

Respond with a JSON object of the form:
{{"results": [
  {{"arxiv_id": "...", "rationale": "One sentence on why this paper looks promising."}},
  ... exactly {k} entries total ...
]}}

Each arxiv_id MUST be one of the IDs from the input. Return ONLY valid JSON, no markdown fences."""


SUMMARIZE_SYSTEM = """You are an expert ML researcher summarizing new arxiv papers.
For each paper, provide:
1. A concise 2-4 sentence summary capturing what the paper does and why it matters.
2. A list of key contributions (1-4 items, each a single sentence).
3. Relevance tags from: [architectures, training, optimization, nlp, vision, robotics, rl, generative, theory, efficiency, safety, multimodal, data, evaluation, agents, other].

Respond with a JSON object of the form:
{
  "results": [
    {
      "arxiv_id": "...",
      "summary": "...",
      "contributions": ["...", "..."],
      "relevance_tags": ["...", "..."]
    },
    ... one entry per input paper ...
  ]
}

Be precise and technical. Focus on what's genuinely novel vs. incremental.
Return ONLY valid JSON, no markdown fences."""


RESUMMARIZE_SYSTEM = """You are an expert ML researcher summarizing arxiv papers using their full text.
You have access to the complete paper, not just the abstract. Provide detailed, technical summaries.
For each paper:
1. A concise 3-5 sentence summary capturing the method, key results, and significance.
2. A list of key contributions (1-4 items, each a single sentence) drawn from the actual experiments and results.
3. Relevance tags from: [architectures, training, optimization, nlp, vision, robotics, rl, generative, theory, efficiency, safety, multimodal, data, evaluation, agents, other].
4. Author affiliations — extract the unique university or organization names from the paper header (e.g. ["MIT", "Google DeepMind"]). If not clearly stated, return an empty list.

Respond with a JSON object of the form:
{
  "results": [
    {
      "arxiv_id": "...",
      "summary": "...",
      "contributions": ["...", "..."],
      "relevance_tags": ["...", "..."],
      "affiliations": ["...", "..."]
    },
    ... one entry per input paper ...
  ]
}

Be precise and technical. Highlight specific quantitative results where available.
Return ONLY valid JSON, no markdown fences."""


def _rank_system(top_n: int) -> str:
    return f"""You are a senior ML researcher selecting the day's most important papers.
You will receive summaries of today's new arxiv papers. Select exactly {top_n} papers that represent
the most significant, novel, or impactful work. Prioritize:
- Genuine methodological novelty over incremental improvements
- Strong empirical results with clear SOTA gains
- New architectures, training paradigms, or theoretical insights
- Papers likely to influence future research directions
- Breadth across subfields (don't pick all NLP papers if there's great vision/RL work too)

Respond with a JSON object of the form:
{{"results": [
  {{"arxiv_id": "...", "rank": 1, "rationale": "One sentence on why this paper made the cut."}},
  ... exactly {top_n} entries, ranked #1 (best) to #{top_n} ...
]}}

Return ONLY valid JSON, no markdown fences."""


# ── Tournament cull ──────────────────────────────────────────

def cull_abstracts_tournament(
    papers: list[dict],
    survivors_per_batch: int = 5,
    batch_size: int = 50,
    provider: str = "groq",
) -> list[dict]:
    """Select top candidates from raw abstracts using tournament-style batching.
    Shuffles papers to avoid positional bias, splits into batches, and asks
    the LLM to pick the best from each batch. No summarization is performed."""
    shuffled = papers.copy()
    random.shuffle(shuffled)

    batches = [shuffled[i:i + batch_size] for i in range(0, len(shuffled), batch_size)]
    paper_lookup = {p["arxiv_id"]: p for p in papers}
    surviving_ids: list[dict] = []  # [{arxiv_id, rationale}]

    for batch in tqdm(batches, desc="Tournament cull (abstracts)", unit="batch"):
        # Proportionally adjust survivors for smaller last batch
        k = max(1, round(survivors_per_batch * len(batch) / batch_size))

        papers_text = "\n\n".join(
            f"--- Paper {p['arxiv_id']} ---\n"
            f"Title: {p['title']}\n"
            f"Authors: {', '.join(p['authors'][:5])}{'...' if len(p['authors']) > 5 else ''}\n"
            f"Categories: {', '.join(p['categories'])}\n"
            f"Abstract: {p['abstract'][:250] + ' ... ' + p['abstract'][-250:] if len(p['abstract']) > 500 else p['abstract']}"
            for p in batch
        )

        text = _call_llm(
            system=CULL_SYSTEM.format(k=k),
            user=f"Review these {len(batch)} papers and select the top {k}:\n\n{papers_text}",
            model=_model_for(provider, "cull"),
            provider=provider,
            max_tokens=4096,
        )

        try:
            selections = _parse_json_list(text)
            # Drop any non-dict items (e.g. plain strings) — the model occasionally
            # returns a list of arxiv_ids instead of structured objects.
            selections = [s for s in selections if isinstance(s, dict)]
            if not selections:
                raise ValueError("no dict-shaped selections")
            surviving_ids.extend(selections)
        except (json.JSONDecodeError, ValueError) as e:
            tqdm.write(f"  WARNING: parse error in cull batch: {e}")
            tqdm.write(f"  Raw response (first 500 chars): {text[:500]}")
            # Fallback: keep first k papers from this batch
            for p in batch[:k]:
                surviving_ids.append({"arxiv_id": p["arxiv_id"], "rationale": "Fallback (parse error)"})

    # Return full paper dicts for survivors
    survivors = []
    for s in surviving_ids:
        if not isinstance(s, dict):
            continue
        aid = s.get("arxiv_id", "")
        if aid in paper_lookup:
            paper = paper_lookup[aid].copy()
            paper["cull_rationale"] = s.get("rationale", "")
            survivors.append(paper)

    print(f"  Culled {len(papers)} → {len(survivors)} papers.")
    return survivors


# ── Summarization ────────────────────────────────────────────

def summarize_batch(papers: list[dict], provider: str = "groq") -> list[dict]:
    """Summarize a batch of papers from their abstracts."""
    papers_text = "\n\n".join(
        f"--- Paper {p['arxiv_id']} ---\n"
        f"Title: {p['title']}\n"
        f"Authors: {', '.join(p['authors'][:5])}{'...' if len(p['authors']) > 5 else ''}\n"
        f"Categories: {', '.join(p['categories'])}\n"
        f"Abstract: {p['abstract']}"
        for p in papers
    )

    text = _call_llm(
        system=SUMMARIZE_SYSTEM,
        user=f"Summarize these {len(papers)} papers:\n\n{papers_text}",
        model=_model_for(provider, "summarize"),
        provider=provider,
        max_tokens=4096,
    )

    try:
        results = _parse_json_list(text)
        return [r for r in results if isinstance(r, dict)]
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error in summarize batch: {e}")
        print(f"  Raw response (first 500 chars): {text[:500]}")
        return []


def summarize_all_abstracts(
    papers: list[dict],
    chunk_size: int = 20,
    provider: str = "groq",
) -> list[PaperSummary]:
    """Summarize papers from abstracts in batches. Used in --skip-pdf mode."""
    chunks = [papers[i:i + chunk_size] for i in range(0, len(papers), chunk_size)]
    all_summaries = []
    paper_lookup = {p["arxiv_id"]: p for p in papers}

    for chunk in tqdm(chunks, desc="Summarizing (abstracts)", unit="batch"):
        batch_results = summarize_batch(chunk, provider=provider)
        for s in batch_results:
            aid = s.get("arxiv_id", "")
            orig = paper_lookup.get(aid, {})
            summary = PaperSummary(
                arxiv_id=aid,
                title=orig.get("title", "Unknown"),
                authors=orig.get("authors", []),
                categories=orig.get("categories", []),
                link=orig.get("link", f"https://arxiv.org/abs/{aid}"),
                summary=s.get("summary", ""),
                contributions=s.get("contributions", []),
                relevance_tags=s.get("relevance_tags", []),
                abstract=orig.get("abstract", ""),
            )
            all_summaries.append(summary)

    return all_summaries


# ── Full-text re-summarization ───────────────────────────────

def summarize_with_full_text(
    papers: list[dict],
    full_texts: dict[str, str],
    chunk_size: int = 5,
    provider: str = "groq",
) -> list[PaperSummary]:
    """Summarize papers using full paper text. Papers without full text
    fall back to abstract-based summarization."""
    with_text = [p for p in papers if p["arxiv_id"] in full_texts]
    without_text = [p for p in papers if p["arxiv_id"] not in full_texts]

    if without_text:
        print(f"  {len(without_text)} papers without full text — summarizing from abstracts.")

    # Summarize papers that have full text
    all_summaries = []
    paper_lookup = {p["arxiv_id"]: p for p in papers}

    if with_text:
        chunks = [with_text[i:i + chunk_size] for i in range(0, len(with_text), chunk_size)]
        for chunk in tqdm(chunks, desc="Summarizing (full text)", unit="batch"):
            papers_text = "\n\n".join(
                f"--- Paper {p['arxiv_id']} ---\n"
                f"Title: {p['title']}\n"
                f"Authors: {', '.join(p['authors'][:5])}{'...' if len(p['authors']) > 5 else ''}\n"
                f"Categories: {', '.join(p['categories'])}\n"
                f"Full Text:\n{full_texts[p['arxiv_id']][:8000]}"
                for p in chunk
            )

            text = _call_llm(
                system=RESUMMARIZE_SYSTEM,
                user=f"Summarize these {len(chunk)} papers using their full text:\n\n{papers_text}",
                model=_model_for(provider, "summarize"),
                provider=provider,
                max_tokens=8192,
                # 60s post-call sleep is to stay under Groq 70B's TPM limit; not needed for ollama.
                post_call_sleep=None if provider == "ollama" else 60,
            )

            try:
                batch_results = _parse_json_list(text)
                batch_results = [r for r in batch_results if isinstance(r, dict)]
            except json.JSONDecodeError as e:
                tqdm.write(f"    WARNING: JSON parse error in full-text batch: {e}")
                # Fallback: create bare summaries from abstracts for this chunk
                for p in chunk:
                    all_summaries.append(PaperSummary(
                        arxiv_id=p["arxiv_id"],
                        title=p.get("title", "Unknown"),
                        authors=p.get("authors", []),
                        categories=p.get("categories", []),
                        link=p.get("link", f"https://arxiv.org/abs/{p['arxiv_id']}"),
                        summary=p.get("abstract", ""),
                        contributions=[],
                        relevance_tags=[],
                        abstract=p.get("abstract", ""),
                    ))
                continue

            for r in batch_results:
                aid = r.get("arxiv_id", "")
                orig = paper_lookup.get(aid, {})
                all_summaries.append(PaperSummary(
                    arxiv_id=aid,
                    title=orig.get("title", "Unknown"),
                    authors=orig.get("authors", []),
                    categories=orig.get("categories", []),
                    link=orig.get("link", f"https://arxiv.org/abs/{aid}"),
                    summary=r.get("summary", ""),
                    contributions=r.get("contributions", []),
                    relevance_tags=r.get("relevance_tags", []),
                    affiliations=r.get("affiliations", []),
                    abstract=orig.get("abstract", ""),
                ))

    # Fallback: summarize papers without full text from abstracts
    if without_text:
        fallback = summarize_all_abstracts(without_text, chunk_size=20, provider=provider)
        all_summaries.extend(fallback)

    return all_summaries


# ── Ranking ──────────────────────────────────────────────────

def rank_and_select(
    summaries: list[PaperSummary],
    top_n: int = 15,
    provider: str = "groq",
) -> list[PaperSummary]:
    """Rank all summaries and select the top N."""
    if len(summaries) <= top_n:
        print(f"Only {len(summaries)} papers — returning all without ranking.")
        for i, s in enumerate(summaries):
            s.rank = i + 1
            s.rank_rationale = "Included (fewer papers than slots)."
        return summaries

    model = _model_for(provider, "rank")

    summaries_text = "\n\n".join(
        f"[{s.arxiv_id}] {s.title}\n"
        f"Tags: {', '.join(s.relevance_tags)}\n"
        f"Summary: {s.summary}\n"
        f"Contributions: {'; '.join(s.contributions)}"
        for s in summaries
    )

    tqdm.write(f"Ranking {len(summaries)} papers to select top {top_n} (provider: {provider})...")

    text = _call_llm(
        system=_rank_system(top_n),
        user=(
            f"Here are {len(summaries)} papers from today's arxiv submissions. "
            f"Select the top {top_n}.\n\n{summaries_text}"
        ),
        model=model,
        provider=provider,
        max_tokens=4096,
    )

    try:
        rankings = _parse_json_list(text)
        rankings = [r for r in rankings if isinstance(r, dict)]
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error in ranking: {e}")
        print(f"  Falling back to first {top_n} papers.")
        for i, s in enumerate(summaries[:top_n]):
            s.rank = i + 1
        return summaries[:top_n]

    # Merge rankings back into summaries
    summary_lookup = {s.arxiv_id: s for s in summaries}
    selected = []
    for r in rankings:
        aid = r.get("arxiv_id", "")
        if aid in summary_lookup:
            s = summary_lookup[aid]
            s.rank = r.get("rank", 99)
            s.rank_rationale = r.get("rationale", "")
            selected.append(s)

    selected.sort(key=lambda s: s.rank)
    print(f"  Selected {len(selected)} papers.")
    return selected[:top_n]


if __name__ == "__main__":
    # Quick test with dummy data
    test_papers = [{
        "arxiv_id": "2401.00001",
        "title": "Test Paper on Transformers",
        "authors": ["Alice", "Bob"],
        "categories": ["cs.LG"],
        "link": "https://arxiv.org/abs/2401.00001",
        "abstract": "We propose a new attention mechanism that reduces compute by 50%.",
    }]
    results = cull_abstracts_tournament(test_papers, survivors_per_batch=1, batch_size=10)
    print(f"Culled to {len(results)} papers:")
    for p in results:
        print(f"  [{p['arxiv_id']}] {p['title']}")
