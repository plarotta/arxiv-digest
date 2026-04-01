"""
generate_site.py — Generate a static HTML page from ranked paper summaries.

Produces a single self-contained HTML file suitable for GitHub Pages or S3.
"""

import json
import os
from datetime import datetime, timezone
from jinja2 import Template


HTML_TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Digest — {{ date_display }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0c0c0e;
    --surface: #141418;
    --surface-hover: #1a1a20;
    --border: #232329;
    --border-accent: #2d2d36;
    --text: #e8e6e3;
    --text-secondary: #8a8891;
    --text-dim: #5c5a63;
    --accent: #c4a7ff;
    --accent-muted: #7c5cbf;
    --tag-bg: #1e1b2e;
    --tag-text: #b39ddb;
    --rank-gold: #f0c674;
    --rank-silver: #a8b5c8;
    --rank-bronze: #c49a6c;
    --rank-default: #5c5a63;
    --cat-lg: #6ec6ff;
    --cat-cl: #aed581;
    --cat-ai: #ffab91;
    --cat-cv: #ce93d8;
    --cat-ro: #80deea;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }

  .noise-overlay {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9999;
    opacity: 0.025;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }

  header {
    padding: 3rem 2rem 2rem;
    max-width: 900px;
    margin: 0 auto;
    border-bottom: 1px solid var(--border);
  }

  header h1 {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 2.4rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    color: var(--text);
    margin-bottom: 0.4rem;
  }

  header h1 span.accent { color: var(--accent); }

  .subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-dim);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .meta-bar {
    display: flex;
    gap: 1.5rem;
    margin-top: 1rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-family: 'IBM Plex Mono', monospace;
  }

  .meta-bar .stat { display: flex; align-items: center; gap: 0.35rem; }
  .meta-bar .dot { color: var(--accent-muted); }

  main {
    max-width: 900px;
    margin: 0 auto;
    padding: 1.5rem 2rem 4rem;
  }

  .paper-card {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
    background: var(--surface);
    transition: border-color 0.2s, background 0.2s;
    position: relative;
  }

  .paper-card:hover {
    border-color: var(--border-accent);
    background: var(--surface-hover);
  }

  .card-header {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    margin-bottom: 0.75rem;
  }

  .rank-badge {
    flex-shrink: 0;
    width: 2.2rem;
    height: 2.2rem;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    border: 1px solid;
  }

  .rank-1 { color: var(--rank-gold); border-color: var(--rank-gold); background: rgba(240,198,116,0.08); }
  .rank-2 { color: var(--rank-silver); border-color: var(--rank-silver); background: rgba(168,181,200,0.06); }
  .rank-3 { color: var(--rank-bronze); border-color: var(--rank-bronze); background: rgba(196,154,108,0.06); }
  .rank-other { color: var(--rank-default); border-color: var(--border); background: transparent; }

  .card-title {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 400;
    line-height: 1.35;
    color: var(--text);
  }

  .card-title a {
    color: inherit;
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 0.2s;
  }

  .card-title a:hover { border-bottom-color: var(--accent); }

  .card-authors {
    font-size: 0.78rem;
    color: var(--text-dim);
    margin: 0.25rem 0 0.6rem;
    padding-left: 3.2rem;
  }

  .card-summary {
    font-size: 0.9rem;
    line-height: 1.65;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
    padding-left: 3.2rem;
  }

  .contributions {
    padding-left: 3.2rem;
    margin-bottom: 0.75rem;
  }

  .contribution {
    font-size: 0.82rem;
    color: var(--text-secondary);
    padding: 0.3rem 0 0.3rem 1rem;
    border-left: 2px solid var(--border-accent);
    margin-bottom: 0.3rem;
  }

  .card-footer {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    padding-left: 3.2rem;
    align-items: center;
  }

  .tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    background: var(--tag-bg);
    color: var(--tag-text);
    letter-spacing: 0.02em;
  }

  .cat-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-weight: 500;
  }

  .cat-cs-LG { background: rgba(110,198,255,0.1); color: var(--cat-lg); }
  .cat-cs-CL { background: rgba(174,213,129,0.1); color: var(--cat-cl); }
  .cat-cs-AI { background: rgba(255,171,145,0.1); color: var(--cat-ai); }
  .cat-cs-CV { background: rgba(206,147,216,0.1); color: var(--cat-cv); }
  .cat-cs-RO { background: rgba(128,222,234,0.1); color: var(--cat-ro); }

  .rank-rationale {
    font-size: 0.78rem;
    color: var(--text-dim);
    font-style: italic;
    padding-left: 3.2rem;
    margin-bottom: 0.5rem;
  }

  footer {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
    border-top: 1px solid var(--border);
    font-size: 0.75rem;
    color: var(--text-dim);
    font-family: 'IBM Plex Mono', monospace;
    text-align: center;
  }

  @media (max-width: 640px) {
    header { padding: 2rem 1.2rem 1.5rem; }
    main { padding: 1rem 1.2rem 3rem; }
    header h1 { font-size: 1.8rem; }
    .card-authors, .card-summary, .contributions, .card-footer, .rank-rationale {
      padding-left: 0;
    }
    .card-header { flex-direction: column; gap: 0.5rem; }
    .rank-badge { width: 1.8rem; height: 1.8rem; font-size: 0.75rem; }
  }
</style>
</head>
<body>
<div class="noise-overlay"></div>

<header>
  <h1>ML <span class="accent">Digest</span></h1>
  <p class="subtitle">Top papers from arxiv &middot; {{ date_display }}</p>
  <div class="meta-bar">
    <div class="stat"><span class="dot">&bull;</span> {{ total_fetched }} papers scanned</div>
    <div class="stat"><span class="dot">&bull;</span> {{ papers | length }} selected</div>
    <div class="stat"><span class="dot">&bull;</span> {{ categories | join(', ') }}</div>
  </div>
</header>

<main>
  {% for paper in papers %}
  <article class="paper-card">
    <div class="card-header">
      <div class="rank-badge {% if paper.rank == 1 %}rank-1{% elif paper.rank == 2 %}rank-2{% elif paper.rank == 3 %}rank-3{% else %}rank-other{% endif %}">
        {{ paper.rank }}
      </div>
      <div class="card-title">
        <a href="{{ paper.link }}" target="_blank" rel="noopener">{{ paper.title }}</a>
      </div>
    </div>

    <div class="card-authors">{{ paper.authors[:5] | join(', ') }}{% if paper.authors | length > 5 %} et al.{% endif %}</div>

    <div class="card-summary">{{ paper.summary }}</div>

    {% if paper.contributions %}
    <div class="contributions">
      {% for c in paper.contributions %}
      <div class="contribution">{{ c }}</div>
      {% endfor %}
    </div>
    {% endif %}

    {% if paper.rank_rationale %}
    <div class="rank-rationale">&ldquo;{{ paper.rank_rationale }}&rdquo;</div>
    {% endif %}

    <div class="card-footer">
      {% for cat in paper.categories %}
      <span class="cat-tag cat-{{ cat | replace('.', '-') }}">{{ cat }}</span>
      {% endfor %}
      {% for tag in paper.relevance_tags %}
      <span class="tag">{{ tag }}</span>
      {% endfor %}
    </div>
  </article>
  {% endfor %}
</main>

<footer>
  Generated by ML Digest &middot; Summaries by Llama 3.3 &middot; {{ generated_at }}
</footer>

</body>
</html>""")


def generate_html(
    papers: list[dict],
    total_fetched: int,
    categories: list[str],
    output_path: str = "site/index.html",
) -> str:
    """Render the static HTML digest page."""

    now = datetime.now(timezone.utc)
    date_display = now.strftime("%B %d, %Y")
    generated_at = now.strftime("%Y-%m-%d %H:%M UTC")

    html = HTML_TEMPLATE.render(
        papers=papers,
        total_fetched=total_fetched,
        categories=categories,
        date_display=date_display,
        generated_at=generated_at,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Site generated: {output_path}")
    return output_path


if __name__ == "__main__":
    # Generate a preview with dummy data
    dummy_papers = [
        {
            "rank": i + 1,
            "title": f"Sample Paper #{i+1}: A Novel Approach to Something",
            "authors": ["Alice", "Bob", "Carol"],
            "categories": ["cs.LG", "cs.AI"],
            "link": f"https://arxiv.org/abs/2401.{10000+i}",
            "summary": "This paper introduces a new method that achieves state-of-the-art results on common benchmarks while requiring significantly less compute.",
            "contributions": [
                "A new architecture component that reduces FLOPs by 40%",
                "SOTA results on ImageNet, COCO, and ADE20K",
            ],
            "relevance_tags": ["architectures", "efficiency"],
            "rank_rationale": "Novel architecture with strong empirical validation.",
        }
        for i in range(15)
    ]
    generate_html(
        papers=dummy_papers,
        total_fetched=287,
        categories=["cs.LG", "cs.CL", "cs.AI", "cs.CV", "cs.RO"],
        output_path="site/index.html",
    )
