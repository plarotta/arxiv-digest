"""
fetch_arxiv.py — Fetch new ML papers from arxiv RSS feeds.

Pulls from multiple category feeds, deduplicates cross-listed papers,
and returns a list of paper metadata dicts.
"""

import feedparser
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


CATEGORIES = [
    "cs.LG",   # Machine Learning
    "cs.CL",   # Computation and Language
    "cs.AI",   # Artificial Intelligence
    "cs.CV",   # Computer Vision
    "cs.RO",   # Robotics
]

RSS_BASE = "https://rss.arxiv.org/rss/{category}"


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    link: str
    primary_category: str
    published: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_arxiv_id(link: str) -> str:
    """Extract arxiv ID from link like http://arxiv.org/abs/2401.12345"""
    match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", link)
    return match.group(1) if match else link


def _clean_text(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_authors(entry) -> list[str]:
    """Extract author names from RSS entry."""
    if hasattr(entry, "authors"):
        return [a.get("name", "") for a in entry.authors if a.get("name")]
    # Fallback: parse from author field
    author_str = getattr(entry, "author", "")
    if author_str:
        return [a.strip() for a in author_str.split(",")]
    return ["Unknown"]


def _parse_categories(entry) -> list[str]:
    """Extract categories/tags from RSS entry."""
    if hasattr(entry, "tags"):
        return [t.get("term", "") for t in entry.tags if t.get("term")]
    return []


def fetch_category(category: str) -> list[ArxivPaper]:
    """Fetch new papers from a single arxiv category RSS feed."""
    url = RSS_BASE.format(category=category)
    print(f"  Fetching {url} ...")
    feed = feedparser.parse(url)

    papers = []
    for entry in feed.entries:
        # Skip "replacements" — we only want new submissions
        title = _clean_text(entry.get("title", ""))
        if title.lower().startswith("replaced:") or "(replaced)" in title.lower():
            continue

        # Some RSS feeds prefix with category tags like "cs.LG: Title"
        # The arxiv RSS uses <title> directly
        link = entry.get("link", "")
        arxiv_id = _extract_arxiv_id(link)
        abstract = _clean_text(entry.get("summary", entry.get("description", "")))
        categories = _parse_categories(entry)
        authors = _parse_authors(entry)

        paper = ArxivPaper(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            abstract=abstract,
            categories=categories if categories else [category],
            link=f"https://arxiv.org/abs/{arxiv_id}",
            primary_category=category,
            published=entry.get("published", entry.get("updated", "")),
        )
        papers.append(paper)

    print(f"    Found {len(papers)} new papers in {category}")
    return papers


def fetch_all(categories: list[str] = None) -> list[ArxivPaper]:
    """
    Fetch new papers from all configured categories.
    Deduplicates cross-listed papers (keeps first occurrence, merges categories).
    """
    categories = categories or CATEGORIES
    seen: dict[str, ArxivPaper] = {}

    for cat in categories:
        papers = fetch_category(cat)
        for paper in papers:
            if paper.arxiv_id in seen:
                # Merge categories for cross-listed papers
                existing = seen[paper.arxiv_id]
                for c in paper.categories:
                    if c not in existing.categories:
                        existing.categories.append(c)
            else:
                seen[paper.arxiv_id] = paper
        time.sleep(1)  # Be polite to arxiv servers

    all_papers = list(seen.values())
    print(f"\nTotal unique papers: {len(all_papers)}")
    return all_papers


if __name__ == "__main__":
    papers = fetch_all()
    for p in papers[:5]:
        print(f"\n[{p.arxiv_id}] {p.title}")
        print(f"  Categories: {', '.join(p.categories)}")
        print(f"  Abstract: {p.abstract[:150]}...")
