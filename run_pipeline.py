#!/usr/bin/env python3
"""
run_pipeline.py — Full arxiv ML digest pipeline.

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --output site/     # Custom output directory
    python run_pipeline.py --top 10           # Select top 10 instead of 15
    python run_pipeline.py --categories cs.LG cs.CL  # Override categories
    python run_pipeline.py --dry-run          # Fetch only, no API calls
    python run_pipeline.py --rank-provider gemini  # Use Gemini Pro for final ranking

Requires:
    GROQ_API_KEY environment variable set.
    Optionally GEMINI_API_KEY for --rank-provider gemini.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fetch_arxiv import fetch_all, CATEGORIES
from summarize_and_rank import (
    cull_abstracts_tournament,
    summarize_all_abstracts,
    summarize_with_full_text,
    rank_and_select,
)
from generate_site import generate_html
from pdf_extraction import download_and_extract


def save_json(data: list[dict], path: str):
    """Save intermediate data as JSON for debugging / caching."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="ML Digest — daily arxiv paper rankings")
    parser.add_argument("--output", default="site", help="Output directory for HTML")
    parser.add_argument("--top", type=int, default=15, help="Number of top papers to select")
    parser.add_argument("--categories", nargs="+", default=None, help="arxiv categories to fetch")
    parser.add_argument("--shortlist", type=int, default=60, help="Number of candidates to shortlist via tournament cull")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF download; summarize from abstracts only")
    parser.add_argument("--dry-run", action="store_true", help="Fetch papers only, skip API calls")
    parser.add_argument("--from-cache", type=str, default=None, help="Load papers from a cached JSON file instead of fetching RSS")
    parser.add_argument("--cache-dir", default=".cache", help="Directory for intermediate JSON files")
    parser.add_argument("--rank-provider", choices=["groq", "gemini"], default="groq",
                        help="LLM provider for the final ranking step (default: groq)")
    args = parser.parse_args()

    categories = args.categories or CATEGORIES
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"  ML Digest Pipeline — {date_str}")
    print(f"  Categories: {', '.join(categories)}")
    print(f"  Top N: {args.top}")
    print(f"  Rank provider: {args.rank_provider}")
    print("=" * 60)

    # ── Step 1: Fetch papers ──────────────────────────────────
    if args.from_cache:
        print(f"\n[1/6] Loading papers from cache: {args.from_cache}")
        with open(args.from_cache) as f:
            papers_dicts = json.load(f)
        print(f"  Loaded {len(papers_dicts)} papers from cache.")
    else:
        print("\n[1/6] Fetching new papers from arxiv RSS...")
        papers = fetch_all(categories)

        if not papers:
            print("No papers found. arxiv RSS may not have updated yet (try after ~20:00 UTC).")
            sys.exit(0)

        papers_dicts = [p.to_dict() for p in papers]
        save_json(papers_dicts, f"{args.cache_dir}/{date_str}_fetched.json")

    if args.dry_run:
        print(f"\n[DRY RUN] Fetched {len(papers_dicts)} papers. Skipping LLM calls.")
        sys.exit(0)

    # ── Env var checks ────────────────────────────────────────
    if "GROQ_API_KEY" not in os.environ:
        print("ERROR: GROQ_API_KEY not set. Export it and retry.")
        sys.exit(1)
    if args.rank_provider == "gemini" and "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: --rank-provider gemini requires GEMINI_API_KEY or GOOGLE_API_KEY.")
        sys.exit(1)

    # ── Step 2: Tournament cull ───────────────────────────────
    shortlist_n = max(args.shortlist, args.top)
    # Uses the smaller 8B model for culling — batch size kept small to stay
    # under Groq free tier's 6k TPM limit (~400 tokens per paper)
    batch_size = 5
    num_batches = max(1, (len(papers_dicts) + batch_size - 1) // batch_size)
    survivors_per_batch = max(1, round(shortlist_n / num_batches))

    print(f"\n[2/6] Tournament cull: {len(papers_dicts)} papers → ~{shortlist_n} candidates "
          f"({num_batches} batches of ~{batch_size}, {survivors_per_batch} survivors each)...")
    culled = cull_abstracts_tournament(
        papers_dicts,
        survivors_per_batch=survivors_per_batch,
        batch_size=batch_size,
    )
    save_json(culled, f"{args.cache_dir}/{date_str}_culled.json")

    if args.skip_pdf:
        # ── Abstract-only mode ────────────────────────────────
        print(f"\n[3/6] Summarizing {len(culled)} papers from abstracts (skip-pdf mode)...")
        summaries = summarize_all_abstracts(culled)
        summaries_dicts = [s.to_dict() for s in summaries]
        save_json(summaries_dicts, f"{args.cache_dir}/{date_str}_summaries.json")

        print(f"\n[4/6] Ranking and selecting top {args.top} papers...")
        selected = rank_and_select(summaries, top_n=args.top, provider=args.rank_provider)
        step_label = "[5/6]"
    else:
        # ── Step 3: Download PDFs ─────────────────────────────
        print(f"\n[3/6] Downloading PDFs for {len(culled)} culled papers...")
        arxiv_ids = [p["arxiv_id"] for p in culled]
        full_texts = download_and_extract(arxiv_ids, cache_dir=f"{args.cache_dir}/pdfs")

        # ── Step 4: Summarize with full text ──────────────────
        print(f"\n[4/6] Summarizing with full paper text...")
        summaries = summarize_with_full_text(culled, full_texts, chunk_size=1)
        summaries_dicts = [s.to_dict() for s in summaries]
        save_json(summaries_dicts, f"{args.cache_dir}/{date_str}_fulltext_summaries.json")

        # ── Step 5: Final ranking ─────────────────────────────
        print(f"\n[5/6] Final ranking — selecting top {args.top} papers...")
        selected = rank_and_select(summaries, top_n=args.top, provider=args.rank_provider)
        step_label = "[6/6]"

    selected_dicts = [s.to_dict() for s in selected]
    save_json(selected_dicts, f"{args.cache_dir}/{date_str}_selected.json")

    # ── Step 6: Generate site ─────────────────────────────────
    print(f"\n{step_label} Generating static site...")
    output_path = os.path.join(args.output, "index.html")
    generate_html(
        papers=selected_dicts,
        total_fetched=len(papers_dicts),
        categories=categories,
        output_path=output_path,
    )

    # Also save the current digest as a dated file for archives
    archive_path = os.path.join(args.output, "archive", f"{date_str}.html")
    generate_html(
        papers=selected_dicts,
        total_fetched=len(papers_dicts),
        categories=categories,
        output_path=archive_path,
    )

    print("\n" + "=" * 60)
    print(f"  Done! {len(selected)} papers selected from {len(papers_dicts)} total.")
    print(f"  Site:    {output_path}")
    print(f"  Archive: {archive_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
