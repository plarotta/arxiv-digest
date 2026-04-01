"""
pdf_extraction.py — Download arxiv PDFs and extract text for full-paper summarization.

Downloads PDFs with rate limiting (arxiv requires ≥3s between requests),
extracts text via PyMuPDF, and truncates to a reasonable length for the LLM.
"""

import os
import re
import time
import urllib.request
import fitz  # pymupdf
from tqdm import tqdm


ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
DOWNLOAD_DELAY = 4  # seconds between downloads (arxiv asks for ≥3s)
MAX_CHARS = 25_000  # ~6-7k tokens — covers intro, methods, results for most papers
USER_AGENT = "arxiv-digest-bot/1.0 (daily ML digest; mailto:noreply@example.com)"


def download_pdf(arxiv_id: str, cache_dir: str = ".cache/pdfs") -> str | None:
    """Download a PDF from arxiv. Returns path on success, None on failure.
    Skips download if the file is already cached."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{arxiv_id}.pdf")

    if os.path.exists(path):
        return path

    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(path, "wb") as f:
                f.write(resp.read())
        return path
    except Exception as e:
        print(f"  WARNING: Failed to download {arxiv_id}: {e}")
        return None


def extract_text(pdf_path: str) -> str | None:
    """Extract text from a PDF using PyMuPDF, excluding images and tables.
    Returns None if extraction fails or yields negligible text."""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            text_parts = []
            for block in blocks:
                # Skip image blocks
                if block["type"] == 1:
                    continue
                # Extract text from text blocks
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(s["text"] for s in spans).strip()
                    if line_text:
                        text_parts.append(line_text)
            pages.append("\n".join(text_parts))
        doc.close()
        text = "\n".join(pages).strip()
        text = _strip_tables_and_captions(text)
        if len(text) < 200:
            return None
        return text
    except Exception as e:
        print(f"  WARNING: Failed to extract text from {pdf_path}: {e}")
        return None


def _strip_tables_and_captions(text: str) -> str:
    """Remove table content and figure/table captions from extracted text."""
    lines = text.split("\n")
    filtered = []
    for line in lines:
        # Skip figure/table captions (e.g. "Figure 1:", "Table 3.", "Fig. 2:")
        if re.match(r"^\s*(Figure|Fig\.|Table)\s+\d", line, re.IGNORECASE):
            continue
        # Skip lines that look like table rows: mostly numbers, pipes, or
        # short fragments separated by lots of whitespace
        stripped = line.strip()
        if stripped and len(stripped) > 1:
            non_digit_space = re.sub(r"[\d\s\.\,\-\+\%\|±]", "", stripped)
            if len(non_digit_space) < len(stripped) * 0.3 and len(stripped) > 10:
                continue
        filtered.append(line)
    return "\n".join(filtered)


def _strip_references(text: str) -> str:
    """Remove the references/bibliography section from the end of a paper."""
    # Match common section headers for references, case-insensitive.
    # Look for a line that starts the references section (e.g. "References\n",
    # "Bibliography\n", "REFERENCES\n") and drop everything after it.
    match = re.search(
        r"\n\s*(References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n",
        text,
    )
    if match:
        return text[:match.start()].rstrip()
    return text


def truncate_text(text: str, max_chars: int = MAX_CHARS) -> str:
    """Strip references, then truncate to max_chars at a paragraph boundary."""
    text = _strip_references(text)
    if len(text) <= max_chars:
        return text
    # Find the last double-newline before the limit to break at a paragraph
    truncated = text[:max_chars]
    last_break = truncated.rfind("\n\n")
    if last_break > max_chars * 0.8:
        truncated = truncated[:last_break]
    return truncated + "\n\n[...truncated]"


def download_and_extract(
    arxiv_ids: list[str],
    cache_dir: str = ".cache/pdfs",
) -> dict[str, str]:
    """Download PDFs and extract text for a list of papers.
    Returns a dict of arxiv_id -> truncated full text.
    Papers that fail download or extraction are silently skipped."""
    results = {}
    pbar = tqdm(arxiv_ids, desc="Downloading PDFs", unit="paper")
    for i, aid in enumerate(pbar):
        pbar.set_postfix(paper=aid)
        path = download_pdf(aid, cache_dir)
        if path:
            text = extract_text(path)
            if text:
                results[aid] = truncate_text(text)
            else:
                tqdm.write(f"  Skipped {aid} (no extractable text)")
        if i < len(arxiv_ids) - 1:
            time.sleep(DOWNLOAD_DELAY)

    return results


if __name__ == "__main__":
    # Quick test with a known paper
    import sys
    aid = sys.argv[1] if len(sys.argv) > 1 else "2401.00001"
    texts = download_and_extract([aid])
    for arxiv_id, text in texts.items():
        print(f"\n[{arxiv_id}] ({len(text)} chars)")
        print(text[:500])
