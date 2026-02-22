#!/usr/bin/env python3
"""
=============================================================================
  AI Web Searcher — How LLMs Find & Summarize Up-to-Date Information
=============================================================================
  Uses ONLY free tools — NO API key or credit card required by default.

  Stack:
    • DuckDuckGo Search  → duckduckgo_search  (free, no key)
    • Web Scraping       → requests + BeautifulSoup4
    • Summarization      → sumy (extractive, local) — no API key needed
    • Optional LLM boost → Groq API (free tier, no credit card required)
                           Sign up at: https://console.groq.com

  Install dependencies:
    pip install duckduckgo_search requests beautifulsoup4 sumy nltk groq

  Usage:
    python ai_web_searcher.py
    python ai_web_searcher.py --query "Nebraska Omaha NCAA Basketball stats"
    python ai_web_searcher.py --groq-key YOUR_KEY   # optional LLM summarizer
=============================================================================
"""

import sys
import re
import time
import argparse
import textwrap
from datetime import datetime
from urllib.parse import urlparse

# ── Dependency check ─────────────────────────────────────────────────────────
MISSING = []
try:
    from ddgs import DDGS
except ImportError:
    MISSING.append("duckduckgo_search")

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    MISSING.append("requests beautifulsoup4")

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

if MISSING:
    print("❌  Missing dependencies. Run:")
    print(f"    pip install {' '.join(MISSING)} sumy nltk groq")
    sys.exit(1)

# ── ANSI colour helpers ───────────────────────────────────────────────────────
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
DIM   = "\033[2m"
RESET = "\033[0m"

def header(text):  print(f"\n{BOLD}{CYAN}{'─'*70}\n  {text}\n{'─'*70}{RESET}")
def step(text):    print(f"\n{BOLD}{GREEN}▶ {text}{RESET}")
def info(text):    print(f"  {DIM}{text}{RESET}")
def warn(text):    print(f"  {YELLOW}⚠  {text}{RESET}")
def error(text):   print(f"  {RED}✗ {text}{RESET}")
def result(text):  print(f"  {text}")


# =============================================================================
#  STEP 1 — Search DuckDuckGo for relevant links
# =============================================================================

def search_web(query: str, max_results: int = 8) -> list[dict]:
    """
    Uses DuckDuckGo (via duckduckgo_search) to find recent, relevant URLs.

    How it works under the hood (same as major LLM tools):
      1. The query is tokenised and sent to DDG's search endpoint.
      2. DDG ranks results by a mix of PageRank-style authority, recency,
         and keyword relevance.
      3. We receive a list of {title, url, body} snippets — no scraping needed
         for the initial link list.
    """
    step("Searching the web with DuckDuckGo …")
    info(f'Query: "{query}"')

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href",  ""),
                    "snippet": r.get("body",  ""),
                })
    except Exception as exc:
        error(f"DuckDuckGo search failed: {exc}")
        return []

    print()
    for i, r in enumerate(results, 1):
        domain = urlparse(r["url"]).netloc
        print(f"  {BOLD}[{i}]{RESET} {r['title']}")
        print(f"       {CYAN}{r['url']}{RESET}")
        snippet = textwrap.shorten(r["snippet"], 100)
        print(f"       {DIM}{snippet}{RESET}\n")

    return results



def run(query: str, groq_key: str | None, max_results: int, max_pages: int):
    header(f"AI WEB SEARCHER  ·  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"\n  {BOLD}Query:{RESET} {query}\n")

    # 1. Search
    results = search_web(query, max_results=max_results)
    if not results:
        error("No search results returned. Check your internet connection.")
        sys.exit(1)

# =============================================================================
#  CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI Web Searcher — free, no API key required by default",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
Examples:
  python ai_web_searcher.py
  python ai_web_searcher.py --query "Nebraska Omaha NCAA Basketball team stats 2025"
  python ai_web_searcher.py --query "latest SpaceX launch" --max-results 10
  python ai_web_searcher.py --query "..." --groq-key gsk_xxxx
        """)
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Search query / question (prompted interactively if omitted)"
    )
    parser.add_argument(
        "--groq-key",
        type=str,
        default=None,
        metavar="KEY",
        help="Optional Groq API key for LLM-powered answers (free at console.groq.com)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=8,
        help="Number of search results to retrieve (default: 8)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Number of pages to fully scrape (default: 5)"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show how LLMs search the web, then exit"
    )

    args = parser.parse_args()

    query = args.query
    if not query:
        print(f"\n{BOLD}Enter your question:{RESET} ", end="")
        query = input().strip()
        if not query:
            error("No query provided.")
            sys.exit(1)

    run(
        query=query,
        groq_key=args.groq_key,
        max_results=args.max_results,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()
