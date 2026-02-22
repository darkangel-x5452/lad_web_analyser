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


# =============================================================================
#  STEP 2 — Scrape full content from each URL
# =============================================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}

def scrape_page(url: str, timeout: int = 10) -> str:
    """Fetch a URL and extract clean text using BeautifulSoup."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "form", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text[:8000]          # cap at ~8 k chars per page
    except Exception as exc:
        warn(f"Could not scrape {url}: {exc}")
        return ""


def scrape_all(results: list[dict], max_pages: int = 5) -> list[dict]:
    """Scrape text from the top N results."""
    step(f"Scraping top {max_pages} pages for full content …")
    enriched = []
    for r in results[:max_pages]:
        info(f"Fetching: {r['url']}")
        text = scrape_page(r["url"])
        if text:
            r["full_text"] = text
            enriched.append(r)
            info(f"  → {len(text):,} chars extracted")
        time.sleep(0.5)            # polite crawl delay
    return enriched


# =============================================================================
#  STEP 3a — Extractive summarisation (local, no API key needed)
# =============================================================================

def extractive_summary(text: str, sentences: int = 6, lang: str = "english") -> str:
    """
    Uses LSA (Latent Semantic Analysis) via the `sumy` library to pick the
    most information-dense sentences from a block of text.
    No internet connection or API key required.
    """
    if not SUMY_AVAILABLE:
        # Fallback: return first N sentences naively
        sents = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sents[:sentences])

    parser   = PlaintextParser.from_string(text, Tokenizer(lang))
    stemmer  = Stemmer(lang)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(lang)
    summary_sents = summarizer(parser.document, sentences)
    return " ".join(str(s) for s in summary_sents)


# =============================================================================
#  STEP 3b — LLM summarisation via Groq (free tier, no credit card)
# =============================================================================

def groq_summary(combined_text: str, query: str, api_key: str) -> str:
    """
    Sends scraped content to Groq's free inference API (llama-3 model).
    Free tier: https://console.groq.com — no credit card required.
    """
    if not GROQ_AVAILABLE:
        error("groq package not installed. Run: pip install groq")
        return ""

    client = Groq(api_key=api_key)
    prompt = (
        f"You are a research assistant. Answer the following question using ONLY "
        f"the provided web content. Be specific, cite numbers and facts, and note "
        f"the date of any statistics.\n\n"
        f"QUESTION: {query}\n\n"
        f"WEB CONTENT:\n{combined_text[:12000]}\n\n"
        f"Answer:"
    )
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        return chat.choices[0].message.content.strip()
    except Exception as exc:
        error(f"Groq API error: {exc}")
        return ""


# =============================================================================
#  STEP 4 — Assemble & present the final answer
# =============================================================================

def build_answer(pages: list[dict], query: str, groq_key: str | None) -> str:
    """Combine scraped pages and produce a coherent answer."""
    step("Generating answer from scraped content …")

    combined = "\n\n".join(
        f"[Source: {p['url']}]\n{p['full_text']}" for p in pages
    )

    if groq_key:
        info("Using Groq LLM for AI-powered answer …")
        answer = groq_summary(combined, query, groq_key)
        if answer:
            return answer

    # Fallback: per-page extractive summary
    info("Using local extractive summarisation (sumy / LSA) …")
    parts = []
    for p in pages:
        summary = extractive_summary(p["full_text"], sentences=4)
        parts.append(f"• {p['title']}\n  {summary}")

    return "\n\n".join(parts)


# =============================================================================
#  Main orchestrator
# =============================================================================

def explain_how_llms_search():
    """Print an educational overview of how LLMs do web search."""
    header("HOW LLMs / GenAI TOOLS SEARCH THE WEB")
    print(textwrap.dedent(f"""
  {BOLD}1. Query Reformulation{RESET}
     The raw user prompt is rewritten into a concise, keyword-rich search
     query (e.g. "Nebraska Omaha NCAA Basketball stats 2024-25").

  {BOLD}2. Search Engine API Call{RESET}
     The reformulated query hits a search engine:
     • Google Search API / Programmable Search Engine (paid)
     • Bing Search API (Azure, freemium)
     • DuckDuckGo (free, no key) ← what this app uses
     • SerpAPI, Brave Search API, Tavily AI, etc.

  {BOLD}3. Link Ranking & Retrieval{RESET}
     The search engine returns ranked results (title, URL, snippet) using:
     • PageRank / authority scoring
     • Keyword relevance (BM25 / TF-IDF)
     • Recency boost for time-sensitive queries

  {BOLD}4. Web Page Scraping{RESET}
     The top-N URLs are fetched and parsed (HTML → clean text) via:
     • requests + BeautifulSoup (Python, open-source)
     • Playwright / Puppeteer (for JS-heavy pages)
     • Dedicated services: Firecrawl, Diffbot, Apify

  {BOLD}5. Retrieval-Augmented Generation (RAG){RESET}
     Scraped text is injected into the LLM's context window as "grounding
     documents". The LLM is instructed to answer ONLY from that content,
     avoiding hallucination and including up-to-date facts.

  {BOLD}6. Citation & Source Attribution{RESET}
     The answer is returned alongside the source URLs so users can verify.

  This app replicates steps 1-6 using 100% free, open-source tools.
"""))


def run(query: str, groq_key: str | None, max_results: int, max_pages: int):
    header(f"AI WEB SEARCHER  ·  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"\n  {BOLD}Query:{RESET} {query}\n")

    # 1. Search
    results = search_web(query, max_results=max_results)
    if not results:
        error("No search results returned. Check your internet connection.")
        sys.exit(1)

    # 2. Scrape
    pages = scrape_all(results, max_pages=max_pages)
    if not pages:
        error("Could not scrape any pages.")
        sys.exit(1)

    # 3. Summarise
    answer = build_answer(pages, query, groq_key)

    # 4. Present
    header("ANSWER")
    print()
    for line in answer.split("\n"):
        print(f"  {line}")

    header("SOURCES USED")
    for i, p in enumerate(pages, 1):
        print(f"  {BOLD}[{i}]{RESET} {p['title']}")
        print(f"       {CYAN}{p['url']}{RESET}\n")

    print(f"\n  {DIM}Scraped {len(pages)} pages · "
          f"{sum(len(p['full_text']) for p in pages):,} chars processed{RESET}\n")


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

    if args.explain:
        explain_how_llms_search()
        sys.exit(0)

    explain_how_llms_search()

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
