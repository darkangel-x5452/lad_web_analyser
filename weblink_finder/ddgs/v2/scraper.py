#!/usr/bin/env python3
"""
DeepLink Scraper
────────────────
Finds specific, deep subpage URLs that are directly relevant to your query
using DuckDuckGo (free, no API key) + BeautifulSoup for result enrichment.

Usage:
    python scraper.py
    python scraper.py --query "LA Lakers NBA statistics 2024" --max 15
    python scraper.py --query "..." --export results.json
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import urlparse

# ── Rich UI ──────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── DuckDuckGo Search ────────────────────────────────────────────────────────
try:
    from ddgs import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

# ── HTTP & Parsing ───────────────────────────────────────────────────────────
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredLink:
    url: str
    title: str
    snippet: str
    score: float
    depth: int          # URL path depth (higher = more specific)
    signals: list[str]  # Why this link is relevant
    source: str = "ddg"

    def display_score(self) -> str:
        filled = int(self.score / 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"[{bar}] {self.score:.1f}/100"


# ─────────────────────────────────────────────────────────────────────────────
# Query Intelligence — decompose natural language into targeted searches
# ─────────────────────────────────────────────────────────────────────────────

GENERIC_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "youtube.com", "reddit.com", "wikipedia.org", "amazon.com",
    "ebay.com", "pinterest.com", "tiktok.com",
}

SHALLOW_PATHS = {"", "/", "/home", "/index", "/about", "/contact", "/news"}

SPORTS_SITE_PATTERNS = [
    r"espn\.com/.+/stats",
    r"espn\.com/.+/roster",
    r"espn\.com/.+/schedule",
    r"nba\.com/stats",
    r"basketball-reference\.com/teams",
    r"basketball-reference\.com/players",
    r"statmuse\.com/.+",
    r"sports-reference\.com/.+",
]


def extract_intent_keywords(query: str) -> dict:
    """
    Parse a natural-language query into structured signals.
    Extracts: entities, stats keywords, sport/league hints, year hints.
    """
    q = query.lower()

    # Year / season detection
    year_match = re.findall(r'\b(20\d{2}[-–]?\d{0,2})\b', query)
    years = year_match if year_match else []

    # Stat-type keywords
    stat_words = re.findall(
        r'\b(stats?|statistics?|standings?|roster|schedule|scores?|'
        r'results?|rankings?|leaders?|box.?score|per.game|average|'
        r'points|assists|rebounds|wins|losses|record|playoffs?|'
        r'salary|contract|trades?|draft|injury|preview|recap)\b', q)

    # Sport detection
    sports = re.findall(
        r'\b(basketball|football|soccer|baseball|hockey|tennis|'
        r'golf|mma|nfl|nba|mlb|nhl|ncaa|wnba|mls|f1|formula)\b', q)

    # Remove filler words to get clean entity tokens
    stopwords = {
        "get", "the", "for", "from", "of", "a", "an", "and", "or",
        "in", "on", "at", "to", "with", "show", "me", "find", "what",
        "are", "is", "team", "competition", "league", "sport", "please",
        "give", "list", "all", "their", "its", "can", "you"
    }
    tokens = [
        w for w in re.findall(r'\b[a-zA-Z0-9]+\b', query)
        if w.lower() not in stopwords and len(w) > 1
    ]

    return {
        "raw": query,
        "tokens": tokens,
        "stat_words": list(set(stat_words)),
        "sports": list(set(sports)),
        "years": years,
    }


def build_search_queries(intent: dict) -> list[str]:
    """
    Generate 4–6 highly targeted search queries from intent signals.
    Each query aims at a different angle (stats page, roster, schedule, etc.)
    """
    raw = intent["raw"]
    tokens = " ".join(intent["tokens"])
    stats = intent["stat_words"]
    years = intent["years"]
    year_str = years[0] if years else ""

    queries = []

    # 1. Direct query as-is (most natural)
    queries.append(raw)

    # 2. Token-only + most important stat keyword
    if stats:
        queries.append(f"{tokens} {stats[0]} {year_str}".strip())
    else:
        queries.append(f"{tokens} statistics {year_str}".strip())

    # 3. site-targeted variants for known authoritative domains
    authoritative = ["site:espn.com", "site:basketball-reference.com",
                     "site:statmuse.com", "site:nba.com",
                     "site:sports-reference.com"]
    for site in authoritative[:2]:  # limit to avoid too many queries
        queries.append(f"{tokens} {site}")

    # 4. Specific stat sub-types
    for stat in stats[:2]:
        queries.append(f"{tokens} {stat} {year_str} page".strip())

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:7]  # cap at 7 search passes


# ─────────────────────────────────────────────────────────────────────────────
# Relevance Scoring Engine
# ─────────────────────────────────────────────────────────────────────────────

def url_depth(url: str) -> int:
    """Count meaningful path segments in a URL."""
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    return len(parts)


def score_link(url: str, title: str, snippet: str, intent: dict) -> tuple[float, list[str]]:
    """
    Return (score 0–100, [reason strings]).
    Scoring factors:
      - URL depth (deeper = more specific = better)
      - Keyword hits in URL path
      - Keyword hits in title / snippet
      - Presence of stat-type words
      - Penalties for generic/shallow pages
    """
    signals = []
    score = 0.0

    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.lower()
    full_text = (title + " " + snippet + " " + path).lower()
    tokens_lower = [t.lower() for t in intent["tokens"]]
    stat_words = intent["stat_words"]

    # ── Penalty: generic / social domains ──────────────────────────────────
    if domain in GENERIC_DOMAINS:
        return 0.0, ["❌ Generic/social domain — skipped"]

    # ── Penalty: shallow paths ──────────────────────────────────────────────
    depth = url_depth(url)
    if path in SHALLOW_PATHS or depth == 0:
        score -= 30
        signals.append(f"⚠ Homepage / shallow path (depth={depth})")
    elif depth == 1:
        score += 5
    elif depth == 2:
        score += 15
        signals.append(f"✓ Specific subpage (depth={depth})")
    elif depth >= 3:
        score += 25
        signals.append(f"✓✓ Deep subpage (depth={depth})")

    # ── Bonus: URL path contains query tokens ──────────────────────────────
    token_hits_in_path = sum(1 for t in tokens_lower if t in path)
    if token_hits_in_path:
        bonus = min(token_hits_in_path * 8, 24)
        score += bonus
        signals.append(f"✓ {token_hits_in_path} query keyword(s) in URL path (+{bonus:.0f})")

    # ── Bonus: title contains query tokens ─────────────────────────────────
    token_hits_title = sum(1 for t in tokens_lower if t in title.lower())
    if token_hits_title:
        bonus = min(token_hits_title * 6, 20)
        score += bonus
        signals.append(f"✓ {token_hits_title} keyword(s) in title (+{bonus:.0f})")

    # ── Bonus: stat/data keywords ──────────────────────────────────────────
    stat_hits = sum(1 for s in stat_words if s in full_text)
    if stat_hits:
        bonus = min(stat_hits * 7, 21)
        score += bonus
        signals.append(f"✓ {stat_hits} stat keyword(s) matched (+{bonus:.0f})")

    # ── Bonus: snippet contains query tokens ───────────────────────────────
    snippet_hits = sum(1 for t in tokens_lower if t in snippet.lower())
    if snippet_hits >= 2:
        score += 10
        signals.append(f"✓ Strong snippet relevance ({snippet_hits} keywords)")

    # ── Bonus: URL contains numeric IDs (team/player IDs → very specific) ──
    if re.search(r'/\d{3,}', path):
        score += 8
        signals.append("✓ URL contains specific ID (team/player page)")

    # ── Bonus: URL contains year ────────────────────────────────────────────
    if intent["years"] and any(y[:4] in url for y in intent["years"]):
        score += 6
        signals.append("✓ URL contains matching year/season")

    # ── Penalty: URL is a tag/category/search results page ─────────────────
    generic_url_fragments = ["/tag/", "/category/", "/search?", "/topics/",
                              "/author/", "/page/", "?s=", "?q="]
    if any(f in url for f in generic_url_fragments):
        score -= 15
        signals.append("⚠ Tag/category/search URL — penalized")

    # Clamp
    score = max(0.0, min(100.0, score))
    return score, signals


# ─────────────────────────────────────────────────────────────────────────────
# DuckDuckGo Search + Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def search_ddg(query: str, max_results: int = 10) -> list[dict]:
    """Run a DuckDuckGo search and return raw results."""
    if not HAS_DDG:
        raise RuntimeError("duckduckgo_search is not installed. Run: pip install duckduckgo_search")
    with DDGS() as ddgs:
        results = []
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "url": r.get("href", ""),
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
            })
        return results


def deduplicate(links: list[ScoredLink]) -> list[ScoredLink]:
    """Remove duplicate URLs, keeping the highest-scored version."""
    seen: dict[str, ScoredLink] = {}
    for link in links:
        canonical = link.url.split("?")[0].rstrip("/")
        if canonical not in seen or link.score > seen[canonical].score:
            seen[canonical] = link
    return list(seen.values())


def run_scraper(query: str, max_results: int = 12, min_score: float = 20.0,
                verbose: bool = False) -> list[ScoredLink]:
    """
    Main pipeline:
      1. Parse query intent
      2. Generate targeted sub-queries
      3. Search DuckDuckGo for each
      4. Score every result
      5. Deduplicate & rank
    """
    intent = extract_intent_keywords(query)
    search_queries = build_search_queries(intent)

    all_raw: list[dict] = []
    seen_queries = set()

    for sq in search_queries:
        if sq in seen_queries:
            continue
        seen_queries.add(sq)
        try:
            results = search_ddg(sq, max_results=8)
            for r in results:
                r["_query"] = sq
            all_raw.extend(results)
            time.sleep(0.4)   # polite delay between DDG calls
        except Exception as e:
            if verbose:
                print(f"[warn] search failed for '{sq}': {e}")

    # Score every result
    scored: list[ScoredLink] = []
    for r in all_raw:
        url = r.get("url", "")
        if not url or not url.startswith("http"):
            continue
        sc, signals = score_link(url, r["title"], r["snippet"], intent)
        if sc >= min_score:
            scored.append(ScoredLink(
                url=url,
                title=r["title"],
                snippet=r["snippet"][:200],
                score=sc,
                depth=url_depth(url),
                signals=signals,
                source=r.get("_query", "ddg"),
            ))

    # Deduplicate, sort by score descending
    unique = deduplicate(scored)
    unique.sort(key=lambda x: x.score, reverse=True)

    return unique[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# CLI & Display
# ─────────────────────────────────────────────────────────────────────────────

def print_results_rich(links: list[ScoredLink], query: str, intent: dict) -> None:
    console = Console()

    console.print()
    console.print(Panel(
        f"[bold cyan]🔍 Query:[/bold cyan]  {query}\n"
        f"[dim]Tokens:[/dim] {' • '.join(intent['tokens'])}\n"
        f"[dim]Stat signals:[/dim] {', '.join(intent['stat_words']) or 'general'}\n"
        f"[dim]Results found:[/dim] {len(links)}",
        title="[bold white]DeepLink Scraper[/bold white]",
        border_style="cyan",
        padding=(0, 2),
    ))

    if not links:
        console.print("[bold red]No sufficiently relevant links found.[/bold red]")
        console.print("Try a more specific query, or lower --min-score.")
        return

    for i, link in enumerate(links, 1):
        # Score color
        if link.score >= 70:
            score_color = "bright_green"
        elif link.score >= 45:
            score_color = "yellow"
        else:
            score_color = "red"

        console.print(Rule(
            f"[bold]{i:02d}[/bold]  [{score_color}]{link.score:.1f}/100[/{score_color}]  "
            f"[dim]depth={link.depth}[/dim]",
            style="dim"
        ))

        # Title + URL
        console.print(f"  [bold white]{link.title or '(no title)'}[/bold white]")
        console.print(f"  [bold cyan underline]{link.url}[/bold cyan underline]")

        # Snippet
        if link.snippet:
            console.print(f"  [dim]{link.snippet.strip()}[/dim]")

        # Signals
        if link.signals:
            for sig in link.signals:
                console.print(f"    [dim]{sig}[/dim]")

        console.print()


def print_results_plain(links: list[ScoredLink], query: str) -> None:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Results: {len(links)}")
    print("="*60)
    for i, link in enumerate(links, 1):
        print(f"\n{i:02d}. [{link.score:.1f}/100] {link.title}")
        print(f"    {link.url}")
        print(f"    {link.snippet[:120]}...")
        for sig in link.signals:
            print(f"      {sig}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DeepLink Scraper — find specific, relevant subpage URLs"
    )
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Search query (prompted if not provided)")
    parser.add_argument("--max", "-n", type=int, default=12,
                        help="Max results to return (default: 12)")
    parser.add_argument("--min-score", type=float, default=20.0,
                        help="Minimum relevance score 0-100 (default: 20)")
    parser.add_argument("--export", "-e", type=str, default=None,
                        help="Export results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # ── Dependency check ──────────────────────────────────────────────────
    missing = []
    if not HAS_DDG:
        missing.append("duckduckgo_search")
    if missing:
        print(f"[error] Missing packages: {', '.join(missing)}")
        print(f"Install with:  pip install {' '.join(missing)}")
        sys.exit(1)

    # ── Get query ─────────────────────────────────────────────────────────
    query = args.query
    if not query:
        if HAS_RICH:
            console = Console()
            console.print(Panel(
                "[bold cyan]DeepLink Scraper[/bold cyan]\n"
                "[dim]Finds specific, deep subpages directly relevant to your query.[/dim]",
                border_style="cyan"
            ))
            query = Prompt.ask("\n[bold]Enter your query[/bold]")
        else:
            print("DeepLink Scraper — Enter your query:")
            query = input("> ").strip()

    if not query:
        print("No query provided. Exiting.")
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────
    intent = extract_intent_keywords(query)

    if HAS_RICH:
        console = Console()
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[dim]{task.fields[status]}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Scraping...", total=100, status="building queries")
            progress.update(task, advance=20, status="running searches")
            links = run_scraper(query, max_results=args.max,
                                min_score=args.min_score, verbose=args.verbose)
            progress.update(task, advance=80, status="done")

        print_results_rich(links, query, intent)
    else:
        print("Searching...")
        links = run_scraper(query, max_results=args.max,
                            min_score=args.min_score, verbose=args.verbose)
        print_results_plain(links, query)

    # ── Export ────────────────────────────────────────────────────────────
    if args.export:
        data = {
            "query": query,
            "intent": intent,
            "results": [asdict(l) for l in links],
        }
        with open(args.export, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✅ Exported {len(links)} results → {args.export}")


if __name__ == "__main__":
    main()
