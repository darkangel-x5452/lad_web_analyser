#!/usr/bin/env python3
"""
DuckDuckGo Search Script using the `duckduckgo-search` (DDGS) library.

Install:
    pip install duckduckgo-search

Usage:
    python duckduckgo_search.py "your search query"
    python duckduckgo_search.py "your search query" --max 10
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()  # loads .env from current directory

try:
    from ddgs import DDGS
except ImportError:
    print("Missing dependency. Install it with:\n\n    pip install duckduckgo-search\n")
    sys.exit(1)


def search(query: str, max_results: int = 10) -> list[dict]:
    """Return web search results for the given query."""
    with DDGS() as ddgs:
        return list(ddgs.text(
            query,
            max_results=max_results,
            safesearch="Off",
            timelimit="d",  # past day

            ))


def filter_results(results: list[dict]) -> list[dict]:
    context = "Live coverage of the "
    heading_context = f") Live Score - {os.environ['SPORT_SITE_1']}"
    filtered_ls = []
    if not results:
        print("No results found.")
        raise ValueError("No results found.")
    for i, r in enumerate(results, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}] {r.get('title', 'No title')}")
        print(f"    URL     : {r.get('href', 'N/A')}")
        heading = r.get('title', 'No title')
        snippet = r.get("body", "")
        href = r.get('href', 'N/A')
        if not snippet.startswith(context) \
            and not (context in snippet and "minutes ago " in snippet) \
            and not heading_context in heading:
            continue
        if "/game/" not in href:
            continue
        snippet = snippet[:200] + ("…" if len(snippet) > 200 else "")
        filtered_ls.append(r)
        
    return filtered_ls


def ddgs_main(
        query: str = os.environ.get("DEMO_LINK_QUERY", "NO QUERY PROVIDED"),
) -> list[dict]:
    # parser = argparse.ArgumentParser(description="Search DuckDuckGo and return links.")
    # parser.add_argument("query", help="Search query string")
    # parser.add_argument("--max", type=int, default=10, metavar="N",
    #                     help="Maximum number of results to return (default: 10)")
    # args = parser.parse_args()

    if query == "NO QUERY PROVIDED":
        print("No query provided. Please set the DEMO_LINK_QUERY environment variable.")
        return
    print(f'\nSearching DuckDuckGo for: "{query}"')
    results = search(query, max_results=5)
    filtered = filter_results(results)
    return filtered


if __name__ == "__main__":
    ddgs_main()