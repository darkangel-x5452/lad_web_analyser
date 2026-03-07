#!/usr/bin/env python3
"""
DuckDuckGo Search Script
Usage: python duckduckgo_search.py "your search query"
       python duckduckgo_search.py "your search query" --max 10
"""

import os
import urllib.request
import urllib.parse
import json
import argparse
import sys

from dotenv import load_dotenv
load_dotenv()  # loads .env from current directory

def search_duckduckgo(query: str, max_results: int = 10) -> list[dict]:
    """
    Search DuckDuckGo and return a list of results with title, URL, and snippet.
    Uses the DuckDuckGo Instant Answer API (free, no API key required).
    Falls back to HTML scraping for organic link results.
    """
    results = []

    # --- 1. Instant Answer API (structured data) ---
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    })
    api_url = f"https://api.duckduckgo.com/?{params}"

    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DDGSearchScript/1.0)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        # RelatedTopics contains the closest thing to organic links
        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            # Some entries are grouped (have a "Topics" sub-list)
            if "Topics" in topic:
                for sub in topic["Topics"]:
                    if len(results) >= max_results:
                        break
                    url = sub.get("FirstURL", "")
                    text = sub.get("Text", "")
                    if url:
                        results.append({"title": text[:80] or url, "url": url, "snippet": text})
            else:
                url = topic.get("FirstURL", "")
                text = topic.get("Text", "")
                if url:
                    results.append({"title": text[:80] or url, "url": url, "snippet": text})

    except Exception as e:
        print(f"[Instant Answer API error] {e}", file=sys.stderr)

    # --- 2. HTML scrape fallback for real web results ---
    # DuckDuckGo's HTML endpoint returns standard web results without JS
    if len(results) < max_results:
        html_params = urllib.parse.urlencode({"q": query, "kl": "us-en"})
        html_url = f"https://html.duckduckgo.com/html/?{html_params}"

        req2 = urllib.request.Request(
            html_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        try:
            with urllib.request.urlopen(req2, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Parse result blocks: each result sits inside <div class="result">
            seen_urls = {r["url"] for r in results}
            blocks = html.split('<div class="result__body">')

            for block in blocks[1:]:  # skip everything before the first result
                if len(results) >= max_results:
                    break

                # Extract URL
                url = ""
                if 'class="result__url"' in block:
                    url_start = block.find('href="', block.find('class="result__url"'))
                    if url_start != -1:
                        url_start += 6
                        url_end = block.find('"', url_start)
                        raw = block[url_start:url_end]
                        # DDG wraps URLs; unwrap if needed
                        if raw.startswith("//duckduckgo.com/l/?"):
                            qs = urllib.parse.parse_qs(urllib.parse.urlparse("https:" + raw).query)
                            url = qs.get("uddg", [raw])[0]
                        else:
                            url = raw if raw.startswith("http") else "https://" + raw.lstrip("/")

                # Extract title
                title = ""
                if 'class="result__a"' in block:
                    t_start = block.find(">", block.find('class="result__a"')) + 1
                    t_end = block.find("</a>", t_start)
                    title = _strip_tags(block[t_start:t_end]).strip()

                # Extract snippet
                snippet = ""
                if 'class="result__snippet"' in block:
                    s_start = block.find(">", block.find('class="result__snippet"')) + 1
                    s_end = block.find("</a>", s_start)
                    snippet = _strip_tags(block[s_start:s_end]).strip()

                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append({"title": title or url, "url": url, "snippet": snippet})

        except Exception as e:
            print(f"[HTML scrape error] {e}", file=sys.stderr)

    return results[:max_results]


def _strip_tags(text: str) -> str:
    """Remove HTML tags from a string."""
    result, inside = [], False
    for ch in text:
        if ch == "<":
            inside = True
        elif ch == ">":
            inside = False
        elif not inside:
            result.append(ch)
    return "".join(result)


def print_results(results: list[dict]) -> None:
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n{'─'*60}")
        print(f"[{i}] {r['title']}")
        print(f"    URL     : {r['url']}")
        if r.get("snippet"):
            snippet = r["snippet"][:200] + ("…" if len(r["snippet"]) > 200 else "")
            print(f"    Snippet : {snippet}")
    print(f"\n{'─'*60}")
    print(f"Total results shown: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Search DuckDuckGo and return links.")
    # parser.add_argument("query", help="Search query string")
    # parser.add_argument("--max", type=int, default=10, metavar="N",
    #                     help="Maximum number of results to return (default: 10)")
    # args = parser.parse_args()

    query = os.environ["DEMO_LINK_QUERY"]
    print(f'\nSearching DuckDuckGo for: "{query}"')
    results = search_duckduckgo(query, max_results=5)
    print_results(results)


if __name__ == "__main__":
    main()