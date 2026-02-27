"""
Webpage Query Extractor  –  DOM-first, Vision-fallback Edition
==============================================================
Extracts query-relevant information from a webpage as clean Markdown.

Strategy (why DOM beats Vision for accuracy):
─────────────────────────────────────────────
  Vision models read *pixels* – digits like 1/7, 0/6, 3/8 get confused.
  Playwright can pull the *actual text* straight from the DOM – perfectly
  accurate, no OCR involved.  A small LLM then filters and formats it.

  Vision is kept as a fallback for canvas-rendered pages (e.g. Tableau,
  some charting tools) where the DOM holds no readable text.

Backends  (--backend):
──────────────────────
  dom+groq    (DEFAULT) – Playwright extracts DOM text, Groq Llama filters it.
                          100% accurate values. No GPU needed.
                          Free key: https://console.groq.com
                          Env var : GROQ_API_KEY

  dom+ollama             – Same DOM extraction, local Ollama LLM for filtering.
                           Fully private. No GPU required (text-only LLM is tiny).
                           Env var : OLLAMA_HOST, OLLAMA_MODEL

  vision+groq            – Screenshot → Groq Llama Vision.
                           Use when the page is canvas/SVG rendered.

  vision+ollama          – Screenshot → local Ollama vision model.
                           Use when the page is canvas/SVG rendered.

Install
───────
    pip install playwright groq ollama pillow
    playwright install chromium

CLI Usage
─────────
    # DOM + Groq (default, most accurate)
    python webpage_query_extractor.py \\
        --url   "https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table" \\
        --query "Get the medal count table"

    # DOM + Ollama (local, most accurate + private)
    python webpage_query_extractor.py \\
        --url     "https://..." \\
        --query   "Get the stats table" \\
        --backend dom+ollama --ollama-model llama3.2

    # Vision fallback (canvas/SVG pages only)
    python webpage_query_extractor.py \\
        --url     "https://..." \\
        --query   "..." \\
        --backend vision+groq

Library Usage
─────────────
    from webpage_query_extractor import extract_from_webpage

    md = extract_from_webpage(url="https://...", query="Get the pricing table")
    print(md)
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import tempfile
from pathlib import Path

# ── Playwright ────────────────────────────────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("Missing: pip install playwright && playwright install chromium")

# ── Pillow (vision path only) ─────────────────────────────────────────────────
try:
    from PIL import Image
except ImportError:
    Image = None   # only required for vision backends


# ── Lazy backend imports ──────────────────────────────────────────────────────
def _import_groq():
    try:
        from groq import Groq
        return Groq
    except ImportError:
        sys.exit("Missing: pip install groq")

def _import_ollama():
    try:
        import ollama
        return ollama
    except ImportError:
        sys.exit("Missing: pip install ollama")


# ── Constants ─────────────────────────────────────────────────────────────────
SCREENSHOT_WIDTH  = 1440
SCREENSHOT_HEIGHT = 900

# Groq models
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEXT_MODEL   = "llama-3.3-70b-versatile"   # text-only, faster + accurate

# Ollama models
OLLAMA_VISION_MODEL = "llama3.2-vision" # SUcks
OLLAMA_TEXT_MODEL   = "llama3.2:3b"                 # text-only, tiny, no GPU needed
OLLAMA_DEFAULT_HOST = "http://localhost:11434"

# How many characters of DOM text to send (avoid token limits)
DOM_MAX_CHARS = 80_000

SYSTEM_PROMPT_TEXT = """\
You are a precise information-extraction assistant.
You receive raw text scraped from a webpage (may include menus, footers, etc.)
and a user query.

Your job:
1. Find ONLY the content directly relevant to the query.
2. Reproduce any tables as proper Markdown tables (| col | col | with --- separator rows).
3. Use Markdown: ## headings, **bold** labels, bullet lists where natural.
4. Ignore navigation menus, ads, site headers, footers, cookie notices,
   subscription prompts, social share buttons – anything unrelated to the query.
5. Preserve all numbers, percentages, and statistics EXACTLY as they appear.
6. Output ONLY the extracted Markdown – no preamble, no disclaimers."""

SYSTEM_PROMPT_VISION = """\
You are a precise information-extraction assistant.
You are given a screenshot of a webpage and a user query.

Your job:
1. Find ONLY the information directly relevant to the query.
2. Reproduce tables as proper Markdown tables (| col | with --- separator rows).
3. Use Markdown: ## headings, **bold** labels, bullet lists where natural.
4. Ignore navigation bars, ads, headers, footers, cookie banners, next-match
   widgets, subscription prompts – anything unrelated to the query.
5. Copy all numbers and statistics with pixel-perfect accuracy.
6. Output ONLY the extracted Markdown – no preamble, no disclaimers."""


# ─────────────────────────────────────────────────────────────────────────────
# DOM extraction  (Playwright)
# ─────────────────────────────────────────────────────────────────────────────

def extract_dom_text(url: str) -> str:
    """
    Load the page in headless Chromium and return structured text built from
    the live DOM: <table> elements are converted to TSV, then all visible text
    is appended.  This is lossless – every number is taken straight from the
    HTML, not read from pixels.
    """
    print(f"[1/3] Extracting DOM text: {url}")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": SCREENSHOT_WIDTH, "height": SCREENSHOT_HEIGHT},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page.route(
            "**/{ads,analytics,tracking,doubleclick,googlesyndication}**",
            lambda route: route.abort(),
        )
        page.goto(url, wait_until="networkidle", timeout=30_000)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1_500)

        # ── Extract tables as TSV first (preserves structure perfectly) ────────
        tables_tsv: str = page.evaluate("""() => {
            const tables = document.querySelectorAll('table');
            return Array.from(tables).map((tbl, ti) => {
                const rows = Array.from(tbl.querySelectorAll('tr'));
                const tsv  = rows.map(r =>
                    Array.from(r.querySelectorAll('th,td'))
                        .map(c => c.innerText.trim().replace(/\\n/g, ' '))
                        .join('\\t')
                ).join('\\n');
                return `[TABLE ${ti + 1}]\\n${tsv}`;
            }).join('\\n\\n');
        }""")

        # ── Full visible text (headings, paragraphs, lists, etc.) ──────────────
        full_text: str = page.evaluate("""() => {
            // Remove script/style/nav/footer noise from a clone
            const clone = document.body.cloneNode(true);
            ['script','style','nav','footer','header',
             'aside','noscript','iframe'].forEach(tag => {
                clone.querySelectorAll(tag).forEach(el => el.remove());
            });
            return clone.innerText;
        }""")

        browser.close()

    # Combine: tables first (they're the most structured), then body text
    combined = ""
    if tables_tsv.strip():
        combined += "=== TABLES EXTRACTED FROM PAGE ===\n\n" + tables_tsv + "\n\n"
    combined += "=== PAGE TEXT ===\n\n" + full_text

    # Trim to avoid blowing token limits
    if len(combined) > DOM_MAX_CHARS:
        combined = combined[:DOM_MAX_CHARS] + "\n...[truncated]"

    print(f"    DOM text extracted  ({len(combined):,} chars)")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Screenshot capture  (vision path)
# ─────────────────────────────────────────────────────────────────────────────

def capture_screenshot(url: str, output_path: str) -> str:
    """Headless Chromium full-page PNG screenshot."""
    print(f"[1/3] Capturing screenshot: {url}")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": SCREENSHOT_WIDTH, "height": SCREENSHOT_HEIGHT},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page.route(
            "**/{ads,analytics,tracking,doubleclick,googlesyndication}**",
            lambda route: route.abort(),
        )
        page.goto(url, wait_until="networkidle", timeout=30_000)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1_500)
        page.screenshot(path=output_path, full_page=True)
        browser.close()

    kb = Path(output_path).stat().st_size / 1024
    print(f"    Screenshot saved -> {output_path}  ({kb:.1f} KB)")
    return output_path


def prepare_image(path: str, max_height: int = 6000, max_mb: float = 4.0) -> str:
    """Resize screenshot if needed to stay within model upload limits."""
    if Image is None:
        sys.exit("Missing: pip install pillow  (required for vision backends)")
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    with Image.open(path) as img:
        w, h = img.size
    if h <= max_height and size_mb <= max_mb:
        return path
    with Image.open(path) as img:
        ratio = min(max_height / h, (max_mb / size_mb) ** 0.5, 1.0)
        nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
        tmp = tempfile.mktemp(suffix=".png")
        img.resize((nw, nh), Image.LANCZOS).save(tmp, "PNG", optimize=True)
    print(f"    Resized: {w}x{h} -> {nw}x{nh}")
    return tmp


def _to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Groq  (text + vision)
# ─────────────────────────────────────────────────────────────────────────────

def query_groq_text(dom_text: str, query: str, api_key: str,
                    model: str = GROQ_TEXT_MODEL) -> str:
    """Send extracted DOM text to Groq text LLM – fast, accurate, free."""
    Groq = _import_groq()
    client = Groq(api_key=api_key)
    print(f"[2/3] Querying Groq text model ({model})...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEXT},
            {"role": "user",   "content": (
                f"Query: {query}\n\n"
                f"Webpage content:\n\n{dom_text}"
            )},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def query_groq_vision(image_path: str, query: str, api_key: str,
                      model: str = GROQ_VISION_MODEL) -> str:
    """Send screenshot to Groq vision LLM."""
    Groq = _import_groq()
    client = Groq(api_key=api_key)
    print(f"[2/3] Querying Groq vision model ({model})...")
    b64 = _to_b64(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_VISION},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text",
                 "text": f"Query: {query}\n\nExtract relevant info from this screenshot."},
            ]},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# Ollama  (text + vision)
# ─────────────────────────────────────────────────────────────────────────────

def query_ollama_text(dom_text: str, query: str,
                      model: str = OLLAMA_TEXT_MODEL,
                      host: str  = OLLAMA_DEFAULT_HOST) -> str:
    """Send extracted DOM text to a local Ollama text model."""
    ollama = _import_ollama()
    if host != OLLAMA_DEFAULT_HOST:
        os.environ["OLLAMA_HOST"] = host
    print(f"[2/3] Querying Ollama text model ({model})...")

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEXT},
            {"role": "user",   "content": (
                f"Query: {query}\n\n"
                f"Webpage content:\n\n{dom_text}"
            )},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


def query_ollama_vision(image_path: str, query: str,
                        model: str = OLLAMA_VISION_MODEL,
                        host: str  = OLLAMA_DEFAULT_HOST) -> str:
    """Send screenshot to a local Ollama vision model."""
    ollama = _import_ollama()
    if host != OLLAMA_DEFAULT_HOST:
        os.environ["OLLAMA_HOST"] = host
    print(f"[2/3] Querying Ollama vision model ({model})...")
    b64 = _to_b64(image_path)

    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                f"{SYSTEM_PROMPT_VISION}\n\n"
                f"Query: {query}\n\n"
                "Extract the relevant information from the attached screenshot."
            ),
            "images": [b64],
        }],
        options={"temperature": 0.0, "num_gpu": 999},
    )
    return response["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

VALID_BACKENDS = ("dom+groq", "dom+ollama", "vision+groq", "vision+ollama")

def extract_from_webpage(
    url: str,
    query: str,
    backend: str          = "dom+groq",
    api_key: str | None   = None,
    groq_text_model: str  = GROQ_TEXT_MODEL,
    groq_vision_model: str= GROQ_VISION_MODEL,
    ollama_text_model: str= OLLAMA_TEXT_MODEL,
    ollama_vision_model: str = OLLAMA_VISION_MODEL,
    ollama_host: str      = OLLAMA_DEFAULT_HOST,
    keep_screenshot: bool = False,
) -> str:
    """
    Main entry point.

    Args:
        url:                 Webpage URL to analyse.
        query:               What to extract (natural language).
        backend:             One of: dom+groq (default), dom+ollama,
                             vision+groq, vision+ollama.
        api_key:             Groq API key (or GROQ_API_KEY env var).
        groq_text_model:     Groq text model override.
        groq_vision_model:   Groq vision model override.
        ollama_text_model:   Ollama text model override.
        ollama_vision_model: Ollama vision model override.
        ollama_host:         Ollama server URL override.
        keep_screenshot:     Keep temp PNG (vision backends only).

    Returns:
        Extracted content as a Markdown string.
    """
    backend = backend.lower()
    if backend not in VALID_BACKENDS:
        sys.exit(f"Unknown backend '{backend}'.\nChoose from: {', '.join(VALID_BACKENDS)}")

    use_groq   = "groq"   in backend
    use_vision = "vision" in backend

    if use_groq:
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            sys.exit(
                "No Groq API key.\n"
                "  Free key : https://console.groq.com\n"
                "  Env var  : export GROQ_API_KEY=your_key"
            )

    tmp_png   = None
    ready_img = None

    try:
        if use_vision:
            tmp_png = tempfile.mktemp(suffix="_webpage.png")
            capture_screenshot(url, tmp_png)
            ready_img = prepare_image(tmp_png)
            if use_groq:
                md = query_groq_vision(ready_img, query, api_key, groq_vision_model)
            else:
                md = query_ollama_vision(ready_img, query, ollama_vision_model, ollama_host)
        else:
            dom_text = extract_dom_text(url)
            if use_groq:
                md = query_groq_text(dom_text, query, api_key, groq_text_model)
            else:
                md = query_ollama_text(dom_text, query, ollama_text_model, ollama_host)

    finally:
        if not keep_screenshot:
            for p in filter(None, [tmp_png, ready_img]):
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass

    print("[3/3] Extraction complete.\n")
    return md


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(
        url_input: str,
        query_input: str,
        output_file: str
) -> None:
#     parser = argparse.ArgumentParser(
#         description="Extract query-relevant info from a webpage using DOM or vision LLMs.",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Backends:
#   dom+groq      DOM text → Groq Llama  (DEFAULT – most accurate, free cloud API)
#   dom+ollama    DOM text → local Ollama LLM  (accurate + fully private)
#   vision+groq   Screenshot → Groq Llama Vision  (use for canvas/SVG pages)
#   vision+ollama Screenshot → local Ollama vision model

# Examples:
#   # Accurate table extraction (DOM, default)
#   python webpage_query_extractor.py \\
#       --url   "https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table" \\
#       --query "Get the medal count table"

#   # Local + private (no API key)
#   python webpage_query_extractor.py \\
#       --url     "https://..." --query "Get the stats" \\
#       --backend dom+ollama --ollama-text-model llama3.2

#   # Canvas/SVG page (vision fallback)
#   python webpage_query_extractor.py \\
#       --url "https://..." --query "..." --backend vision+groq

#   # Save output
#   python webpage_query_extractor.py --url ... --query ... --output results.md
# """,
#     )

#     parser.add_argument("--url",    required=True)
#     parser.add_argument("--query",  required=True)
#     parser.add_argument("--output", default=None, help="Save Markdown to file")
#     parser.add_argument("--keep-screenshot", action="store_true")

#     bg = parser.add_argument_group("Backend")
#     bg.add_argument("--backend", default="dom+groq", choices=list(VALID_BACKENDS),
#                     help="Default: dom+groq")

#     gg = parser.add_argument_group("Groq options")
#     gg.add_argument("--api-key",           default=None)
#     gg.add_argument("--groq-text-model",   default=GROQ_TEXT_MODEL,
#                     help=f"Default: {GROQ_TEXT_MODEL}")
#     gg.add_argument("--groq-vision-model", default=GROQ_VISION_MODEL,
#                     help=f"Default: {GROQ_VISION_MODEL}")

#     og = parser.add_argument_group("Ollama options")
#     og.add_argument("--ollama-text-model",   default=OLLAMA_TEXT_MODEL,
#                     help=f"Default: {OLLAMA_TEXT_MODEL}")
#     og.add_argument("--ollama-vision-model", default=OLLAMA_VISION_MODEL,
#                     help=f"Default: {OLLAMA_VISION_MODEL}")
#     og.add_argument("--ollama-host",         default=OLLAMA_DEFAULT_HOST,
#                     help=f"Default: {OLLAMA_DEFAULT_HOST}")

#     args = parser.parse_args()

    result = extract_from_webpage(
        url=url_input,
        query=query_input,
        backend="dom+ollama",
        api_key=None,
        # groq_text_model=GROQ_TEXT_MODEL,
        # groq_vision_model=GROQ_VISION_MODEL,
        ollama_text_model=OLLAMA_TEXT_MODEL,
        ollama_vision_model=OLLAMA_VISION_MODEL,
        ollama_host=OLLAMA_DEFAULT_HOST,
        keep_screenshot=False,
    )

    print("─" * 70)
    print(result)
    print("─" * 70)

    if output_file:
        Path(output_file).write_text(result, encoding="utf-8")
        print(f"\nSaved -> {output_file}")


if __name__ == "__main__":
    main(
        url_input=os.getenv("DEMO_URL_LINK"),
        query_input=os.getenv("DEMO_URL_QUERY"),
        output_file="data/webpage_analyser/v6/demo_output_free_dom.md",
    )
