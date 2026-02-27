"""
Webpage Query Extractor
------------------------
Takes a URL + a natural language query, screenshots the page, and uses
Google Gemini (free tier) to extract only the relevant information as Markdown.

Requirements:
    pip install playwright google-generativeai pillow
    playwright install chromium

Usage:
    python webpage_query_extractor.py \
        --url "https://www.nba.com/stats/teams/traditional" \
        --query "Get the team statistics table"

    # Or as a library:
    from webpage_query_extractor import extract_from_webpage
    result = extract_from_webpage(url, query)
    print(result)

Environment:
    GEMINI_API_KEY  – get a free key at https://aistudio.google.com/app/apikey
"""

import argparse
import base64
import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env from current directory

# ── dependencies ──────────────────────────────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("Missing: pip install playwright && playwright install chromium")

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    sys.exit("Missing: pip install google-generativeai")

try:
    from PIL import Image
except ImportError:
    sys.exit("Missing: pip install pillow")

# ── Gemini model (free tier) ──────────────────────────────────────────────────
# https://ai.google.dev/gemini-api/docs/pricing
# GEMINI_MODEL = "gemini-3-flash-preview"          # * A customer-submitted request to Gemini may result in one or more queries to Google Search. You will be charged for each individual search query performed.
# GEMINI_MODEL = "gemini-2.5-pro"          # Free, fast, excellent at vision+tables
# GEMINI_MODEL = "gemma-3-27b-it"          # Free, fast, excellent at vision+tables
GEMINI_MODEL = "gemini-2.5-flash"          # Free, fast, excellent at vision+tables
SCREENSHOT_WIDTH  = 1440
SCREENSHOT_HEIGHT = 900

# ── Safety settings: allow all so structured data is never blocked ────────────
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Capture a full-page screenshot with Playwright
# ─────────────────────────────────────────────────────────────────────────────

def capture_screenshot(url: str, output_path: str) -> str:
    """
    Launches a headless Chromium browser, loads `url`, and saves a full-page
    screenshot (PNG) to `output_path`.  Returns the path for convenience.
    """
    print(f"[1/3] Capturing screenshot: {url}")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": SCREENSHOT_WIDTH, "height": SCREENSHOT_HEIGHT}
        )
        # Block ads / trackers for a cleaner render
        page.route(
            "**/{ads,analytics,tracking,doubleclick}**",
            lambda route: route.abort()
        )
        page.goto(url, wait_until="networkidle", timeout=30_000)
        # Optional: scroll to trigger lazy-loaded content
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1500)

        page.screenshot(path=output_path, full_page=True)
        browser.close()

    size = Path(output_path).stat().st_size / 1024
    print(f"    Screenshot saved → {output_path}  ({size:.1f} KB)")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Resize if the image is very tall (Gemini cap: ~4 MB / 4000 px)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_image(path: str, max_height: int = 8000) -> str:
    """
    If the screenshot exceeds max_height pixels, resize it proportionally
    and save to a temp file.  Returns the (possibly new) file path.
    """
    with Image.open(path) as img:
        w, h = img.size
        if h <= max_height:
            return path
        ratio  = max_height / h
        new_w  = int(w * ratio)
        resized = img.resize((new_w, max_height), Image.LANCZOS)
        tmp = tempfile.mktemp(suffix=".png")
        resized.save(tmp, "PNG", optimize=True)
        print(f"    Image resized to {new_w}×{max_height} → {tmp}")
        return tmp


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Ask Gemini to extract only what the query asks for
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise information-extraction assistant.
You are given a screenshot of a webpage and a user query.

Your job:
1. Find ONLY the information directly relevant to the query.
2. Reproduce tables as proper Markdown tables (with | separators and header rows).
3. Use appropriate Markdown: ## headings, **bold** for labels, bullet lists when needed.
4. Ignore navigation bars, ads, headers, footers, cookie banners, "next match" widgets,
   subscription prompts, or anything else NOT directly related to the query.
5. If multiple sections match the query, include all of them with clear Markdown headings.
6. Never add commentary, disclaimers, or filler text – output ONLY the extracted content.
"""

def query_gemini(image_path: str, query: str, api_key: str) -> str:
    """Send the screenshot + query to Gemini and return the Markdown answer."""
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)

    print(f"[2/3] Sending to Gemini ({GEMINI_MODEL})…")
    with open(image_path, "rb") as f:
        image_data = f.read()

    image_part = {
        "mime_type": "image/png",
        "data": base64.b64encode(image_data).decode("utf-8"),
    }
    user_prompt = f"Query: {query}\n\nExtract the relevant information from this webpage screenshot."

    response = model.generate_content(
        [image_part, user_prompt],
        safety_settings=SAFETY_SETTINGS,
        generation_config={"temperature": 0.1},   # Low temp = factual / exact
    )
    return response.text


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_webpage(
    url: str,
    query: str,
    api_key: str | None = None,
    keep_screenshot: bool = False,
) -> str:
    """
    Main entry point.

    Args:
        url:             The webpage URL to analyse.
        query:           Natural-language query describing what to extract.
        api_key:         Gemini API key. Falls back to GEMINI_API_KEY env var.
        keep_screenshot: If True, do NOT delete the temporary screenshot.

    Returns:
        Markdown string with the extracted information.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "No Gemini API key found.\n"
            "  • Set env var:  export GEMINI_API_KEY=your_key\n"
            "  • Or pass --api-key\n"
            "  • Free key:     https://aistudio.google.com/app/apikey"
        )

    # Temp file for the screenshot
    tmp_png = tempfile.mktemp(suffix="_webpage.png")
    try:
        capture_screenshot(url, tmp_png)
        ready_img = prepare_image(tmp_png)
        markdown  = query_gemini(ready_img, query, api_key)
    finally:
        if not keep_screenshot:
            for p in {tmp_png, ready_img if 'ready_img' in dir() else tmp_png}:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass

    print("[3/3] Extraction complete.\n")
    return markdown


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(
        url_input: str,
        query_input: str,
        output_file: str,
):
    # parser = argparse.ArgumentParser(
    #     description="Extract query-relevant info from a webpage screenshot using Gemini."
    # )
    # parser.add_argument("--url",           required=True, help="Webpage URL to analyse")
    # parser.add_argument("--query",         required=True, help="What information to extract")
    # parser.add_argument("--api-key",       default=None,  help="Gemini API key (or set GEMINI_API_KEY)")
    # parser.add_argument("--output",        default=None,  help="Save Markdown output to this file")
    # parser.add_argument("--keep-screenshot", action="store_true",
    #                     help="Keep the screenshot PNG for inspection")
    # args = parser.parse_args()

    result = extract_from_webpage(
        url=url_input,
        query=query_input,
        api_key=os.getenv("GEMINI_API_KEY"),
        keep_screenshot=True,
    )

    # ── Print to terminal ─────────────────────────────────────────────────────
    print("=" * 70)
    print(result)
    print("=" * 70)

    # ── Optionally save to file ───────────────────────────────────────────────
    if output_file:
        Path(output_file).write_text(result, encoding="utf-8")
        print(f"\nSaved → {output_file}")


if __name__ == "__main__":
    main(
        url_input=os.getenv("DEMO_URL_LINK"),
        query_input=os.getenv("DEMO_URL_QUERY"),
        output_file="data/webpage_analyser/v6/demo_output_paid.md",
    )
