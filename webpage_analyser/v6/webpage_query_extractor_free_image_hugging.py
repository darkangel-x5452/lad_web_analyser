"""
Webpage Query Extractor  –  Vision + HuggingFace Edition
=========================================================
Takes a URL + a natural language query, screenshots the full page with
Playwright, then uses a vision LLM to extract only the query-relevant
information and return it as clean Markdown (tables included).

Backends  (--backend):
──────────────────────
  groq   (DEFAULT) – Llama 4 Scout Vision via Groq's free cloud API.
                     No GPU needed. Extremely fast.
                     Free key : https://console.groq.com
                     Env var  : GROQ_API_KEY

  huggingface      – Open-source vision models via HuggingFace Inference API.
                     Free tier available (rate-limited).
                     Free key : https://huggingface.co/settings/tokens
                     Env var  : HF_TOKEN
                     Default model : Qwen/Qwen2-VL-7B-Instruct
                     Other options :
                       meta-llama/Llama-3.2-11B-Vision-Instruct
                       microsoft/Phi-3.5-vision-instruct
                       mistralai/Pixtral-12B-2409

Install
───────
    pip install playwright groq huggingface_hub pillow
    playwright install chromium

CLI Usage
─────────
    # Groq (default)
    python webpage_query_extractor.py \\
        --url   "https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table" \\
        --query "Get the medal count table"

    # HuggingFace (free, open-source models)
    python webpage_query_extractor.py \\
        --url      "https://www.bbc.com/sport/football" \\
        --query    "Get the latest football scores" \\
        --backend  huggingface

    # HuggingFace with a specific model
    python webpage_query_extractor.py \\
        --url      "https://..." \\
        --query    "..." \\
        --backend  huggingface \\
        --hf-model "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Save output
    python webpage_query_extractor.py --url ... --query ... --output results.md

Library Usage
─────────────
    from webpage_query_extractor import extract_from_webpage

    md = extract_from_webpage(
        url="https://...",
        query="Get the pricing table",
        backend="huggingface",       # or "groq"
    )
    print(md)
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import tempfile
from pathlib import Path


# ── Playwright ────────────────────────────────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("Missing: pip install playwright && playwright install chromium")

# ── Pillow ────────────────────────────────────────────────────────────────────
try:
    from PIL import Image
except ImportError:
    sys.exit("Missing: pip install pillow")


# ── Lazy backend imports ──────────────────────────────────────────────────────
def _import_groq():
    try:
        from groq import Groq
        return Groq
    except ImportError:
        sys.exit("Missing: pip install groq")


def _import_hf():
    try:
        from huggingface_hub import InferenceClient
        return InferenceClient
    except ImportError:
        sys.exit("Missing: pip install huggingface_hub")


# ── Constants ─────────────────────────────────────────────────────────────────
SCREENSHOT_WIDTH  = 1440
SCREENSHOT_HEIGHT = 900

# Groq – hosted Llama vision (free tier)
GROQ_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# Alternatives:
#   llama-3.2-11b-vision-preview   – lighter / fastest
#   llama-3.2-90b-vision-preview   – most accurate

# HuggingFace – open-source vision models (free Inference API)
HF_DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
# Alternatives (pass via --hf-model):
#   meta-llama/Llama-3.2-11B-Vision-Instruct   – Llama family, strong accuracy
#   microsoft/Phi-3.5-vision-instruct           – compact, very capable
#   mistralai/Pixtral-12B-2409                  – Mistral vision, 12B
#   Qwen/Qwen2-VL-72B-Instruct                  – largest, highest accuracy

SYSTEM_PROMPT = (
    "You are a precise information-extraction assistant.\n"
    "You are given a screenshot of a webpage and a user query.\n\n"
    "Your job:\n"
    "1. Find ONLY the information directly relevant to the query.\n"
    "2. Reproduce tables as proper Markdown tables "
    "(| header | cols | with a --- separator row).\n"
    "3. Use appropriate Markdown: ## headings, **bold** for labels, "
    "bullet lists when needed.\n"
    "4. Ignore navigation bars, ads, site headers, footers, cookie banners, "
    "'next match' widgets, subscription prompts, social media buttons – "
    "anything NOT directly related to the query.\n"
    "5. If multiple sections match the query, include all with clear Markdown headings.\n"
    "6. Output ONLY the extracted Markdown – no preamble, no disclaimers, no filler text."
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Full-page screenshot via Playwright
# ─────────────────────────────────────────────────────────────────────────────

def capture_screenshot(url: str, output_path: str) -> str:
    """
    Launch headless Chromium, load `url`, scroll to trigger lazy content,
    and save a full-page PNG to `output_path`.
    """
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
        # Block ads/trackers for a cleaner render
        page.route(
            "**/{ads,analytics,tracking,doubleclick,googlesyndication}**",
            lambda route: route.abort(),
        )
        page.goto(url, wait_until="networkidle", timeout=30_000)
        # Scroll to trigger lazy-loaded images/tables
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1_500)
        page.screenshot(path=output_path, full_page=True)
        browser.close()

    kb = Path(output_path).stat().st_size / 1024
    print(f"    Screenshot saved -> {output_path}  ({kb:.1f} KB)")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Resize to stay within model upload limits
# ─────────────────────────────────────────────────────────────────────────────

def prepare_image(path: str, max_height: int = 6000, max_mb: float = 4.0) -> str:
    """
    Resize proportionally if height > max_height OR file size > max_mb.
    Returns the (possibly new) file path.
    """
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    with Image.open(path) as img:
        w, h = img.size

    if h <= max_height and size_mb <= max_mb:
        return path

    with Image.open(path) as img:
        ratio = min(max_height / h, (max_mb / size_mb) ** 0.5, 1.0)
        nw = max(1, int(w * ratio))
        nh = max(1, int(h * ratio))
        tmp = tempfile.mktemp(suffix=".png")
        img.resize((nw, nh), Image.LANCZOS).save(tmp, "PNG", optimize=True)

    new_mb = Path(tmp).stat().st_size / (1024 * 1024)
    print(f"    Resized: {w}x{h} -> {nw}x{nh}  ({new_mb:.2f} MB)")
    return tmp


def _to_b64(path: str) -> str:
    """Read a PNG and return its Base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Step 3a – Groq backend  (Llama Vision, free cloud API)
# ─────────────────────────────────────────────────────────────────────────────

def query_groq(
    image_path: str,
    query: str,
    api_key: str,
    model: str = GROQ_DEFAULT_MODEL,
) -> str:
    """
    Send the screenshot to Groq's hosted Llama Vision endpoint.

    Free-tier models (pass via --groq-model):
        meta-llama/llama-4-scout-17b-16e-instruct  (latest, default)
        llama-3.2-11b-vision-preview               (lighter, fastest)
        llama-3.2-90b-vision-preview               (most accurate)
    """
    Groq = _import_groq()
    client = Groq(api_key=api_key)
    print(f"[2/3] Querying Groq ({model})...")

    b64 = _to_b64(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Query: {query}\n\n"
                            "Extract the relevant information from this webpage screenshot."
                        ),
                    },
                ],
            },
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b – HuggingFace backend  (open-source vision models, free API)
# ─────────────────────────────────────────────────────────────────────────────

def query_huggingface(
    image_path: str,
    query: str,
    api_key: str,
    model: str = HF_DEFAULT_MODEL,
) -> str:
    """
    Send the screenshot to a HuggingFace Inference API vision model.

    Free-tier models (pass via --hf-model):
        Qwen/Qwen2-VL-7B-Instruct                  (default, well-rounded)
        meta-llama/Llama-3.2-11B-Vision-Instruct   (Llama family)
        microsoft/Phi-3.5-vision-instruct           (compact, fast)
        mistralai/Pixtral-12B-2409                  (Mistral vision)
        Qwen/Qwen2-VL-72B-Instruct                  (largest, best accuracy)

    Note: Free tier is rate-limited. For higher limits get a PRO token
          or use a self-hosted Inference Endpoint.
    """
    InferenceClient = _import_hf()
    client = InferenceClient(
        model=model,
        token=api_key,
    )
    print(f"[2/3] Querying HuggingFace Inference API ({model})...")

    # Read image and encode as data-URI for the messages API
    b64 = _to_b64(image_path)
    image_data_uri = f"data:image/png;base64,{b64}"

    # HuggingFace InferenceClient supports the OpenAI-compatible chat interface
    # for vision models when using the `chat_completion` method
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Query: {query}\n\n"
                            "Extract the relevant information from this webpage screenshot."
                        ),
                    },
                ],
            },
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_webpage(
    url: str,
    query: str,
    backend: str           = "groq",
    api_key: str | None    = None,
    groq_model: str        = GROQ_DEFAULT_MODEL,
    hf_model: str          = HF_DEFAULT_MODEL,
    keep_screenshot: bool  = False,
) -> str:
    """
    Main entry point.

    Args:
        url:             Webpage URL to screenshot and analyse.
        query:           What to extract (natural language).
        backend:         "groq" (default, free Llama cloud) or "huggingface".
        api_key:         API key. Falls back to GROQ_API_KEY / HF_TOKEN env vars.
        groq_model:      Override the Groq model.
        hf_model:        Override the HuggingFace model.
        keep_screenshot: Keep the temp PNG on disk.

    Returns:
        Extracted content as a Markdown string.
    """
    backend = backend.lower()
    if backend not in ("groq", "huggingface", "hf"):
        sys.exit(f"Unknown backend '{backend}'. Choose 'groq' or 'huggingface'.")

    # Normalise alias
    if backend == "hf":
        backend = "huggingface"

    # Resolve API key
    if backend == "groq":
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            sys.exit(
                "No Groq API key found.\n"
                "  Free key : https://console.groq.com  (sign up -> API Keys)\n"
                "  Env var  : export GROQ_API_KEY=your_key\n"
                "  Or pass  : --api-key your_key"
            )
    else:
        api_key = api_key or os.environ.get("HF_TOKEN")
        if not api_key:
            sys.exit(
                "No HuggingFace token found.\n"
                "  Free token : https://huggingface.co/settings/tokens\n"
                "  Env var    : export HF_TOKEN=hf_...\n"
                "  Or pass    : --api-key hf_..."
            )

    tmp_png   = tempfile.mktemp(suffix="_webpage.png")
    ready_img = tmp_png

    try:
        capture_screenshot(url, tmp_png)
        ready_img = prepare_image(tmp_png)

        if backend == "groq":
            markdown = query_groq(ready_img, query, api_key, model=groq_model)
        else:
            markdown = query_huggingface(ready_img, query, api_key, model=hf_model)

    finally:
        if not keep_screenshot:
            for p in {tmp_png, ready_img}:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass

    print("[3/3] Extraction complete.\n")
    return markdown


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract query-relevant info from a webpage screenshot using "
            "open-source vision LLMs (Groq cloud or HuggingFace Inference API)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Groq – default, free, fast
  python webpage_query_extractor.py \\
      --url   "https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table" \\
      --query "Get the medal count table"

  # HuggingFace – default model (Qwen2-VL-7B)
  python webpage_query_extractor.py \\
      --url     "https://www.nba.com/stats/teams/traditional" \\
      --query   "Team statistics table" \\
      --backend huggingface

  # HuggingFace – Llama 3.2 Vision
  python webpage_query_extractor.py \\
      --url      "https://..." --query "..." \\
      --backend  huggingface \\
      --hf-model "meta-llama/Llama-3.2-11B-Vision-Instruct"

  # HuggingFace – Phi-3.5 (compact + fast)
  python webpage_query_extractor.py \\
      --url      "https://..." --query "..." \\
      --backend  huggingface \\
      --hf-model "microsoft/Phi-3.5-vision-instruct"

  # Save output to Markdown file
  python webpage_query_extractor.py --url ... --query ... --output results.md
""",
    )

    parser.add_argument("--url",    required=True, help="Webpage URL to analyse")
    parser.add_argument("--query",  required=True, help="What information to extract")
    parser.add_argument("--output", default=None,  help="Save Markdown output to this file")
    parser.add_argument("--keep-screenshot", action="store_true",
                        help="Keep the temporary screenshot PNG on disk")

    bg = parser.add_argument_group("Backend")
    bg.add_argument(
        "--backend", default="groq", choices=["groq", "huggingface", "hf"],
        help="'groq' = free Llama cloud API (default) | 'huggingface' = HF Inference API",
    )

    gg = parser.add_argument_group("Groq options")
    gg.add_argument("--api-key",    default=None,
                    help="API key for selected backend (or set GROQ_API_KEY / HF_TOKEN)")
    gg.add_argument("--groq-model", default=GROQ_DEFAULT_MODEL,
                    help=f"Groq model (default: {GROQ_DEFAULT_MODEL})\n"
                         "  llama-3.2-11b-vision-preview  – lighter/fastest\n"
                         "  llama-3.2-90b-vision-preview  – most accurate")

    hg = parser.add_argument_group("HuggingFace options")
    hg.add_argument(
        "--hf-model", default=HF_DEFAULT_MODEL,
        help=(
            f"HuggingFace model (default: {HF_DEFAULT_MODEL})\n"
            "  meta-llama/Llama-3.2-11B-Vision-Instruct\n"
            "  microsoft/Phi-3.5-vision-instruct\n"
            "  mistralai/Pixtral-12B-2409\n"
            "  Qwen/Qwen2-VL-72B-Instruct"
        ),
    )

    args = parser.parse_args()

    result = extract_from_webpage(
        url=args.url,
        query=args.query,
        backend=args.backend,
        api_key=args.api_key,
        groq_model=args.groq_model,
        hf_model=args.hf_model,
        keep_screenshot=args.keep_screenshot,
    )

    print("─" * 70)
    print(result)
    print("─" * 70)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()