"""
Webpage Query Extractor  –  Open-Source Vision Edition
=======================================================
Takes a URL + a natural language query, screenshots the full page with
Playwright, then uses an open-source vision LLM to extract only the
query-relevant information and return it as clean Markdown (with tables).

Two backend choices (select with --backend):
─────────────────────────────────────────────
  groq   (DEFAULT) – Llama 3.2 / Llama 4 Vision via Groq's free cloud API.
                     No GPU needed. Extremely fast.
                     Free key: https://console.groq.com  (sign up → API Keys)
                     Env var : GROQ_API_KEY

  ollama            – Any vision model running locally via Ollama.
                     Fully private, no internet call for inference.
                     Install : https://ollama.com
                     Then pull a model, e.g.:
                         ollama pull llama3.2-vision   # 8 B, ~6 GB RAM
                         ollama pull llava             # classic, ~4 GB RAM
                         ollama pull moondream         # tiny,   ~2 GB RAM
                     Env var : OLLAMA_HOST  (default: http://localhost:11434)
                               OLLAMA_MODEL (default: llama3.2-vision)

Install
───────
    pip install playwright groq ollama pillow
    playwright install chromium

CLI Usage
─────────
    # Groq (default)
    python webpage_query_extractor.py \\
        --url   "https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table" \\
        --query "Get the medal count table"

    # Ollama (local)
    python webpage_query_extractor.py \\
        --url     "https://www.bbc.com/sport/football" \\
        --query   "Get the latest football scores" \\
        --backend ollama --ollama-model llava

    # Save output to file
    python webpage_query_extractor.py --url ... --query ... --output results.md

Library Usage
─────────────
    from webpage_query_extractor import extract_from_webpage

    md = extract_from_webpage(
        url="https://...",
        query="Get the pricing table",
        backend="groq",       # or "ollama"
    )
    print(md)
"""

from __future__ import annotations

import base64
from datetime import datetime
import os
import re
import sys
import tempfile
from pathlib import Path
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions

from dotenv import load_dotenv

# from configs.screen_emulations.emulations import emulate_device

load_dotenv()  # loads .env from current directory


# ── Playwright ────────────────────────────────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit(
        "Missing dependency.\n  Run: pip install playwright && playwright install chromium"
    )

# ── Pillow ────────────────────────────────────────────────────────────────────
try:
    from PIL import Image
except ImportError:
    sys.exit("Missing dependency.\n  Run: pip install pillow")


# ── Lazy imports for optional backends ───────────────────────────────────────
def _import_groq():
    try:
        from groq import Groq

        return Groq
    except ImportError:
        sys.exit("Missing dependency for Groq backend.\n  Run: pip install groq")


def _import_ollama():
    try:
        import ollama

        return ollama
    except ImportError:
        sys.exit("Missing dependency for Ollama backend.\n  Run: pip install ollama")


# ── Constants ─────────────────────────────────────────────────────────────────
SCREENSHOT_WIDTH = 1440
SCREENSHOT_HEIGHT = 900

# Groq: open-source Meta Llama vision models served free via Groq Cloud
GROQ_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# Alternative Groq models:
#   "llama-3.2-11b-vision-preview"   – lighter, very fast
#   "llama-3.2-90b-vision-preview"   – larger, more accurate

# Ollama: locally running model
# https://benchmarking.nanonets.com/
# BAD RESULTS:
# OLLAMA_DEFAULT_MODEL = "llava:13b" #  I'm unable to view images or perform tasks that involve visual analysis of webpages. However, if you provide me with the query and any relevant data points, I can help you structure them into a Markdown table format.
# OLLAMA_DEFAULT_MODEL="gemma3:12b" # bad results
# OLLAMA_DEFAULT_MODEL="minicpm-v:8b" # bad results

# GOOD RESULTS:
# UNTESTED RESULTS:
# OLLAMA_DEFAULT_MODEL = "llama3.2-vision"
# OLLAMA_DEFAULT_MODEL="granite3.2-vision:2b" # Takes long time
# OLLAMA_DEFAULT_MODEL = "qwen3-vl:8b"  # Takes a while
# OLLAMA_DEFAULT_MODEL = "glm-ocr:bf16"
OLLAMA_DEFAULT_MODEL = "qwen3-vl:235b-cloud"
# OLLAMA_DEFAULT_MODEL = "gpt-oss:120b-cloud"
# OLLAMA_DEFAULT_MODEL = "gpt-oss:120b"
# OLLAMA_DEFAULT_MODEL = "qwen3.5:397b-cloud"
# InternVL2-8B
# qwen2.5-vl-72b-instruct
# mistral-small-3.1-24b-instruct
# llama-4-maverick(400B-A17B)
# • Models like Llama 3.2-11B-Vision can be used with CPU offload/quantisation if needed.
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
# Alternative Ollama models:
#   "llava"        – classic multimodal (~4 GB RAM)
#   "llava:13b"    – more accurate     (~8 GB RAM)
#   "moondream"    – tiny              (~2 GB RAM)
#   "bakllava"     – Mistral + LLaVA

# Shared extraction prompt for all backends
SYSTEM_PROMPT = (
    "You are a precise information-extraction assistant.\n"
    "You are given a screenshot of information and a user query.\n\n"
    "Your job:\n"
    "1. Find ONLY the information directly relevant to the query.\n"
    "2. Reproduce tables as proper Markdown tables "
    "   (| header | cols | with a --- separator row).\n"
    "3. Reproduce other relevant information as proper Markdown format."
    "   (| header | cols | with a --- separator row).\n"
    "4. Use appropriate Markdown: ## headings, **bold** for labels, "
    "   bullet lists when needed.\n"
    "5. Ignore navigation bars, ads, site headers, footers, cookie banners, "
    "   'next match' widgets, subscription prompts, social media buttons, "
    "   and anything else NOT directly related to the query.\n"
    "6. If multiple sections match the query, include all with clear Markdown headings.\n"
    "7. Output ONLY the extracted Markdown — no preamble, no disclaimers, no filler text."
)

DEVICES = {
    # --- Desktop ---
    "desktop_4k": {
        "width": 3840,
        "height": 2160,
        "deviceScaleFactor": 1.0,
        "mobile": False,
        "userAgent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "platform": "Linux",
    },
    "desktop_1920": {
        "width": 1920,
        "height": 1080,
        "deviceScaleFactor": 1.0,
        "mobile": False,
        "userAgent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "platform": "Linux",
    },
    # --- Tablet ---
    "ipad_pro": {
        "width": 1024,
        "height": 1366,
        "deviceScaleFactor": 2.0,
        "mobile": False,
        "userAgent": "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
        "platform": "iPad",
    },
    # --- Mobile ---
    "iphone_14_pro_max": {
        "width": 430,
        "height": 932,
        "deviceScaleFactor": 3.0,
        "mobile": True,
        "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
        "platform": "iPhone",
    },
    "samsung_galaxy_s23": {
        "width": 393,
        "height": 851,
        "deviceScaleFactor": 3.0,
        "mobile": True,
        "userAgent": "Mozilla/5.0 (Linux; Android 13; SM-S911B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        "platform": "Android",
    },
    "pixel_7": {
        "width": 412,
        "height": 915,
        "deviceScaleFactor": 2.625,
        "mobile": True,
        "userAgent": "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        "platform": "Android",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Full-page screenshot via Playwright
# ─────────────────────────────────────────────────────────────────────────────
def emulate_device(driver, device_name: str):
    """Apply full Chrome DevTools-style device emulation after driver.get()"""
    d = DEVICES[device_name]

    driver.execute_cdp_cmd(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": d["width"],
            "height": d["height"],
            "deviceScaleFactor": d["deviceScaleFactor"],
            "mobile": d["mobile"],
            "screenWidth": d["width"],
            "screenHeight": d["height"],
            "positionX": 0,
            "positionY": 0,
        },
    )

    driver.execute_cdp_cmd(
        "Network.setUserAgentOverride",
        {
            "userAgent": d["userAgent"],
            "platform": d["platform"],
        },
    )

    driver.execute_cdp_cmd(
        "Emulation.setTouchEmulationEnabled",
        {
            "enabled": True,
            "maxTouchPoints": 5,
        },
    )

    print(
        f"[✓] Emulating: {device_name} ({d['width']}×{d['height']} @{d['deviceScaleFactor']}x)"
    )


def capture_screenshot_desktop(url: str, output_path: str) -> str:
    """
    Launch headless Chromium, load `url`, scroll to trigger lazy content,
    and save a full-page PNG to `output_path`.
    """
    print(f"[1/3] Capturing screenshot: {url}")

    chrome_options = ChromeOptions()

    # --- WSL-specific flags ---
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    # chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])

    # Start with a wide viewport so page renders at desktop width
    # Height will be overridden dynamically after measuring the full page
    # chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--window-size=3840,2160")

    # --- Force-factor: increases CSS pixel density so text stays large and sharp ---
    # 2.0 = retina/HiDPI equivalent. Lower to 1.5 if file sizes get too large.
    # chrome_options.add_argument("--force-device-scale-factor=2.0")
    chrome_options.add_argument("--force-device-scale-factor=1.0")

    chrome_service = ChromeService()
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    try:
        driver.get(url)

        # --- Step 1: Let the initial page load settle ---
        time.sleep(2)

        # --- Step 2: Scroll down gradually to trigger lazy-loaded content ---
        # Many pages (e.g. sports stats, news feeds) only render content when scrolled into view
        scroll_pause = 0.4  # seconds between each scroll step
        scroll_step = 800  # pixels per scroll step

        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll down in steps rather than one jump — triggers more lazy loaders
            current_pos = 0
            while current_pos < last_height:
                driver.execute_script(f"window.scrollTo(0, {current_pos});")
                time.sleep(scroll_pause)
                current_pos += scroll_step

            # Wait for any newly loaded content to render
            time.sleep(1.5)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break  # No more content loaded — we've reached the true bottom
            last_height = new_height

        # --- Step 3: Scroll back to top so nothing is cut off ---
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)

        # --- Step 4: Resize the browser window to the FULL page height ---
        # This is the key step — Selenium only screenshots the current viewport
        # unless you expand the window to match total page dimensions
        total_width = driver.execute_script("return document.body.scrollWidth")
        total_height = driver.execute_script("return document.body.scrollHeight")

        # Add small buffer for any floating elements or sticky footers
        driver.set_window_size(total_width + 100, total_height + 200)
        time.sleep(0.5)  # Let layout reflow after resize

        # --- Step 5: Take the screenshot ---
        screenshot = driver.get_screenshot_as_png()

        with open(output_path, "wb") as file:
            file.write(screenshot)

        print(f"[✓] Full-page screenshot saved → {output_path}")
        print(f"    Page dimensions: {total_width}px wide × {total_height}px tall")

    finally:
        driver.quit()

    return output_path


def capture_screenshot_mobile(url: str, output_path: str) -> str:
    """
    Launch headless Chromium, load `url`, scroll to trigger lazy content,
    and save a full-page PNG to `output_path`.
    """
    print(f"{datetime.now()},[1/3] Capturing screenshot: {url}")

    chrome_options = ChromeOptions()

    # --- WSL-specific flags ---
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    # chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])

    # Start with a wide viewport so page renders at desktop width
    # Height will be overridden dynamically after measuring the full page
    # chrome_options.add_argument("--window-size=3840,2160")
    # chrome_options.add_argument("--window-size=3840,2160")

    # --- Force-factor: increases CSS pixel density so text stays large and sharp ---
    # 2.0 = retina/HiDPI equivalent. Lower to 1.5 if file sizes get too large.
    # chrome_options.add_argument("--force-device-scale-factor=1.0")

    chrome_service = ChromeService()
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    devices = [
        "iphone_14_pro_max",
        "ipad_pro",
        "samsung_galaxy_s23",
        "pixel_7",
        "desktop_4k",
        "desktop_1920",
    ]
    device_used = devices[0]
    try:
        driver.get(url)
        emulate_device(driver, device_used)

        # Refresh so the site re-renders fully with the emulated device
        driver.refresh()
        time.sleep(3)

        # --- Scroll to trigger all lazy-loaded content ---
        scroll_pause = 0.4
        scroll_step = 600
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            current_pos = 0
            while current_pos < last_height:
                driver.execute_script(f"window.scrollTo(0, {current_pos});")
                time.sleep(scroll_pause)
                current_pos += scroll_step

            time.sleep(1.5)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # --- Remove ad containers ---
        driver.execute_script(
            """
            const adSelectors = [
                'iframe',
                '[id*="ad"]',       '[class*="ad"]',
                '[id*="banner"]',   '[class*="banner"]',
                '[id*="sponsor"]',  '[class*="sponsor"]',
                '.advertisement',   '.adsbygoogle',
                'ins.adsbygoogle'
            ];
            adSelectors.forEach(sel => {
                document.querySelectorAll(sel).forEach(el => el.remove());
            });
        """
        )

        # --- Scroll back to top ---
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)

        # --- Measure true full page dimensions ---
        full_width = driver.execute_script(
            """
            return Math.max(
                document.body.scrollWidth,
                document.body.offsetWidth,
                document.documentElement.scrollWidth,
                document.documentElement.offsetWidth,
                document.documentElement.clientWidth
            );
        """
        )
        full_height = driver.execute_script(
            """
            return Math.max(
                document.body.scrollHeight,
                document.body.offsetHeight,
                document.documentElement.scrollHeight,
                document.documentElement.offsetHeight,
                document.documentElement.clientHeight
            );
        """
        )

        print(f"    Full page size: {full_width}px × {full_height}px")

        # --- KEY FIX: Use CDP override again instead of set_window_size ---
        # set_window_size() conflicts with CDP emulation and causes clipping.
        # Re-applying setDeviceMetricsOverride with the full page height
        # tells the emulator to expand to the entire page without losing device context.

        d = DEVICES[device_used]
        driver.execute_cdp_cmd(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": d["width"],  # keep original device width
                "height": full_height,  # expand to full page height
                "deviceScaleFactor": d["deviceScaleFactor"],
                "mobile": d["mobile"],
                "screenWidth": d["width"],
                "screenHeight": full_height,
                "positionX": 0,
                "positionY": 0,
            },
        )
        time.sleep(0.5)

        # --- Take the screenshot ---
        screenshot = driver.get_screenshot_as_png()
        with open(f"{output_path}_{device_used}.png", "wb") as file:
            file.write(screenshot)

        print(f"[✓] Full-page screenshot saved → {output_path}")

    finally:
        driver.quit()

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Resize to stay within model upload limits
# ─────────────────────────────────────────────────────────────────────────────


def prepare_image(path: str, max_height: int = 6000, max_mb: float = 4.0) -> str:
    """
    Resize proportionally if height > max_height OR file > max_mb MB.
    Llama / LLaVA vision models handle roughly up to 4 MB / 6000 px well.
    Returns the (possibly new) file path.
    """
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    with Image.open(path) as img:
        w, h = img.size

    if h <= max_height and size_mb <= max_mb:
        return path  # nothing to do

    with Image.open(path) as img:
        ratio = min(max_height / h, 1.0)
        if size_mb > max_mb:
            ratio = min(ratio, (max_mb / size_mb) ** 0.5)
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        tmp = tempfile.mktemp(suffix=".png")
        resized.save(tmp, "PNG", optimize=True)

    new_mb = Path(tmp).stat().st_size / (1024 * 1024)
    print(f"    Resized: {w}x{h} -> {new_w}x{new_h}  ({new_mb:.2f} MB)")
    return tmp


def _to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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

    Free-tier Groq models with vision support (pick via --groq-model):
        meta-llama/llama-4-scout-17b-16e-instruct  (latest, default)
        llama-3.2-11b-vision-preview               (lighter, fastest)
        llama-3.2-90b-vision-preview               (most accurate)
    """
    Groq = _import_groq()
    client = Groq(api_key=api_key)

    print(f"{datetime.now()}, [2/3] Querying Groq ({model})...")
    b64 = _to_base64(image_path)

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
                            "Extract the relevant information from this image."
                        ),
                    },
                ],
            },
        ],
        temperature=0.1,  # Low = factual / exact output
        max_tokens=4096,  # Increase if you have a lot of content to extract (Groq supports up to 32k)
        # max_tokens=8192,  # Increase if you have a lot of content to extract (Groq supports up to 32k)
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b – Ollama backend  (fully local, 100% private)
# ─────────────────────────────────────────────────────────────────────────────


def query_ollama(
    image_path: str,
    query: str,
    model: str = OLLAMA_DEFAULT_MODEL,
    host: str = OLLAMA_DEFAULT_HOST,
) -> str:
    """
    Send the screenshot to a locally running Ollama vision model.

    Pull a model first, e.g.:
        ollama pull llama3.2-vision   (recommended, default)
        ollama pull llava
        ollama pull moondream
    """
    ollama = _import_ollama()

    if host != OLLAMA_DEFAULT_HOST:
        os.environ["OLLAMA_HOST"] = host

    print(f"{datetime.now()}, [2/3] Querying Ollama (model={model}, host={host})...")
    b64 = _to_base64(image_path)

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Query: {query}\n\n"
                    "Extract the relevant information from the attached webpage screenshot."
                ),
                "images": [b64],
            }
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]

def query_ollama_cloud(
    image_path: str,
    query: str,
    model: str = OLLAMA_DEFAULT_MODEL,
    host: str = OLLAMA_DEFAULT_HOST,
) -> str:
    """
    Send the screenshot to a locally running Ollama vision model.

    Pull a model first, e.g.:
        ollama pull llama3.2-vision   (recommended, default)
        ollama pull llava
        ollama pull moondream
    """
    import os
    from ollama import Client

    client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    )

    if host != OLLAMA_DEFAULT_HOST:
        os.environ["OLLAMA_HOST"] = host

    print(f"{datetime.now()}, [2/3] Querying Ollama (model={model}, host={host})...")
    b64 = _to_base64(image_path)
    messages=[
            {
                "role": "user",
                "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Query: {query}\n\n"
                    "Extract the relevant information from the attached webpage screenshot."
                ),
                "images": [b64],
            }
        ]

    for part in client.chat(model, messages=messages, stream=True):
        print(part['message']['content'], end='', flush=True)
    # return response["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def extract_from_webpage(
    url: str,
    query: str,
    backend: str = "ollama",
    api_key: str | None = None,
    groq_model: str = GROQ_DEFAULT_MODEL,
    ollama_model: str = OLLAMA_DEFAULT_MODEL,
    ollama_host: str = OLLAMA_DEFAULT_HOST,
    keep_screenshot: bool = False,
    prepare_image_flg: bool = False,
) -> str:
    """
    Main entry point.

    Args:
        url:             Webpage URL to screenshot and analyse.
        query:           What to extract (natural language).
        backend:         "groq" (default, free cloud) or "ollama" (local).
        api_key:         Groq API key – falls back to GROQ_API_KEY env var.
        groq_model:      Override the Groq model name.
        ollama_model:    Override the Ollama model name.
        ollama_host:     Override the Ollama server URL.
        keep_screenshot: If True, keep the temp PNG on disk.

    Returns:
        Extracted content as a Markdown string.
    """
    backend = backend.lower()
    if backend not in ("groq", "ollama"):
        sys.exit(f"Unknown backend '{backend}'. Choose 'groq' or 'ollama'.")

    if backend == "groq":
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            sys.exit(
                "No Groq API key found.\n"
                "  Get a free key : https://console.groq.com  (sign up -> API Keys)\n"
                "  Set env var    : export GROQ_API_KEY=your_key\n"
                "  Or pass        : --api-key your_key"
            )

    # tmp_png_fp   = tempfile.mktemp(suffix="_webpage.png")
    tmp_png_fp = "data/webpage_analyser/v6/images/webpage.png"
    ready_img = tmp_png_fp

    # try:
    if not os.path.exists(tmp_png_fp):
        # capture_screenshot_mobile(url, tmp_png_fp)
        capture_screenshot_desktop(url, tmp_png_fp)
    if prepare_image_flg is True:
        print("Preparing image (resizing/compressing to fit model limits)...")
        ready_img = prepare_image(tmp_png_fp)
    ready_img = tmp_png_fp

    if backend == "groq":
        markdown = query_groq(ready_img, query, api_key, model=groq_model)
    elif backend == "ollama":
        markdown = query_ollama(
            ready_img, query, model=ollama_model, host=ollama_host
        )
    elif backend == "ollama_cloud":
        markdown = query_ollama_cloud(
            ready_img, query, model=ollama_model, host=ollama_host
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if not keep_screenshot:
        for p in {tmp_png_fp, ready_img}:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    print("{datetime.now()}, [3/3] Extraction complete.\n")
    return markdown


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(
    url_input: str,
    query_input: str,
    output_file: str,
    keep_screenshot: bool = False,
    prepare_image_flg: bool = False,
) -> None:

    result = extract_from_webpage(
        url=url_input,
        query=query_input,
        backend="ollama",
        api_key=None,
        groq_model=None,
        ollama_model=OLLAMA_DEFAULT_MODEL,
        ollama_host=OLLAMA_DEFAULT_HOST,
        keep_screenshot=keep_screenshot,
        prepare_image_flg=prepare_image_flg,
    )

    print("─" * 70)
    print(result)
    print("─" * 70)

    if output_file:
        Path(output_file).write_text(result, encoding="utf-8")
        print(f"\nSaved -> {output_file}")


if __name__ == "__main__":
    clean_model_name = re.sub(r"[^A-Za-z0-9\-\.]", "_", OLLAMA_DEFAULT_MODEL)

    main(
        url_input=os.getenv("DEMO_URL_LINK"),
        query_input=os.getenv("DEMO_URL_QUERY"),
        output_file=f"data/webpage_analyser/v6/demo_output_free_image_{clean_model_name}.md",
        keep_screenshot=True,
        prepare_image_flg=True,
    )
