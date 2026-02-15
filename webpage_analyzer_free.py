#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║          WebVision Analyzer — 100% Free, No Credit Card             ║
╠══════════════════════════════════════════════════════════════════════╣
║  PROVIDER 1 ▸ Google Gemini 2.5 Flash                               ║
║    • Free tier via aistudio.google.com (just a Google account)      ║
║    • 15 req/min · up to xx req/day · 1M token context            ║
║    • Best quality of the free options                                ║
║                                                                      ║
║  PROVIDER 2 ▸ Ollama (fully local, zero cloud, zero account)        ║
║    • Runs on your machine — no internet required after setup         ║
║    • Supports: llava · llava:13b · llama3.2-vision:11b · moondream  ║
║    • Best for privacy-sensitive data                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  BROWSER  ▸ Playwright Chromium (open source, free)                 ║
║  IMAGING  ▸ Pillow (open source, free)                              ║
╚══════════════════════════════════════════════════════════════════════╝

Quick Start:
  1. pip install -r requirements_free.txt
  2. playwright install chromium

  For Gemini provider:
    • Go to https://aistudio.google.com → Get API key (no card needed)
    • export GEMINI_API_KEY=AIza...

  For Ollama provider (fully local):
    • Install from https://ollama.com
    • ollama pull llava          (4 GB, recommended)
    • ollama pull llama3.2-vision:11b   (7 GB, better quality)

  Run:
    python webpage_analyzer_free.py
    python webpage_analyzer_free.py --gemini <URL>
    python webpage_analyzer_free.py --ollama <URL>
    python webpage_analyzer_free.py --demo
"""

import asyncio
import base64
import io
import json
import os
import sys
import textwrap
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.parsers.expat import model

# ── Dependency guards ─────────────────────────────────────────────────────────

def _check(pkg, install_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        name = install_name or pkg
        print(f"\n  [MISSING] {pkg}  →  pip install {name}")
        return None

_playwright_ok  = _check("playwright",           "playwright")
_pil_ok         = _check("PIL",                  "pillow")

# These are optional — only needed for the chosen provider
_genai_ok       = _check("google.generativeai",  "google-generativeai")
_ollama_ok      = _check("ollama",               "ollama")

if not _playwright_ok or not _pil_ok:
    print("\n  Run:  pip install -r requirements_free.txt")
    print("  Then: playwright install chromium\n")
    sys.exit(1)

from playwright.async_api import async_playwright  # noqa: E402
from PIL import Image                               # noqa: E402

# ── Terminal colours ───────────────────────────────────────────────────────────

R  = "\033[0m"      # reset
B  = "\033[1m"      # bold
DM = "\033[2m"      # dim
CY = "\033[36m"     # cyan
GR = "\033[32m"     # green
YL = "\033[33m"     # yellow
MG = "\033[35m"     # magenta
RD = "\033[31m"     # red
BL = "\033[34m"     # blue

def c(text, col): return f"{col}{text}{R}"
def hr(ch="─", n=70): return c(ch * n, DM)
def hdr(text, col=CY): return f"\n{c('  ' + text, B + col)}\n{hr()}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — WEBPAGE CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageCapture:
    url:        str
    png_bytes:  bytes
    title:      str
    width:      int
    height:     int
    saved_path: Optional[Path] = None


async def capture(
    url:          str,
    vp_w:         int   = 1440,
    vp_h:         int   = 900,
    full_page:    bool  = True,
    wait_secs:    float = 3.5,
    save_dir:     Optional[Path] = None,
) -> PageCapture:
    """
    Render a URL with a headless Chromium browser and return a full-page PNG.

    Why Playwright works for JS-heavy sports sites
    ──────────────────────────────────────────────
    Sites can be React/Vue SPAs — the initial HTML is
    nearly empty; all content is injected by JavaScript after load. A plain
    `requests.get()` would return a blank shell. Playwright runs a real
    Chromium engine, executes all JS, waits for the network to go quiet
    (networkidle), then optionally sleeps extra seconds for animations/lazy
    data to finish painting before taking the screenshot.

    The scroll-to-bottom trick triggers lazy-image observers and infinite-
    scroll loaders, ensuring below-the-fold tables and stats are rendered
    before we snap the picture.
    """
    print(c(f"\n🌐  Capturing: {url}", CY))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
            ],
        )
        ctx = await browser.new_context(
            viewport={"width": vp_w, "height": vp_h},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )

        # Skip heavy fonts that don't affect visual content
        await ctx.route("**/*.{woff,woff2,otf,ttf,eot}", lambda r: r.abort())

        page = await ctx.new_page()

        print(c("  ↳ Navigating (waiting for JS to render)…", DM))
        try:
            await page.goto(url, wait_until="networkidle", timeout=50_000)
        except Exception:
            try:
                await page.goto(url, wait_until="load", timeout=50_000)
            except Exception as e:
                print(c(f"  ↳ Warning: navigation issue — {e}", YL))

        print(c(f"  ↳ Waiting {wait_secs}s for dynamic content…", DM))
        await asyncio.sleep(wait_secs)

        # Scroll to trigger lazy loaders, then return to top
        print(c("  ↳ Scrolling to load lazy content…", DM))
        await page.evaluate("""async () => {
            await new Promise(resolve => {
                let prev = 0;
                const t = setInterval(() => {
                    window.scrollBy(0, 900);
                    if (document.body.scrollHeight === prev) {
                        clearInterval(t); resolve();
                    }
                    prev = document.body.scrollHeight;
                }, 250);
            });
        }""")
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.4)

        title     = await page.title()
        png_bytes = await page.screenshot(full_page=full_page, type="png")
        await browser.close()

    img = Image.open(io.BytesIO(png_bytes))
    w, h = img.size
    kb   = len(png_bytes) // 1024
    print(c(f"  ↳ Screenshot ready  {w}×{h}px  {kb} KB", GR))
    print(c(f"  ↳ Title: {title or '(no title)' }", DM))

    saved = None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_"
                       for ch in url[8:])[:60]
        saved = save_dir / f"{safe}_{int(time.time())}.png"
        saved.write_bytes(png_bytes)
        print(c(f"  ↳ Saved → {saved}", DM))

    return PageCapture(url=url, png_bytes=png_bytes, title=title,
                       width=w, height=h, saved_path=saved)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — IMAGE OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════

def optimise_image(
    png_bytes:  bytes,
    max_w:      int = 1400,   # Gemini and LLaVA both read well at this width
    max_h:      int = 7000,   # Cap height for very long pages
    quality:    int = 80,
) -> tuple[bytes, str]:
    """
    Resize + JPEG-compress the screenshot.

    Token cost math (why this matters)
    ───────────────────────────────────
    Gemini charges per pixel tile.  A raw 1440×8000 PNG screenshot at
    full resolution can cost thousands of tokens.  Resizing to ≤1400px
    wide and JPEG-compressing at 80% quality gives near-identical visual
    fidelity for LLM analysis at a fraction of the cost — and also keeps
    Ollama's context window from overflowing.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = img.size

    if w > max_w:
        img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
        w, h = img.size

    if h > max_h:
        img = img.crop((0, 0, w, max_h))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    fw, fh = Image.open(io.BytesIO(data)).size
    print(c(f"  ↳ Optimised → {fw}×{fh}px  {len(data)//1024} KB (JPEG)", DM))
    return data, "image/jpeg"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SHARED PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = """
You are an expert at reading and understanding web page screenshots.
Analyse this screenshot exhaustively and extract every piece of visible information.

Structure your response exactly like this:

## Page Type & Purpose
What kind of page is this? (sports match, standings, comparison, news, product, etc.)

## Navigation & Layout
- Navbar / header items visible
- Sidebar content (if present)
- Footer / footnotes

## All Visible Data (tables, stats, scores)
Extract EVERY number, label, and table cell you can see.
Format tables with | column | separators.
Include all rows and columns even if you think they are less important.

## Teams / Players / Entities
For sports pages:
- List every team name and player name visible
- Their stats, records, current form, win/loss data
- Any rankings, positions, or seedings shown

## Visual Indicators & Icons
- Coloured bars (what do the colours mean?)
- Up/down arrows (trends)
- Stars, shields, logos, badges
- Green = positive? Red = negative? Blue = neutral?

## Matchup Analysis (if applicable)
- Which two teams or entities are being compared?
- Which side has the statistical advantage and in which categories?
- Any injury flags, missing players, or roster notes visible?

## Key Takeaway
One paragraph summarising the most important information on this page.

Be thorough — every cell, every number, every label matters.
""".strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — GEMINI PROVIDER (Free cloud — just a Google account)
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiProvider:
    """
    Uses Google Gemini 2.5 Flash via the free AI Studio API.

    Free tier limits (as of early 2026):
      • 15 requests per minute
      • Up to 1 000 requests per day
      • 1 million token context window
      • No credit card required — just a Google account
      • Get key: https://aistudio.google.com → 'Get API key'

    Sends the image only on the first (extraction) call.
    Follow-up Q&A sends text only — cheaper and faster.
    """
    # MODEL = "gemini-2.5-flash-preview-05-20"   # best free vision model
    # MODEL = "gemini-2.5-pro"
    # MODEL = "gemini-2.5-flash-image"   # best free vision model
    MODEL = "gemini-2.5-flash"   # best free vision model

    def __init__(self, api_key: str):
        if not _genai_ok:
            print(c("  Run: pip install google-generativeai", RD))
            sys.exit(1)
        import google.generativeai as genai
        from google.generativeai import models
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(self.MODEL)
        self._history: list[dict] = []
        self._initial_analysis = ""
        for model in genai.list_models():
            print(model.name)

    def _img_part(self, img_bytes: bytes, media_type: str):
        return {
            "inline_data": {
                "mime_type": media_type,
                "data": base64.standard_b64encode(img_bytes).decode(),
            }
        }

    def extract(self, img_bytes: bytes, media_type: str) -> str:
        """First call: screenshot + extraction prompt → analysis text."""
        print(c(f"\n🔍  Extracting with Gemini ({self.MODEL})…", YL))
        try:
            resp = self._model.generate_content(
                [self._img_part(img_bytes, media_type), EXTRACT_PROMPT]
            )
            self._initial_analysis = resp.text
        except Exception as e:
            return f"[Gemini error during extraction: {e}]"

        # Seed history for multi-turn Q&A (image included in turn 1)
        self._history = [
            {"role": "user",  "parts": [
                self._img_part(img_bytes, media_type), EXTRACT_PROMPT
            ]},
            {"role": "model", "parts": [self._initial_analysis]},
        ]
        return self._initial_analysis

    def ask(self, question: str) -> str:
        """Follow-up Q&A — text only (image already in history)."""
        self._history.append({"role": "user", "parts": [question]})
        try:
            chat = self._model.start_chat(history=self._history[:-1])
            resp = chat.send_message(question)
            answer = resp.text
        except Exception as e:
            # Rate limit? Wait and retry once
            if "429" in str(e) or "quota" in str(e).lower():
                print(c("  Rate limit hit — waiting 60 s…", YL))
                time.sleep(62)
                try:
                    chat = self._model.start_chat(history=self._history[:-1])
                    resp = chat.send_message(question)
                    answer = resp.text
                except Exception as e2:
                    answer = f"[Error: {e2}]"
            else:
                answer = f"[Gemini error: {e}]"
        self._history.append({"role": "model", "parts": [answer]})
        return answer


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — OLLAMA PROVIDER (Fully local, zero cost, zero account)
# ═══════════════════════════════════════════════════════════════════════════════

class OllamaProvider:
    """
    Uses a locally running Ollama vision model.

    Completely free — no API key, no account, no internet after setup.
    Best for privacy-sensitive content.

    Setup:
      1. Download Ollama from https://ollama.com (macOS/Linux/Windows)
      2. ollama pull llava                   # 4 GB, good quality
         ollama pull llama3.2-vision:11b     # 7 GB, better quality
         ollama pull moondream               # 2 GB, lightweight

    The app uses the Ollama REST API directly (no SDK needed).
    Base URL: http://localhost:11434

    Vision models receive base64-encoded images in the 'images' field.
    For Q&A we keep a full conversation history and re-send with each turn
    (Ollama stateless API — no server-side session).
    """

    BASE_URL = "http://localhost:11434"

    RECOMMENDED_MODELS = [
        ("llava",                "4 GB — good quality, most popular"),
        ("llama3.2-vision:11b",  "7 GB — best quality for this task"),
        ("moondream",            "2 GB — lightweight, basic analysis"),
        ("llava:13b",            "8 GB — stronger than 7b llava"),
    ]

    def __init__(self, model: str = "llava"):
        self.model = model
        self._history: list[dict] = []
        self._initial_analysis = ""
        self._img_b64 = ""          # kept for history re-inclusion
        self._media_type = ""

        if not self._ping():
            print(c("\n  Ollama is not running!", RD))
            print(c("  Start it with:  ollama serve", YL))
            print(c("  Then pull a model:  ollama pull llava\n", YL))
            sys.exit(1)
        print(c(f"  ↳ Ollama is running. Model: {model}", GR))

    def _ping(self) -> bool:
        try:
            req = urllib.request.urlopen(f"{self.BASE_URL}/api/tags",
                                         timeout=3)
            return req.status == 200
        except Exception:
            return False

    def _call(self, messages: list) -> str:
        payload = json.dumps({
            "model":    self.model,
            "messages": messages,
            "stream":   False,
        }).encode()
        req = urllib.request.Request(
            f"{self.BASE_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                return data["message"]["content"]
        except urllib.error.URLError as e:
            return f"[Ollama network error: {e}]"
        except Exception as e:
            return f"[Ollama error: {e}]"

    def extract(self, img_bytes: bytes, media_type: str) -> str:
        """First call: image + extraction prompt."""
        print(c(f"\n🔍  Extracting with Ollama ({self.model})…", YL))
        print(c("  (Local model — may take 30–90 s depending on hardware)", DM))

        self._img_b64    = base64.standard_b64encode(img_bytes).decode()
        self._media_type = media_type

        msg = {
            "role":    "user",
            "content": EXTRACT_PROMPT,
            "images":  [self._img_b64],
        }
        result = self._call([msg])
        self._initial_analysis = result

        # Seed history (Ollama needs the full history each turn)
        self._history = [
            msg,
            {"role": "assistant", "content": result},
        ]
        return result

    def ask(self, question: str) -> str:
        """Follow-up Q&A — sends full history each time (Ollama is stateless)."""
        user_msg = {"role": "user", "content": question}
        messages  = self._history + [user_msg]
        answer    = self._call(messages)
        self._history.append(user_msg)
        self._history.append({"role": "assistant", "content": answer})
        return answer


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — SESSION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Session:
    capture:    PageCapture
    provider:   object            # GeminiProvider or OllamaProvider
    analysis:   str = ""
    img_bytes:  bytes = field(default=b"", repr=False)
    media_type: str = ""

    def __post_init__(self):
        self.img_bytes, self.media_type = optimise_image(self.capture.png_bytes)

    def extract(self) -> str:
        self.analysis = self.provider.extract(self.img_bytes, self.media_type)
        return self.analysis

    def ask(self, q: str) -> str:
        return self.provider.ask(q)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_URLS = os.environ["DEMO_URLS"]

DEMO_QUESTIONS = os.environ["DEMO_QUESTIONS"]


def banner():
    print(f"""
{B}{CY}╔══════════════════════════════════════════════════════════════════════╗
║          WebVision Analyzer  ·  100% Free, No Credit Card           ║
║  Providers: Google Gemini (free cloud) · Ollama (local/offline)     ║
╚══════════════════════════════════════════════════════════════════════╝{R}
{DM}  Capture any webpage → AI extracts all context → you ask questions{R}
""")


def wrap_print(text: str, width: int = 100, indent: str = "  "):
    for line in text.splitlines():
        if len(line) > width:
            for wl in textwrap.wrap(line, width=width,
                                    subsequent_indent=indent + "  "):
                print(indent + wl)
        else:
            print(indent + line)


def print_analysis(text: str):
    print(hdr("EXTRACTED PAGE CONTEXT", GR))
    wrap_print(text)
    print(hr())


def print_answer(text: str):
    print(f"\n{c('🤖  Answer:', B + MG)}")
    print(hr("·"))
    wrap_print(text)
    print(hr("·"))


def print_tip_questions():
    print(f"\n{c('  Example questions you can ask:', DM)}")
    for q in DEMO_QUESTIONS:
        print(c(f"    › {q}", DM))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — PROVIDER SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def choose_provider(force: Optional[str] = None) -> object:
    """
    Interactively or programmatically pick Gemini or Ollama.

    Gemini — best quality, needs a free API key (Google account only)
    Ollama  — no account at all, runs locally, needs ollama installed
    """
    if force == "gemini":
        choice = "1"
    elif force == "ollama":
        choice = "2"
    else:
        print(f"""
{hdr('CHOOSE YOUR AI PROVIDER', BL)}
  {c('1', B+GR)} › {B}Google Gemini 2.5 Flash{R}  (free cloud API — best quality)
     Get a free key at: {c('https://aistudio.google.com', CY)}
     No credit card · just your Google account · 1 000 req/day

  {c('2', B+YL)} › {B}Ollama (local / offline){R}  (100% local — no account needed)
     Install: {c('https://ollama.com', CY)}
     Then run: {c('ollama pull llava', DM)}
     Works offline, great for privacy
""")
        try:
            choice = input(c("  Choose provider [1/2]: ", B + CY)).strip()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

    # ── Gemini ──────────────────────────────────────────────────────────────
    if choice == "1":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            print(c("\n  GEMINI_API_KEY not set.", YL))
            print(c("  Get one free (no card) at: https://aistudio.google.com", DM))
            try:
                api_key = input(c("  Paste your Gemini API key: ", CY)).strip()
            except (KeyboardInterrupt, EOFError):
                sys.exit(0)
        if not api_key:
            print(c("  No key provided — exiting.", RD))
            sys.exit(1)
        return GeminiProvider(api_key)

    # ── Ollama ───────────────────────────────────────────────────────────────
    else:
        print(f"\n{c('  Available vision models (via Ollama):', DM)}")
        for i, (m, desc) in enumerate(OllamaProvider.RECOMMENDED_MODELS, 1):
            print(c(f"    {i}. {m:<30} {desc}", DM))
        print(c("    (press Enter for default: llava)", DM))
        try:
            raw = input(c("  Model name or number [llava]: ", CY)).strip()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(OllamaProvider.RECOMMENDED_MODELS):
                model = OllamaProvider.RECOMMENDED_MODELS[idx][0]
            else:
                model = "llava"
        else:
            model = raw or "llava"

        return OllamaProvider(model)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN INTERACTIVE SESSION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

async def run_session(url: str, provider, save_shots: bool = True):
    """Capture → extract → Q&A loop for one URL."""

    save_dir = Path("data/screenshots") if save_shots else None
    cap      = await capture(url, save_dir=save_dir)
    sess     = Session(capture=cap, provider=provider)
    analysis = sess.extract()
    print_analysis(analysis)

    print(f"""
{c('  ASK ANYTHING ABOUT THIS PAGE', B + CY)}
  Type a question and press Enter.
  Commands: {c('new', YL)} — load new URL  |  {c('demo', YL)} — show examples  |  {c('quit', YL)} — exit
""")
    print_tip_questions()

    while True:
        print()
        try:
            user_q = input(c("You › ", B + CY)).strip()
        except (KeyboardInterrupt, EOFError):
            print(c("\nGoodbye!", GR))
            sys.exit(0)

        if not user_q:
            continue
        if user_q.lower() in ("quit", "exit", "q"):
            print(c("Goodbye!", GR))
            sys.exit(0)
        if user_q.lower() in ("new", "n"):
            return       # caller prompts for a new URL
        if user_q.lower() in ("demo", "d"):
            print_tip_questions()
            continue

        print(c("  Thinking…", DM))
        answer = sess.ask(user_q)
        print_answer(answer)


async def main():
    banner()

    # ── CLI argument parsing (minimal, no argparse dep) ───────────────────
    args       = sys.argv[1:]
    force_prov = None
    force_urls = None
    demo_mode  = False

    for a in args:
        if a == "--gemini":  force_prov = "gemini"
        elif a == "--ollama": force_prov = "ollama"
        elif a == "--demo":   demo_mode  = True
        elif a.startswith("http"):
            force_urls = [a]

    if demo_mode:
        force_urls = list(DEMO_URLS)

    # ── Choose provider once (shared across all URL sessions) ────────────
    provider = choose_provider(force=force_prov)

    # ── URL loop ──────────────────────────────────────────────────────────
    while True:
        if force_urls:
            url = force_urls.pop(0)
            if not force_urls:
                force_urls = None
        else:
            print(f"\n{c('  Example URLs:', DM)}")
            for i, u in enumerate(DEMO_URLS, 1):
                print(c(f"    {i}. {u}", DM))
            try:
                raw = input(
                    c("\n  Enter a URL or 1–4 for an example: ", B + CY)
                ).strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not raw:
                continue
            if raw.isdigit() and 1 <= int(raw) <= len(DEMO_URLS):
                url = DEMO_URLS[int(raw) - 1]
            elif raw.startswith("http"):
                url = raw
            else:
                url = f"https://{raw}"

        try:
            await run_session(url, provider)
        except Exception as e:
            import traceback
            print(c(f"\n  Session error: {e}", RD))
            traceback.print_exc()

        try:
            again = input(c("\n  Analyse another URL? [Y/n]: ", CY)).strip().lower()
        except (KeyboardInterrupt, EOFError):
            break
        if again in ("n", "no"):
            break

    print(c("\nAll done. Screenshots saved in ./screenshots/\n", GR))


if __name__ == "__main__":
    asyncio.run(main())
