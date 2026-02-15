# WebVision Analyzer — Free Edition
### 100% free · No credit card · Two AI providers

---

## The Free Stack

| Component | Tool | Cost | Requires |
|-----------|------|------|---------|
| Browser engine | Playwright Chromium | Free / open source | Nothing |
| Image processing | Pillow | Free / open source | Nothing |
| AI Vision (cloud) | Google Gemini 2.5 Flash | Free tier | Google account only |
| AI Vision (local) | Ollama + LLaVA / Llama 3.2 Vision | Free / offline | Disk space |

---

## How Vision AI Understands a Complex Webpage

This is the core question — how does an AI "read" a screenshot packed with tables, icons, navbars, scores, and sidebars?

### Step 1 — The Browser Renders Everything
A plain `requests.get()` call on a modern sports site like SofaScore returns an almost empty HTML shell — all the stats, tables, and scores are injected by JavaScript at runtime. Playwright runs a real Chromium browser engine that executes that JavaScript, waits for the page to fully paint, and then captures a pixel-perfect screenshot. This is the same engine that powers Google Chrome.

### Step 2 — The Image Is Tiled Into Patches
When Claude, Gemini, or LLaVA receives a screenshot, the vision encoder slices it into overlapping 14×14 or 16×16 pixel patches (like a grid of tiny squares). A 1400×3000px image might become 15 000+ patches. Each patch is converted into a high-dimensional numeric vector — an "embedding" that encodes colour, texture, edges, and local shapes.

### Step 3 — A Vision Transformer Processes the Patches
These patches flow through a Vision Transformer (ViT) architecture — the same kind of self-attention mechanism used in language models, but operating on image patches instead of words. Crucially, the model was pre-trained on billions of web pages, screenshots, documents, PDFs, and UI mockups, so it has learned that:
- A horizontal strip at the top with links = navbar
- A narrow vertical column = sidebar
- Alternating-colour rows with a header row = a data table
- Circular team logos next to names = a sports matchup card
- `34–29` in large bold text next to two team names = a score

### Step 4 — Visual Tokens Fuse With Language Tokens
The vision encoder's output is projected into the same embedding space that the language model uses for text tokens. The result: the LLM "reads" a screenshot the same way it reads a document — as a sequence of tokens, some of which happen to encode visual information. This is why you can ask natural-language questions and get grounded, specific answers.

### Step 5 — OCR Comes For Free
Because modern vision models were trained on text-heavy images (PDFs, screenshots, scans), they can read any visible text in any font, size, or orientation. They handle: table cells, axis labels, score cards, player name overlays, tooltip numbers, badge counts, etc.

### Step 6 — The Structured Extraction Prompt
The secret weapon is a well-designed system prompt that tells the model *how* to organise its analysis. Our `EXTRACT_PROMPT` asks the model to output navbars, tables with `|pipe|` formatting, colour-code meanings, matchup data, and a key takeaway. This structures the output so the follow-up Q&A has clean, organised context to reason from.

### Step 7 — Efficient Multi-Turn Q&A
The image is sent **once** in the first message. All follow-up questions are plain text — the model's analysis is already in the conversation history. This makes Q&A fast and token-efficient, even on free-tier rate limits.

---

## Setup Guide

### 1. Install Core Dependencies
```bash
pip install -r requirements_free.txt
playwright install chromium
```

---

### Provider A — Google Gemini 2.5 Flash (Best Free Cloud Option)

**What you get for free (no credit card):**
- 15 requests per minute
- Up to 1 000 requests per day
- 1 million token context window — huge
- Access to Gemini 2.5 Flash (vision + reasoning)

**Get your free API key:**
1. Go to https://aistudio.google.com
2. Sign in with any Google account (Gmail is fine)
3. Click **"Get API key"** in the left sidebar
4. Click **"Create API key"**
5. Copy the key (starts with `AIza...`)

**Set the key:**
```bash
# macOS / Linux
export GEMINI_API_KEY=AIzaSy...

# Windows (Command Prompt)
set GEMINI_API_KEY=AIzaSy...

# Windows (PowerShell)
$env:GEMINI_API_KEY="AIzaSy..."
```

> **Note for EU/UK/Swiss users:** Google's free tier has geographic restrictions for EEA users. If you're affected, use the Ollama local provider instead.

---

### Provider B — Ollama (Fully Local, Zero Account)

**What you get:**
- Completely free, forever
- No API key, no account, no internet after setup
- Runs entirely on your machine — your data never leaves
- Works offline

**Install Ollama:**
- macOS/Linux: `curl -fsSL https://ollama.com/install.sh | sh`
- Windows: Download installer from https://ollama.com

**Pull a vision model (choose one):**

| Command | Size | Quality | Best for |
|---------|------|---------|---------|
| `ollama pull llava` | 4 GB | Good | Most users |
| `ollama pull llama3.2-vision:11b` | 7 GB | Best | Higher accuracy |
| `ollama pull moondream` | 2 GB | Basic | Low RAM machines |
| `ollama pull llava:13b` | 8 GB | Great | Strong GPU machines |

**Start Ollama (if not auto-started):**
```bash
ollama serve
```

---

## Running the App

### Interactive mode
```bash
python webpage_analyzer_free.py
```

### Force a specific provider
```bash
python webpage_analyzer_free.py --gemini
python webpage_analyzer_free.py --ollama
```

### Pass a URL directly
```bash
python webpage_analyzer_free.py --gemini "https://www.example.com"
python webpage_analyzer_free.py --ollama "https://www.example.com"
```

### Run all demo URLs
```bash
python webpage_analyzer_free.py --demo
```

---

## Example Q&A Session

```
  Choose provider [1/2]: 1
  Paste your Gemini API key: AIzaSy...

🌐  Capturing: https://www.example.com
  ↳ Navigating (waiting for JS to render)…
  ↳ Waiting 3.5s for dynamic content…
  ↳ Screenshot ready  1440×4100px  2 312 KB
  ↳ Optimised → 1400×3980px  287 KB (JPEG)

🔍  Extracting with Gemini (gemini-2.5-flash-preview-05-20)…

──────────────────────────────────────────────────────────────────────
  EXTRACTED PAGE CONTEXT
──────────────────────────────────────────────────────────────────────
## Page Type & Purpose
Basketball team comparison page — Brooklyn Nets vs Cleveland Cavaliers

## All Visible Data
| Stat                | Nets   | Cavaliers |
|---------------------|--------|-----------|
| Points per game     | 109.2  | 118.4     |
| Rebounds per game   |  43.1  |  46.8     |
| Assists per game    |  26.3  |  28.5     |
| Win/Loss record     | 14-40  | 35-18     |
...

## Key Takeaway
Cleveland Cavaliers hold a decisive advantage in every statistical
category shown. Their record (35-18) vs Nets (14-40) and a ~9 PPG
scoring edge make them heavy favourites.
──────────────────────────────────────────────────────────────────────

You › Which teams are matched up and who has the advantage?

🤖  Answer:
···················································
The matchup is Brooklyn Nets vs Cleveland Cavaliers.

The Cavaliers have a significant advantage:
• Record: 35-18 (Cavs) vs 14-40 (Nets) — Cavs are nearly 3× better
• Scoring: 118.4 PPG vs 109.2 PPG — 9-point gap per game
• Rebounds and assists both favour Cleveland
• The coloured advantage bars on the page show blue (Cavs) winning
  in almost every category

Verdict: Cleveland is strongly favoured in any matchup with Brooklyn.
···················································

You › quit
```

---

## Troubleshooting

### Page renders blank or incomplete
Some sites (especially SofaScore) have heavy bot-detection. Try:
1. Increase `wait_secs` inside `capture()` from `3.5` to `6.0`
2. Scroll back to top is already done — re-check the saved screenshot in `./screenshots/`

### Gemini 429 Rate Limit
The app automatically waits 62 seconds and retries once. If you hit this frequently, consider using Ollama locally instead.

### Ollama `Connection refused`
Ollama isn't running. Start it:
```bash
ollama serve
```
Then in another terminal:
```bash
python webpage_analyzer_free.py --ollama
```

### Ollama is slow
This is normal for CPU-only inference. An 11B model on a modern laptop takes 60–120 seconds per image analysis. For speed, use `moondream` (2 GB, CPU-friendly) or get a machine with an NVIDIA/Apple Silicon GPU.

### Model not found in Ollama
```bash
ollama pull llava        # or whichever model you want
```

---

## Comparison: Gemini vs Ollama

| | Gemini Free | Ollama Local |
|---|---|---|
| Setup | 5 min (Google account) | 10–20 min (download model) |
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (llava) / ⭐⭐⭐⭐ (llama3.2) |
| Speed | Fast (1–5 s) | Slow on CPU (30–120 s) |
| Privacy | Data sent to Google | Data never leaves your machine |
| Cost | Free (1 000 req/day) | Free forever |
| Internet | Required | Only for initial model download |
| Rate limits | 15 RPM, 1000 RPD | None |
| EU users | Restricted | Works everywhere |
