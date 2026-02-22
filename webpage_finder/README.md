# AI Web Searcher 🔍

A Python application that replicates how LLMs (like ChatGPT, Perplexity, Claude) 
find up-to-date information — using **only free tools, no credit card required**.

---

## How LLMs Search the Web (the "RAG pipeline")

```
User Question
     │
     ▼
1. Query Reformulation   — rewrite into search-friendly keywords
     │
     ▼
2. Search Engine Call    — DuckDuckGo / Bing / Google → ranked URLs + snippets
     │
     ▼
3. Web Scraping          — fetch top N pages, extract clean text
     │
     ▼
4. RAG Context Injection — stuff scraped text into LLM's prompt as "sources"
     │
     ▼
5. LLM Answer            — model answers ONLY from the retrieved content
     │
     ▼
6. Response + Citations  — answer shown alongside source URLs
```

This app does **all 6 steps** — locally and for free.

---

## Installation

```bash
pip install ddgs requests beautifulsoup4 sumy nltk
# Optional: for AI-powered answers via Groq (free, no credit card)
pip install groq
```

---

## Usage

### Interactive mode
```bash
python ai_web_searcher.py
```

### Direct query
```bash
python ai_web_searcher.py --query "Nebraska Omaha NCAA Basketball team stats 2025"
python ai_web_searcher.py --query "Which conference does Iowa play in the NCAA basketball competition?"
```

### With Groq LLM (better answers, still free)
```bash
python ai_web_searcher.py \
  --query "What is the current team statistics for Nebraska Omaha in NCAA Basketball?" \
  --groq-key YOUR_GROQ_KEY
```

### See the educational explanation only
```bash
python ai_web_searcher.py --explain
```

### All options
```
--query        Search question (prompted interactively if omitted)
--groq-key     Groq API key for LLM-powered answers
--max-results  How many DDG results to fetch (default: 8)
--max-pages    How many pages to fully scrape (default: 5)
--explain      Print the RAG pipeline explanation and exit
```

---

## Free Tools Used

| Tool | Purpose | Cost |
|---|---|---|
| `duckduckgo_search` | Web search (no API key) | Free forever |
| `requests` | HTTP fetching | Free (stdlib+ ) |
| `beautifulsoup4` | HTML parsing / scraping | Free |
| `sumy` + `nltk` | Local extractive summarisation (LSA) | Free |
| `groq` (optional) | LLM-powered answer via Llama 3 | Free tier, no CC |

### Getting a free Groq API key
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up with email (no credit card required)
3. Create an API key
4. Pass it with `--groq-key YOUR_KEY`

---

## Example Output

```
──────────────────────────────────────────────────────────────────────
  AI WEB SEARCHER  ·  2025-02-19 14:32
──────────────────────────────────────────────────────────────────────

  Query: What is the current team statistics for Nebraska Omaha in NCAA Basketball?

▶ Searching the web with DuckDuckGo …
  Query: "What is the current team statistics for Nebraska Omaha..."

  [1] Nebraska Omaha Mavericks Basketball - ESPN
       https://www.espn.com/mens-college-basketball/team/_/id/2437
       Nebraska Omaha Mavericks Men's College Basketball — stats, scores ...

  ...

▶ Scraping top 5 pages …
  Fetching: https://www.espn.com/...
    → 6,412 chars extracted

▶ Generating answer …

──────────────────────────────────────────────────────────────────────
  ANSWER
──────────────────────────────────────────────────────────────────────

  • Nebraska Omaha Mavericks Basketball - ESPN
    Nebraska Omaha is currently ranked ... in the Summit League with a record of ...
    Their scoring average is X.X PPG and they allow Y.Y PPG defensively ...
```
