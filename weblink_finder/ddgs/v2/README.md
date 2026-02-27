# DeepLink Scraper 🔍

Find **specific, deep subpage URLs** that are directly relevant to your query — not just homepage links.

---

## Install

```bash
pip install -r requirements.txt
```

---

## Usage

### Interactive mode (prompted)
```bash
python scraper.py
```

### One-shot CLI
```bash
python scraper.py --query "LA Lakers NBA statistics 2024"
python scraper.py -q "Get the statistics for the team LA Lakers from competition NBA sport Basketball"
python scraper.py -q "Patrick Mahomes NFL passing stats 2023" --max 20
python scraper.py -q "Premier League standings 2024-25" --min-score 30
```

### Export to JSON
```bash
python scraper.py -q "Warriors roster 2024 NBA" --export results.json
```

---

## How it Works

| Stage | What happens |
|-------|-------------|
| **Intent parsing** | Extracts entities, stat keywords, sport/year signals from your natural language query |
| **Query fan-out** | Generates 5–7 targeted DuckDuckGo searches from different angles (stats, roster, schedule, site-targeted) |
| **Relevance scoring** | Each URL is scored 0–100 using: URL depth, keyword hits in path/title/snippet, stat-type signal matches, ID patterns, year matches |
| **Penalties** | Generic domains (Twitter, Reddit, Wikipedia), shallow homepages, tag/search-result pages are penalized or dropped |
| **Dedup + rank** | Results are deduplicated by canonical URL and sorted by score |

---

## Scoring Factors

| Signal | Bonus |
|--------|-------|
| URL path depth ≥ 3 | +25 |
| Query keywords in URL path | +8 per match (max +24) |
| Query keywords in title | +6 per match (max +20) |
| Stat keywords matched | +7 per match (max +21) |
| Strong snippet relevance | +10 |
| URL contains specific numeric ID | +8 |
| URL contains matching year | +6 |
| Homepage / shallow path | −30 |
| Tag / category / search URL | −15 |
| Generic social domain | Score = 0 (skipped) |

---

## Example Queries

```
"Get the statistics for the team LA Lakers from competition NBA, sport Basketball"
"Top scorers Premier League 2024-25 season"
"Aaron Rodgers career passing stats NFL"
"Golden State Warriors 2024 roster and salaries"
"F1 standings constructors championship 2024"
"Novak Djokovic Grand Slam win statistics"
```

---

## Options

```
--query / -q     Search query
--max / -n       Max results returned (default: 12)
--min-score      Minimum relevance score to include (default: 20.0)
--export / -e    Export results to JSON file
--verbose / -v   Show search warnings
```

---

## Notes

- Uses **DuckDuckGo** — completely free, no API key required
- Rate-limited politely (0.4s between searches) to avoid blocks
- Works best with specific, entity-rich queries
- Re-run with `--min-score 10` if you get too few results
