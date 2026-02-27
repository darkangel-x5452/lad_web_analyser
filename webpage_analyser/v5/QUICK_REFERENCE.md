# 🚀 Quick Reference Guide

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt --break-system-packages
playwright install chromium

# 2. Set up API key (choose one)
export ANTHROPIC_API_KEY='your-key'  # For Claude
export GOOGLE_API_KEY='your-key'     # For Gemini
# OR install Ollama for local model
```

## Basic Usage

```python
from webpage_analyzer import WebpageAnalyzer

# Initialize
analyzer = WebpageAnalyzer(model="claude")  # or "gemini" or "ollama"

# Analyze webpage
result = analyzer.analyze_webpage(
    url="https://example.com",
    query="Extract the pricing table"
)

print(result)
```

## Common Queries

### Sports
```python
"Get the team statistics including wins, losses, and points per game"
"Extract the player roster with names and positions"
"Show the last 5 game results"
```

### E-commerce
```python
"Extract product specifications from the main content"
"Get the pricing comparison table"
"Show customer ratings and review summary"
```

### Financial
```python
"Extract quarterly earnings from the table"
"Get current stock price and key metrics"
"Show year-over-year revenue comparison"
```

## Model Selection

| Model | Use When | Free Tier |
|-------|----------|-----------|
| **Claude** | Need highest accuracy | Limited |
| **Gemini** | High volume, fast | 1,500/day |
| **Ollama** | Privacy, unlimited | Unlimited |

## Files

- `webpage_analyzer.py` - Main script
- `example_usage.py` - 9 example scenarios
- `test_setup.py` - Verify installation
- `README.md` - Full documentation
- `MODEL_COMPARISON.md` - Detailed model info

## Quick Commands

```bash
# Test setup
python test_setup.py

# Run main example
python webpage_analyzer.py

# Try different examples
python example_usage.py

# Run setup script
bash setup.sh
```

## Troubleshooting

**"API key not set"**
```bash
export ANTHROPIC_API_KEY='your-key'
# or
export GOOGLE_API_KEY='your-key'
```

**"Playwright not found"**
```bash
pip install playwright --break-system-packages
playwright install chromium
```

**"Screenshot is blank"**
- Increase wait_time parameter
- Check if site blocks bots
- Try full_page=False

## Best Practices

1. ✅ Be specific in queries
2. ✅ Mention exact locations (e.g., "main content area")
3. ✅ Request specific format (markdown, table, list)
4. ✅ Increase wait_time for slow sites
5. ✅ Use full_page=True for content below fold

## Examples

```python
# Sports stats
analyzer.analyze_webpage(
    "https://www.nba.com/stats/team/1610612747",
    "Get Lakers record and PPG"
)

# Product specs
analyzer.analyze_webpage(
    "https://www.apple.com/iphone/specs/",
    "Extract display and camera specs"
)

# Financial data
analyzer.analyze_webpage(
    "https://finance.yahoo.com/quote/AAPL",
    "Get current price and market cap"
)
```

## Get Help

- Full docs: `README.md`
- Model comparison: `MODEL_COMPARISON.md`
- Example code: `example_usage.py`
- Test install: `python test_setup.py`
