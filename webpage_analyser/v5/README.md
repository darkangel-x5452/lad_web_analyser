# 🔍 Webpage Image Analyzer

**Extract precise information from webpages using AI vision models**

This tool captures webpage screenshots and uses free GenAI vision models to extract relevant information while filtering out headers, footers, ads, and other noise.

## 🌟 Features

- **3 Free Vision Models** - Choose between Claude, Gemini, or Ollama
- **Smart Extraction** - Filters out irrelevant content automatically
- **High Accuracy** - Uses state-of-the-art vision models
- **Markdown Output** - Clean, formatted results
- **Full Page Capture** - Screenshots entire scrollable pages
- **Easy to Use** - Simple Python API

## 🎯 Use Cases

- Extract sports statistics from team pages
- Pull pricing tables from comparison sites
- Get product specifications from e-commerce
- Extract financial data from reports
- Capture research data from academic sites

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt --break-system-packages
playwright install chromium
```

### 2. Setup API Keys

Choose ONE of the following models:

#### Option A: Claude (Recommended - Best Accuracy)
```bash
# Get API key from: https://console.anthropic.com/
export ANTHROPIC_API_KEY='your-key-here'
```

#### Option B: Google Gemini (Fast & Generous Free Tier)
```bash
# Get API key from: https://aistudio.google.com/app/apikey
export GOOGLE_API_KEY='your-key-here'
```

#### Option C: Ollama (Completely Free, Runs Locally)
```bash
# Install Ollama: https://ollama.ai/download
ollama serve
ollama pull llava
```

### 3. Run Example

```bash
python webpage_analyzer.py
```

## 📖 Usage

### Basic Example

```python
from webpage_analyzer import WebpageAnalyzer

# Initialize with your preferred model
analyzer = WebpageAnalyzer(model="claude")  # or "gemini" or "ollama"

# Analyze a webpage
url = "https://www.nba.com/stats/team/1610612747"
query = "Get me the LA Lakers team statistics"

result = analyzer.analyze_webpage(url, query)
print(result)
```

### Custom Screenshot Settings

```python
# Capture screenshot with custom settings
analyzer.capture_screenshot(
    url="https://example.com",
    output_path="my_screenshot.png",
    wait_time=3,  # Wait 3 seconds for page load
    full_page=True  # Capture entire scrollable page
)

# Then analyze the screenshot
result = analyzer.analyze_screenshot("my_screenshot.png", "Extract pricing table")
```

### Different Queries

```python
analyzer = WebpageAnalyzer(model="gemini")

# Sports statistics
result = analyzer.analyze_webpage(
    "https://www.espn.com/nfl/team/_/name/sf/san-francisco-49ers",
    "Get me the San Francisco 49ers offensive statistics"
)

# Product pricing
result = analyzer.analyze_webpage(
    "https://www.apple.com/iphone-16/specs/",
    "Extract all iPhone 16 Pro technical specifications"
)

# Financial data
result = analyzer.analyze_webpage(
    "https://finance.yahoo.com/quote/AAPL",
    "Get current stock price and key statistics"
)
```

## 🤖 Model Comparison

| Model | Accuracy | Speed | Cost | Setup Difficulty |
|-------|----------|-------|------|------------------|
| **Claude Sonnet 4** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free tier + paid | Easy |
| **Gemini 2.0 Flash** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Very generous free | Easy |
| **Ollama LLaVA** | ⭐⭐⭐ | ⭐⭐⭐ | Completely free | Medium |

### Recommendations:

- **Best Accuracy**: Claude Sonnet 4 - Excellent at understanding complex layouts
- **Best Free Tier**: Gemini 2.0 Flash - 1,500 requests/day free
- **Completely Free**: Ollama LLaVA - No API limits, runs on your machine

## 🔧 Advanced Usage

### Save Output to File

```python
analyzer = WebpageAnalyzer(model="claude")
result = analyzer.analyze_webpage(url, query)

# Save as markdown
with open("output.md", "w") as f:
    f.write(f"# Analysis Results\n\n")
    f.write(result)
```

### Batch Processing

```python
urls = [
    ("https://site1.com", "Extract data 1"),
    ("https://site2.com", "Extract data 2"),
    ("https://site3.com", "Extract data 3"),
]

analyzer = WebpageAnalyzer(model="gemini")

for url, query in urls:
    result = analyzer.analyze_webpage(url, query, 
                                      screenshot_path=f"screenshot_{urls.index((url, query))}.png")
    print(f"Results for {url}:")
    print(result)
    print("-" * 60)
```

### Error Handling

```python
try:
    analyzer = WebpageAnalyzer(model="claude")
    result = analyzer.analyze_webpage(url, query)
except ValueError as e:
    print(f"Setup error: {e}")
except Exception as e:
    print(f"Analysis error: {e}")
```

## 💡 Tips for Best Results

1. **Be Specific**: "Get the scoring stats table" vs "Get statistics"
2. **Mention Location**: "Get the pricing table in the main content area"
3. **Specify Format**: "Extract as a table with columns: name, price, features"
4. **Wait Time**: Increase `wait_time` for slow-loading pages
5. **Full Page**: Use `full_page=True` if data is below the fold

## 🎯 Example Queries

### Sports Statistics
```python
"Extract the current season statistics for the team including wins, losses, points per game"
"Get the player roster with names, positions, and jersey numbers"
"Show me the last 5 game results with scores and dates"
```

### E-commerce
```python
"Extract product specifications including dimensions, weight, and materials"
"Get the pricing comparison table with all plan features"
"Show me customer ratings and review summaries"
```

### Financial Data
```python
"Extract the quarterly earnings data from the table"
"Get current stock price, market cap, and P/E ratio"
"Show me the year-over-year revenue comparison"
```

## 🛠️ Troubleshooting

### "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

### "Playwright installation failed"
```bash
pip install playwright --break-system-packages
playwright install chromium
```

### "Ollama connection refused"
```bash
# Make sure Ollama is running
ollama serve

# In another terminal
ollama pull llava
```

### Screenshot is blank
- Increase `wait_time` parameter
- Check if site blocks automated browsers
- Try disabling headless mode for debugging

## 📊 Output Format

All results are returned in clean markdown format:

```markdown
## LA Lakers Team Statistics

### Season Stats
- **Record**: 45-37
- **Win Rate**: 54.9%
- **Points Per Game**: 112.5
- **Opponent PPG**: 110.8

### Key Metrics
- Field Goal %: 47.2%
- 3-Point %: 36.8%
- Free Throw %: 78.9%
```

## 🔒 Privacy & Security

- Screenshots are saved locally only
- API calls are sent over HTTPS
- Ollama option runs 100% locally with no external API calls
- No data is stored by the tool (except local screenshots)

## 📝 License

MIT License - Use freely for personal and commercial projects

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## ⚠️ Disclaimer

- Respect website terms of service and robots.txt
- Be mindful of rate limits when using APIs
- This tool is for legitimate data extraction only
