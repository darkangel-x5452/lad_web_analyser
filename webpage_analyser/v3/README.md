# Web Content Extractor

A Python tool that intelligently extracts query-relevant information from websites, filters out boilerplate content (headers, footers, navigation), and outputs clean markdown format.

## Features

- ✅ **Intelligent Content Extraction**: Uses trafilatura to automatically remove headers, footers, ads, and navigation
- ✅ **Query-Based Filtering**: Extracts only information relevant to your specific query
- ✅ **Statistics & Data Focus**: Prioritizes numerical data, statistics, and factual content
- ✅ **Markdown Output**: Clean, readable markdown format
- ✅ **100% Free Tools**: No API keys or paid services required
- ✅ **Metadata Extraction**: Includes title, author, date, and source URL

## Installation

```bash
# Install dependencies
pip install -r requirements.txt --break-system-packages

# Or install manually
pip install trafilatura requests beautifulsoup4 lxml --break-system-packages
```

## Quick Start

### Command Line Usage

```bash
# Basic usage
python web_content_extractor.py \
  -u "https://en.wikipedia.org/wiki/Los_Angeles_Lakers" \
  -q "LA Lakers statistics 3-point percentage championships"

# Save to file
python web_content_extractor.py \
  -u "https://example.com/article" \
  -q "specific topic keywords" \
  -o output.md

# Disable filtering (extract all content)
python web_content_extractor.py \
  -u "https://example.com" \
  -q "search terms" \
  --no-filter
```

### Programmatic Usage

```python
from web_content_extractor import extract_main_content, format_output

# Extract content
url = "https://en.wikipedia.org/wiki/Basketball"
query = "basketball court dimensions and player positions"

content = extract_main_content(url)
if content:
    # Format with query filtering
    markdown_output = format_output(content, query, filter_by_query=True)
    
    # Save to file
    with open('output.md', 'w') as f:
        f.write(markdown_output)
    
    # Access raw data
    print(f"Title: {content['metadata']['title']}")
    print(f"Text: {content['text'][:200]}...")
    print(f"Markdown: {content['markdown'][:200]}...")
```

## Examples

### Example 1: Sports Statistics

**Query:** "LA Lakers 3-point statistics and championships"

**URL:** `https://en.wikipedia.org/wiki/Los_Angeles_Lakers`

```bash
python web_content_extractor.py \
  -u "https://en.wikipedia.org/wiki/Los_Angeles_Lakers" \
  -q "LA Lakers 3-point statistics championships" \
  -o lakers_stats.md
```

**Output:** Extracts only sections containing championship records, win percentages, 3-point stats, etc. Removes navigation, "See also" sections, and external links.

### Example 2: Technical Documentation

**Query:** "Python programming features data types"

**URL:** `https://docs.python.org/3/tutorial/introduction.html`

```bash
python web_content_extractor.py \
  -u "https://docs.python.org/3/tutorial/introduction.html" \
  -q "Python data types features" \
  -o python_features.md
```

**Output:** Focuses on data types, features, and code examples. Filters out navigation menus and site boilerplate.

### Example 3: Research Articles

**Query:** "climate change temperature increase data"

**URL:** `https://en.wikipedia.org/wiki/Climate_change`

```bash
python web_content_extractor.py \
  -u "https://en.wikipedia.org/wiki/Climate_change" \
  -q "climate change temperature increase statistics data" \
  -o climate_data.md
```

**Output:** Extracts temperature data, statistical trends, and scientific findings while removing citations sections and external links.

## How It Works

### 1. Content Extraction
- Uses **trafilatura** - a specialized tool designed to extract main content from web pages
- Automatically removes:
  - Headers and footers
  - Navigation menus
  - Advertisements
  - Cookie notices
  - Social media widgets
  - Comments sections

### 2. Query-Based Filtering
- Analyzes query keywords (removes stop words like "the", "a", "and")
- Scores each paragraph based on keyword relevance
- Includes paragraphs with >30% keyword match or at least 2 matching keywords
- Separates relevant sections with horizontal rules

### 3. Statistics & Data Prioritization
- Identifies content containing:
  - Percentages (e.g., "45.2%")
  - Sports statistics (points, rebounds, assists)
  - Monetary values (e.g., "$10M")
  - Large numbers (e.g., "1,234,567")
  - Rates and ratios (e.g., "3.5 per game")
  - Table data
- Removes promotional content ("click here", "subscribe", "follow us")

### 4. Markdown Formatting
- Converts HTML to clean markdown
- Preserves:
  - Headings and structure
  - Tables
  - Links (optional)
  - Bold and italic formatting
- Adds metadata header with query, source URL, and date

## Command Line Arguments

```
-u, --url           URL to extract content from (required)
-q, --query         Query to filter relevant information (required)
-o, --output        Output file path (optional, prints to stdout if not specified)
--no-filter         Disable query-based filtering, extract all content
```

## Interactive Demo

Run the demo script to see examples:

```bash
python demo.py
```

The demo includes:
1. Basic extraction example with Wikipedia
2. Programmatic usage showing API
3. Multiple queries on the same URL
4. Interactive mode (enter your own URLs)

## Output Format

The tool generates markdown with this structure:

```markdown
# Page Title

**Query:** your search query here
**Source:** https://example.com
**Date:** 2024-01-15

---

## Relevant Section 1

Content with statistics and data...

---

## Relevant Section 2

More query-relevant content...
```

## Tools Used

All tools are **free and open-source**:

1. **trafilatura** (MIT License)
   - Intelligent content extraction
   - Removes boilerplate automatically
   - Optimized for article and blog extraction

2. **requests** (Apache 2.0)
   - HTTP client for fetching web pages

3. **beautifulsoup4** (MIT License)
   - HTML parsing (used as fallback)

4. **lxml** (BSD License)
   - Fast XML/HTML parsing

## Limitations

- Requires internet connection to fetch web pages
- Some websites may block automated access (use responsibly)
- JavaScript-heavy sites may not render properly (content must be in initial HTML)
- Query filtering is keyword-based (not semantic understanding)
- Respects robots.txt (built into trafilatura)

## Best Practices

1. **Be Specific with Queries**: "NBA Lakers 3-point percentage" is better than "Lakers info"
2. **Include Key Terms**: Add domain-specific keywords for better filtering
3. **Test Different URLs**: Some sites work better than others for extraction
4. **Use --no-filter for Exploration**: First extract everything, then refine your query
5. **Respect Rate Limits**: Don't hammer servers with rapid requests

## Troubleshooting

**Problem:** "Could not extract content from URL"
- **Solution:** Try with `--no-filter` flag, or check if the URL is accessible

**Problem:** Output is empty or very short
- **Solution:** Your query might be too specific. Try broader keywords

**Problem:** Still getting irrelevant content
- **Solution:** Make your query more specific with unique domain keywords

**Problem:** Installation fails
- **Solution:** Make sure you use `--break-system-packages` flag or create a virtual environment

## Advanced Usage

### Batch Processing

```python
from web_content_extractor import extract_main_content, format_output

urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
]

query = "specific information to extract"

for i, url in enumerate(urls, 1):
    content = extract_main_content(url)
    if content:
        output = format_output(content, query)
        with open(f'output_{i}.md', 'w') as f:
            f.write(output)
```

### Custom Filtering

```python
from web_content_extractor import (
    extract_main_content, 
    filter_content_by_query,
    extract_statistics_and_data
)

content = extract_main_content(url)

# Apply custom filtering
text = content['text']
relevant = filter_content_by_query(text, "your query")
stats_focused = extract_statistics_and_data('\n\n'.join(relevant), "your query")
```

## License

This tool uses open-source libraries. Please refer to individual library licenses:
- trafilatura: Apache License 2.0
- requests: Apache License 2.0
- beautifulsoup4: MIT License
- lxml: BSD License

## Contributing

Feel free to enhance the filtering algorithms, add new features, or improve the extraction quality!

## Support

For issues or questions, refer to the documentation of the underlying libraries:
- [Trafilatura Documentation](https://trafilatura.readthedocs.io/)
- [Requests Documentation](https://requests.readthedocs.io/)
- [Beautiful Soup Documentation](https://www.crummy.com/software/BeautifulSoup/)
