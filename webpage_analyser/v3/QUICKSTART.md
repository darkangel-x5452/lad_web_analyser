# Quick Start Guide

## Getting Started

This package contains a complete web content extraction tool that uses free, open-source libraries to extract query-relevant information from websites and output clean markdown.

## Files Included

1. **web_content_extractor.py** - Main extraction tool (CLI and module)
2. **demo.py** - Interactive demonstration with examples
3. **test_installation.py** - Validate your installation
4. **requirements.txt** - Python dependencies
5. **README.md** - Complete documentation

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

Or install individually:
```bash
pip install trafilatura requests beautifulsoup4 lxml --break-system-packages
```

### 2. Verify Installation

```bash
python test_installation.py
```

This will check that all dependencies are installed and working.

### 3. Run Examples

```bash
python demo.py
```

Choose from interactive examples to see the tool in action.

## Basic Usage Examples

### Example 1: Extract Sports Statistics

```bash
python web_content_extractor.py \
  -u "https://en.wikipedia.org/wiki/Los_Angeles_Lakers" \
  -q "LA Lakers championships 3-point statistics" \
  -o lakers_stats.md
```

**What it does:**
- Fetches the LA Lakers Wikipedia page
- Extracts only sections containing championships and 3-point statistics
- Removes navigation, headers, footers, "See also" sections
- Saves clean markdown to `lakers_stats.md`

### Example 2: Technical Documentation

```bash
python web_content_extractor.py \
  -u "https://en.wikipedia.org/wiki/Python_(programming_language)" \
  -q "Python features syntax data types" \
  -o python_features.md
```

**What it does:**
- Extracts Python language features and syntax information
- Filters to sections about data types and language features
- Removes promotional content and external links

### Example 3: Research Data

```bash
python web_content_extractor.py \
  -u "https://en.wikipedia.org/wiki/Climate_change" \
  -q "climate change temperature statistics data" \
  -o climate_data.md
```

**What it does:**
- Focuses on temperature data and statistics
- Extracts numerical facts and trends
- Removes citation sections and related article links

## Using as a Python Module

```python
from web_content_extractor import extract_main_content, format_output

# Your URL and query
url = "https://example.com/article"
query = "specific information you want"

# Extract content
content = extract_main_content(url)

if content:
    # Format with query filtering
    markdown = format_output(content, query, filter_by_query=True)
    
    # Save or use the markdown
    with open('output.md', 'w') as f:
        f.write(markdown)
    
    print(f"Extracted {len(markdown)} characters")
```

## Key Features

✅ **Smart Content Extraction** - Automatically removes headers, footers, ads, navigation  
✅ **Query-Based Filtering** - Only extracts content relevant to your query  
✅ **Statistics Focus** - Prioritizes numerical data and facts  
✅ **Clean Markdown Output** - Professional, readable format  
✅ **100% Free Tools** - No API keys or paid services required  

## Command Line Options

```
-u, --url           URL to extract from (required)
-q, --query         Query to filter content (required)
-o, --output        Save to file (optional, prints to stdout otherwise)
--no-filter         Extract all content without filtering
```

## Tips for Best Results

1. **Be Specific**: Use specific keywords in your query
   - Good: "Lakers 3-point percentage 2023 season"
   - Poor: "Lakers info"

2. **Include Domain Terms**: Add technical terms specific to your topic
   - Good: "Python list comprehension syntax"
   - Poor: "Python stuff"

3. **Test Different Queries**: Start broad, then narrow down
   - First: "NBA statistics"
   - Then: "NBA Lakers offensive statistics"

4. **Check Output**: Use `--no-filter` first to see all content, then refine query

## Troubleshooting

**Problem**: "Could not extract content"  
**Solution**: Check if URL is accessible, try with --no-filter

**Problem**: Output too short  
**Solution**: Query might be too specific, use broader keywords

**Problem**: Irrelevant content  
**Solution**: Add more specific domain keywords to your query

**Problem**: Installation errors  
**Solution**: Use --break-system-packages flag or create virtual environment

## What Makes This Tool Different?

Unlike simple web scrapers, this tool:

- **Understands Content Structure**: Uses trafilatura's machine learning to identify main content
- **Removes Boilerplate Automatically**: No manual CSS selectors needed
- **Focuses on Your Query**: Intelligent filtering based on keyword relevance
- **Prioritizes Data**: Recognizes statistics, percentages, and numerical facts
- **Produces Clean Output**: Professional markdown format ready to use

## Next Steps

1. Install dependencies: `pip install -r requirements.txt --break-system-packages`
2. Test it: `python test_installation.py`
3. Try examples: `python demo.py`
4. Read full docs: Open `README.md`

## Support

For detailed documentation, see `README.md`

For library documentation:
- [Trafilatura Docs](https://trafilatura.readthedocs.io/)
- [Requests Docs](https://requests.readthedocs.io/)

---

**Ready to extract?** Run `python demo.py` to get started!
