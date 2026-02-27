#!/usr/bin/env python3
"""
Web Content Extractor - Intelligently extract query-relevant information from websites
Uses free tools: trafilatura, beautifulsoup4, requests
"""

import os
import re
import sys
from typing import Optional, Dict, List
from urllib.parse import urlparse
import argparse
from dotenv import load_dotenv
import trafilatura
import requests
from bs4 import BeautifulSoup

load_dotenv()  # loads .env from current directory




def extract_main_content(url: str) -> Optional[Dict[str, str]]:
    """
    Extract main content from a URL using trafilatura
    Returns dict with 'text', 'markdown', and 'metadata'
    """
    import trafilatura
    import requests
    
    try:
        # Fetch the webpage
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        html_content = response.text
        
        # Extract using trafilatura (removes boilerplate automatically)
        extracted_text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            include_links=False,
            no_fallback=False
        )
        
        # Extract with markdown output
        extracted_markdown = trafilatura.extract(
            html_content,
            output_format='markdown',
            include_comments=False,
            include_tables=True,
            include_links=True,
            no_fallback=False
        )
        
        # Extract metadata
        metadata = trafilatura.extract_metadata(html_content)
        
        if not extracted_text:
            print(f"Warning: Could not extract content from {url}")
            return None
            
        return {
            'text': extracted_text,
            'markdown': extracted_markdown,
            'metadata': {
                'title': metadata.title if metadata else None,
                'author': metadata.author if metadata else None,
                'date': metadata.date if metadata else None,
                'url': url
            }
        }
        
    except Exception as e:
        print(f"Error extracting content: {e}")
        return None


def filter_content_by_query(content: str, query: str, context_lines: int = 2) -> List[str]:
    """
    Filter content to only include sections relevant to the query
    Returns list of relevant text chunks
    """
    if not content or not query:
        return [content] if content else []
    
    # Extract keywords from query (simple approach)
    query_keywords = set(re.findall(r'\w+', query.lower()))
    # Remove common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'from', 'get', 'give', 'show', 'find'}
    query_keywords = query_keywords - stop_words
    
    if not query_keywords:
        return [content]
    
    # Split content into paragraphs
    paragraphs = content.split('\n\n')
    relevant_sections = []
    
    for para in paragraphs:
        para_lower = para.lower()
        # Count keyword matches
        matches = sum(1 for keyword in query_keywords if keyword in para_lower)
        
        # Include paragraph if it has significant keyword overlap
        if matches >= min(2, len(query_keywords) * 0.3):  # At least 30% of keywords or 2 keywords
            relevant_sections.append(para)
    
    return relevant_sections if relevant_sections else [content]


def extract_statistics_and_data(markdown_content: str, query: str) -> str:
    """
    Further refine markdown to focus on data, statistics, and facts
    Removes purely navigational or promotional content
    """
    lines = markdown_content.split('\n')
    filtered_lines = []
    
    # Patterns that indicate useful data/statistics
    data_patterns = [
        r'\d+\.?\d*\s*%',  # Percentages
        r'\d+\.?\d*\s*(points|pts|rebounds|assists|goals|wins|losses)',  # Sports stats
        r'\$\d+',  # Monetary values
        r'\d{1,3}(,\d{3})*',  # Large numbers with commas
        r'\d+\.\d+',  # Decimals
        r'\d+\s*(per|\/)',  # Per/rate statistics
    ]
    
    # Patterns to exclude
    exclude_patterns = [
        r'(cookie|privacy|terms|conditions|subscribe|newsletter)',
        r'(click here|read more|learn more|see all)',
        r'(follow us|share|comment|sign up)',
    ]
    
    for line in lines:
        line_lower = line.lower()
        
        # Skip if matches exclude patterns
        if any(re.search(pattern, line_lower) for pattern in exclude_patterns):
            continue
        
        # Include if contains data patterns or tables
        if (any(re.search(pattern, line) for pattern in data_patterns) or
            line.strip().startswith('|') or  # Table rows
            line.strip().startswith('#') or   # Headers
            len(line.strip()) > 30):          # Substantial content
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def format_output(content_dict: Dict, query: str, filter_by_query: bool = True) -> str:
    """
    Format the extracted content as clean markdown with metadata
    """
    if not content_dict:
        return "# Error\n\nCould not extract content from the provided URL."
    
    markdown = content_dict['markdown']
    metadata = content_dict['metadata']
    
    # Filter content if requested
    if filter_by_query and query:
        markdown = extract_statistics_and_data(markdown, query)
        relevant_sections = filter_content_by_query(markdown, query)
        markdown = '\n\n---\n\n'.join(relevant_sections)
    
    # Build output
    output_parts = []
    
    # Add header with metadata
    output_parts.append(f"# {metadata['title'] or 'Extracted Content'}\n")
    output_parts.append(f"**Query:** {query}\n")
    output_parts.append(f"**Source:** {metadata['url']}\n")
    
    if metadata['date']:
        output_parts.append(f"**Date:** {metadata['date']}\n")
    
    output_parts.append("\n---\n")
    
    # Add extracted content
    output_parts.append(markdown)
    
    return '\n'.join(output_parts)


def main(
        url_input: str,
        query_input: str,
        output_file: str,
        no_filter: bool = False
):
#     parser = argparse.ArgumentParser(
#         description='Extract query-relevant information from websites',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python web_content_extractor.py -u "https://example.com/nba-lakers" -q "LA Lakers statistics 3-point percentage"
#   python web_content_extractor.py -u "https://example.com/article" -q "specific topic" -o output.md
#         """
#     )
    
    # parser.add_argument('-u', '--url', required=True, help='URL to extract content from')
    # parser.add_argument('-q', '--query', required=True, help='Query to filter relevant information')
    # parser.add_argument('-o', '--output', help='Output file (default: print to stdout)')
    # parser.add_argument('--no-filter', action='store_true', help='Disable query-based filtering')
    
    # args = parser.parse_args()
    
    # Setup dependencies
        
    print(f"Extracting content from: {url_input}")
    print(f"Query: {query_input}\n")
    
    # Extract content
    content_dict = extract_main_content(url_input)
    
    if not content_dict:
        print("Failed to extract content from URL")
        sys.exit(1)
    
    # Format output
    output = format_output(content_dict, query_input, filter_by_query=not no_filter)
    
    # Save or print
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Content saved to: {output_file}")
    else:
        print("\n" + "="*80)
        print("EXTRACTED CONTENT")
        print("="*80 + "\n")
        print(output)


if __name__ == '__main__':
    main(
        url_input=os.getenv("DEMO_URL_LINK"),
        query_input=os.getenv("DEMO_URL_QUERY"),
        output_file="data/webpage_analyser/v3/demo_output.md",
        no_filter=False
    )
