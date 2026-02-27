#!/usr/bin/env python3
"""
Enhanced Web Content Extractor with GenAI Vision Analysis
Combines text extraction with screenshot analysis using Claude Vision API
"""

import re
import sys
import json
import base64
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse
import argparse
from pathlib import Path


def setup_dependencies():
    """Install required dependencies if not available"""
    required = [
        "trafilatura",
        "requests", 
        "beautifulsoup4",
        "lxml",
        "playwright"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Installing required dependencies: {', '.join(missing)}...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            *missing,
            "--break-system-packages", "-q"
        ])
        
        # Install playwright browsers
        if "playwright" in missing:
            print("Installing Playwright browsers (this may take a moment)...")
            subprocess.check_call([
                sys.executable, "-m", "playwright", "install", "chromium", "--with-deps"
            ])
        
        print("Dependencies installed successfully!\n")


def capture_screenshot(url: str, output_path: str = None) -> Optional[str]:
    """
    Capture a screenshot of the webpage using Playwright
    Returns path to screenshot file
    """
    from playwright.sync_api import sync_playwright
    
    try:
        with sync_playwright() as p:
            # Launch browser in headless mode
            browser = p.chromium.launch(headless=True)
            
            # Create a new page with desktop viewport
            page = browser.new_page(viewport={'width': 1920, 'height': 1080})
            
            # Navigate to URL
            print(f"Loading webpage: {url}")
            page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait a bit for dynamic content to load
            page.wait_for_timeout(2000)
            
            # Generate output path if not provided
            if not output_path:
                domain = urlparse(url).netloc.replace('.', '_')
                output_path = f"screenshot_{domain}.png"
            
            # Take full page screenshot
            page.screenshot(path=output_path, full_page=True)
            
            browser.close()
            
            print(f"✓ Screenshot saved: {output_path}")
            return output_path
            
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def analyze_screenshot_with_claude(image_path: str, query: str, url: str) -> Optional[Dict]:
    """
    Analyze webpage screenshot using Claude Vision API
    Returns analysis results
    """
    import requests
    
    print(f"Analyzing screenshot with Claude Vision API...")
    
    # Encode image
    image_base64 = encode_image_to_base64(image_path)
    if not image_base64:
        return None
    
    # Prepare prompt for Claude
    prompt = f"""You are analyzing a webpage screenshot to extract specific information based on a user query.

URL: {url}
Query: {query}

Please analyze this webpage screenshot and extract ONLY information that is directly relevant to the query. Focus on:
1. Specific data, statistics, numbers, and facts related to the query
2. Key information visible in the main content area
3. Tables, charts, or data visualizations relevant to the query
4. Ignore navigation menus, headers, footers, advertisements, and unrelated content

Provide your analysis in this JSON format:
{{
    "relevant_sections": [
        {{
            "heading": "Section title or topic",
            "content": "The specific relevant information",
            "data_points": ["List of key statistics or facts"]
        }}
    ],
    "key_statistics": ["List of important numbers, percentages, or metrics"],
    "summary": "Brief summary of query-relevant information found"
}}

Be concise and focus only on information that directly answers or relates to the query."""

    try:
        # Call Claude API
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            },
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        
        # Extract text content from response
        text_content = ""
        for block in result.get('content', []):
            if block.get('type') == 'text':
                text_content += block.get('text', '')
        
        # Try to parse as JSON
        try:
            # Remove markdown code fences if present
            json_text = text_content.strip()
            if json_text.startswith('```json'):
                json_text = json_text[7:]
            if json_text.startswith('```'):
                json_text = json_text[3:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]
            
            analysis = json.loads(json_text.strip())
            print(f"✓ Visual analysis completed")
            return analysis
            
        except json.JSONDecodeError:
            # If not JSON, return raw text
            print(f"✓ Visual analysis completed (text format)")
            return {
                "summary": text_content,
                "relevant_sections": [],
                "key_statistics": []
            }
        
    except Exception as e:
        print(f"Error analyzing screenshot: {e}")
        return None


def extract_main_content(url: str) -> Optional[Dict[str, str]]:
    """
    Extract main content from a URL using trafilatura
    Returns dict with 'text', 'markdown', and 'metadata'
    """
    import trafilatura
    import requests
    
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        html_content = response.text
        
        # Extract using trafilatura
        extracted_text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            include_links=False,
            no_fallback=False
        )
        
        extracted_markdown = trafilatura.extract(
            html_content,
            output_format='markdown',
            include_comments=False,
            include_tables=True,
            include_links=True,
            no_fallback=False
        )
        
        metadata = trafilatura.extract_metadata(html_content)
        
        if not extracted_text:
            print(f"Warning: Could not extract text content from {url}")
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


def combine_analyses(text_content: Dict, visual_analysis: Dict, query: str) -> str:
    """
    Combine text extraction and visual analysis into unified markdown output
    """
    output_parts = []
    
    # Header
    title = text_content['metadata'].get('title', 'Extracted Content')
    output_parts.append(f"# {title}\n")
    output_parts.append(f"**Query:** {query}\n")
    output_parts.append(f"**Source:** {text_content['metadata']['url']}\n")
    output_parts.append(f"**Analysis Method:** Text Extraction + AI Visual Analysis\n")
    
    if text_content['metadata'].get('date'):
        output_parts.append(f"**Date:** {text_content['metadata']['date']}\n")
    
    output_parts.append("\n---\n")
    
    # Visual Analysis Results
    if visual_analysis:
        output_parts.append("## 🤖 AI Visual Analysis\n")
        
        # Summary
        if visual_analysis.get('summary'):
            output_parts.append(f"**Summary:** {visual_analysis['summary']}\n")
        
        # Key Statistics
        if visual_analysis.get('key_statistics'):
            output_parts.append("\n**Key Statistics Found:**\n")
            for stat in visual_analysis['key_statistics']:
                output_parts.append(f"- {stat}\n")
        
        # Relevant Sections
        if visual_analysis.get('relevant_sections'):
            output_parts.append("\n**Relevant Content Sections:**\n")
            for section in visual_analysis['relevant_sections']:
                heading = section.get('heading', 'Content')
                output_parts.append(f"\n### {heading}\n")
                output_parts.append(f"{section.get('content', '')}\n")
                
                if section.get('data_points'):
                    output_parts.append("\n**Data Points:**\n")
                    for point in section['data_points']:
                        output_parts.append(f"- {point}\n")
        
        output_parts.append("\n---\n")
    
    # Text Extraction Results
    output_parts.append("## 📄 Text Extraction Results\n")
    output_parts.append(text_content['markdown'])
    
    return '\n'.join(output_parts)


def filter_text_by_query(content: str, query: str) -> str:
    """Filter text content to only include query-relevant sections"""
    if not content or not query:
        return content
    
    # Extract keywords from query
    query_keywords = set(re.findall(r'\w+', query.lower()))
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'from', 'get', 'give', 'show', 'find'}
    query_keywords = query_keywords - stop_words
    
    if not query_keywords:
        return content
    
    # Split into paragraphs and filter
    paragraphs = content.split('\n\n')
    relevant = []
    
    for para in paragraphs:
        para_lower = para.lower()
        matches = sum(1 for keyword in query_keywords if keyword in para_lower)
        
        if matches >= min(2, len(query_keywords) * 0.3):
            relevant.append(para)
    
    return '\n\n'.join(relevant) if relevant else content


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced web content extractor with AI visual analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis (text + visual)
  python enhanced_extractor.py -u "https://example.com" -q "specific query" --vision
  
  # Text only (faster)
  python enhanced_extractor.py -u "https://example.com" -q "query"
  
  # Save screenshot and output
  python enhanced_extractor.py -u "https://example.com" -q "query" --vision -o output.md --screenshot screenshot.png
        """
    )
    
    parser.add_argument('-u', '--url', required=True, help='URL to extract content from')
    parser.add_argument('-q', '--query', required=True, help='Query to filter relevant information')
    parser.add_argument('-o', '--output', help='Output markdown file')
    parser.add_argument('--vision', action='store_true', help='Enable AI visual analysis (slower but more accurate)')
    parser.add_argument('--screenshot', help='Path to save screenshot')
    parser.add_argument('--keep-screenshot', action='store_true', help='Keep screenshot file after analysis')
    parser.add_argument('--text-only', action='store_true', help='Skip screenshot, use text extraction only')
    
    args = parser.parse_args()
    
    # Setup dependencies
    setup_dependencies()
    
    print(f"\n{'='*80}")
    print("ENHANCED WEB CONTENT EXTRACTOR")
    print(f"{'='*80}\n")
    print(f"URL: {args.url}")
    print(f"Query: {args.query}")
    print(f"Mode: {'Text + AI Vision' if args.vision and not args.text_only else 'Text Only'}\n")
    
    # Extract text content
    print("Step 1: Extracting text content...")
    text_content = extract_main_content(args.url)
    
    if not text_content:
        print("Failed to extract text content")
        sys.exit(1)
    
    print(f"✓ Extracted {len(text_content['text'])} characters of text\n")
    
    # Visual analysis if requested
    visual_analysis = None
    screenshot_path = None
    
    if args.vision and not args.text_only:
        print("Step 2: Capturing screenshot...")
        screenshot_path = args.screenshot or "temp_screenshot.png"
        screenshot_path = capture_screenshot(args.url, screenshot_path)
        
        if screenshot_path:
            print("\nStep 3: Analyzing screenshot with AI...")
            visual_analysis = analyze_screenshot_with_claude(screenshot_path, args.query, args.url)
            
            # Clean up screenshot if not keeping it
            if not args.keep_screenshot and not args.screenshot and screenshot_path:
                try:
                    Path(screenshot_path).unlink()
                    print(f"✓ Cleaned up temporary screenshot")
                except:
                    pass
        else:
            print("⚠ Screenshot capture failed, proceeding with text-only analysis")
    
    # Filter text content
    print("\nStep 4: Filtering content by query...")
    filtered_markdown = filter_text_by_query(text_content['markdown'], args.query)
    text_content['markdown'] = filtered_markdown
    
    # Combine results
    print("Step 5: Generating final output...\n")
    if visual_analysis:
        output = combine_analyses(text_content, visual_analysis, args.query)
    else:
        # Text-only output
        output_parts = []
        output_parts.append(f"# {text_content['metadata']['title'] or 'Extracted Content'}\n")
        output_parts.append(f"**Query:** {args.query}\n")
        output_parts.append(f"**Source:** {text_content['metadata']['url']}\n")
        output_parts.append("\n---\n")
        output_parts.append(text_content['markdown'])
        output = '\n'.join(output_parts)
    
    # Save or print
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"✓ Output saved to: {args.output}")
    else:
        print("\n" + "="*80)
        print("EXTRACTED CONTENT")
        print("="*80 + "\n")
        print(output)
    
    print(f"\n{'='*80}")
    print("✓ EXTRACTION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
