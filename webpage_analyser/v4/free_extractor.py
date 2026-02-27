#!/usr/bin/env python3
"""
FREE Web Content Extractor with Visual Analysis
Uses open-source OCR tools (PaddleOCR, EasyOCR) for screenshot analysis
No paid APIs required - 100% free!
"""

import re
import sys
import json
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse
import argparse
from pathlib import Path


def setup_dependencies():
    """Install required dependencies if not available"""
    packages = {
        'core': ['trafilatura', 'requests', 'beautifulsoup4', 'lxml', 'playwright'],
        'ocr': ['paddleocr', 'easyocr', 'pillow', 'opencv-python']
    }
    
    missing = []
    
    # Check core packages
    for package in packages['core']:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Installing core dependencies: {', '.join(missing)}...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            *missing, "--break-system-packages", "-q"
        ])
        
        if "playwright" in missing:
            print("Installing Playwright browsers...")
            subprocess.check_call([
                sys.executable, "-m", "playwright", "install", "chromium", "--with-deps"
            ])
    
    # For OCR, install on demand
    print("Core dependencies ready!\n")


def capture_screenshot(url: str, output_path: str = None) -> Optional[str]:
    """Capture webpage screenshot using Playwright"""
    from playwright.sync_api import sync_playwright
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1920, 'height': 1080})
            
            print(f"Loading: {url}")
            page.goto(url, wait_until='networkidle', timeout=30000)
            page.wait_for_timeout(2000)
            
            if not output_path:
                domain = urlparse(url).netloc.replace('.', '_')
                output_path = f"screenshot_{domain}.png"
            
            page.screenshot(path=output_path, full_page=True)
            browser.close()
            
            print(f"✓ Screenshot saved: {output_path}")
            return output_path
            
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None


def analyze_with_paddleocr(image_path: str, query: str) -> Dict:
    """
    Analyze screenshot using PaddleOCR (FREE, highly accurate)
    Best for: Chinese+English, tables, structured text
    """
    try:
        print("Loading PaddleOCR (this may take a moment on first run)...")
        from paddleocr import PaddleOCR
        import cv2
        
        # Initialize PaddleOCR
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False,
            use_gpu=False  # Set True if you have GPU
        )
        
        print("Analyzing screenshot with PaddleOCR...")
        
        # Read image
        img = cv2.imread(image_path)
        
        # Perform OCR
        result = ocr.ocr(image_path, cls=True)
        
        # Extract text with positions
        extracted_text = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.5:  # Filter low confidence
                extracted_text.append(text)
        
        full_text = '\n'.join(extracted_text)
        
        # Filter by query
        relevant_text = filter_ocr_text_by_query(full_text, query)
        
        return {
            'method': 'PaddleOCR',
            'text': full_text,
            'relevant_text': relevant_text,
            'total_lines': len(extracted_text),
            'confidence': 'high'
        }
        
    except ImportError:
        print("PaddleOCR not installed. Installing now...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "paddleocr", "paddlepaddle", "--break-system-packages", "-q"
        ])
        return analyze_with_paddleocr(image_path, query)
    except Exception as e:
        print(f"PaddleOCR error: {e}")
        return None


def analyze_with_easyocr(image_path: str, query: str) -> Dict:
    """
    Analyze screenshot using EasyOCR (FREE, supports 80+ languages)
    Best for: Multi-language, general purpose
    """
    try:
        print("Loading EasyOCR...")
        import easyocr
        
        # Initialize EasyOCR
        reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if available
        
        print("Analyzing screenshot with EasyOCR...")
        
        # Perform OCR
        result = reader.readtext(image_path)
        
        # Extract text
        extracted_text = []
        for detection in result:
            text = detection[1]
            confidence = detection[2]
            if confidence > 0.5:
                extracted_text.append(text)
        
        full_text = '\n'.join(extracted_text)
        
        # Filter by query
        relevant_text = filter_ocr_text_by_query(full_text, query)
        
        return {
            'method': 'EasyOCR',
            'text': full_text,
            'relevant_text': relevant_text,
            'total_lines': len(extracted_text),
            'confidence': 'high'
        }
        
    except ImportError:
        print("EasyOCR not installed. Installing now...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "easyocr", "--break-system-packages", "-q"
        ])
        return analyze_with_easyocr(image_path, query)
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return None


def analyze_with_tesseract(image_path: str, query: str) -> Dict:
    """
    Analyze screenshot using Tesseract OCR (FREE, widely used)
    Best for: Simple text extraction, lightweight
    """
    try:
        import pytesseract
        from PIL import Image
        
        print("Analyzing screenshot with Tesseract OCR...")
        
        # Open image
        img = Image.open(image_path)
        
        # Perform OCR
        text = pytesseract.image_to_string(img)
        
        # Filter by query
        relevant_text = filter_ocr_text_by_query(text, query)
        
        return {
            'method': 'Tesseract',
            'text': text,
            'relevant_text': relevant_text,
            'total_lines': len(text.split('\n')),
            'confidence': 'medium'
        }
        
    except ImportError:
        print("Tesseract not installed. Install with: sudo apt-get install tesseract-ocr")
        print("And: pip install pytesseract --break-system-packages")
        return None
    except Exception as e:
        print(f"Tesseract error: {e}")
        return None


def filter_ocr_text_by_query(text: str, query: str) -> str:
    """Filter OCR text to only include query-relevant lines"""
    if not text or not query:
        return text
    
    # Extract query keywords
    query_keywords = set(re.findall(r'\w+', query.lower()))
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'from'}
    query_keywords = query_keywords - stop_words
    
    if not query_keywords:
        return text
    
    # Filter lines
    lines = text.split('\n')
    relevant_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_lower = line.lower()
        matches = sum(1 for keyword in query_keywords if keyword in line_lower)
        
        # Include line if it has keywords or looks like data
        has_numbers = bool(re.search(r'\d+', line))
        has_keywords = matches >= 1
        
        if has_keywords or (has_numbers and len(line) > 5):
            relevant_lines.append(line)
    
    return '\n'.join(relevant_lines)


def extract_main_content(url: str) -> Optional[Dict]:
    """Extract text content using trafilatura"""
    import trafilatura
    import requests
    
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        html_content = response.text
        
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


def combine_text_and_ocr(text_content: Dict, ocr_result: Dict, query: str) -> str:
    """Combine HTML text extraction with OCR results"""
    output_parts = []
    
    # Header
    title = text_content['metadata'].get('title', 'Extracted Content')
    output_parts.append(f"# {title}\n")
    output_parts.append(f"**Query:** {query}\n")
    output_parts.append(f"**Source:** {text_content['metadata']['url']}\n")
    output_parts.append(f"**Extraction Method:** HTML Text + OCR ({ocr_result['method']})\n")
    
    if text_content['metadata'].get('date'):
        output_parts.append(f"**Date:** {text_content['metadata']['date']}\n")
    
    output_parts.append("\n---\n")
    
    # OCR Analysis
    if ocr_result and ocr_result.get('relevant_text'):
        output_parts.append("## 📸 Visual Content Analysis (OCR)\n")
        output_parts.append(f"**Method:** {ocr_result['method']} (Free, Open-Source)\n")
        output_parts.append(f"**Lines Extracted:** {ocr_result['total_lines']}\n")
        output_parts.append(f"**Confidence:** {ocr_result['confidence']}\n\n")
        
        output_parts.append("**Query-Relevant Visual Content:**\n")
        output_parts.append("```\n")
        output_parts.append(ocr_result['relevant_text'][:2000])  # Limit length
        output_parts.append("\n```\n")
        
        output_parts.append("\n---\n")
    
    # HTML Text
    output_parts.append("## 📄 HTML Text Extraction\n")
    output_parts.append(text_content['markdown'])
    
    return '\n'.join(output_parts)


def main():
    parser = argparse.ArgumentParser(
        description='FREE web content extractor with visual OCR analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FREE OCR Methods (No API costs):
  --ocr paddle     PaddleOCR (Best accuracy, supports tables)
  --ocr easy       EasyOCR (80+ languages, good accuracy)
  --ocr tesseract  Tesseract (Lightweight, classic)
  --ocr auto       Try best available (default)

Examples:
  # Use PaddleOCR (most accurate)
  python free_extractor.py -u URL -q "query" --ocr paddle
  
  # Use EasyOCR (multi-language)
  python free_extractor.py -u URL -q "query" --ocr easy
  
  # Auto-select best available
  python free_extractor.py -u URL -q "query" --ocr auto
  
  # Text-only (no OCR, fastest)
  python free_extractor.py -u URL -q "query"
        """
    )
    
    parser.add_argument('-u', '--url', required=True, help='URL to extract from')
    parser.add_argument('-q', '--query', required=True, help='Query for filtering')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--ocr', choices=['paddle', 'easy', 'tesseract', 'auto'], 
                       help='Enable OCR analysis (free)')
    parser.add_argument('--screenshot', help='Screenshot path')
    parser.add_argument('--keep-screenshot', action='store_true')
    
    args = parser.parse_args()
    
    # Setup
    setup_dependencies()
    
    print(f"\n{'='*80}")
    print("FREE WEB CONTENT EXTRACTOR WITH OCR")
    print(f"{'='*80}\n")
    print(f"URL: {args.url}")
    print(f"Query: {args.query}")
    print(f"OCR: {args.ocr or 'Disabled (text-only)'}\n")
    
    # Extract HTML text
    print("Step 1: Extracting HTML text content...")
    text_content = extract_main_content(args.url)
    
    if not text_content:
        print("Failed to extract content")
        sys.exit(1)
    
    print(f"✓ Extracted {len(text_content['text'])} characters\n")
    
    # OCR analysis if requested
    ocr_result = None
    
    if args.ocr:
        print("Step 2: Capturing screenshot...")
        screenshot_path = args.screenshot or "temp_screenshot.png"
        screenshot_path = capture_screenshot(args.url, screenshot_path)
        
        if screenshot_path:
            print("\nStep 3: Analyzing screenshot with OCR...")
            
            if args.ocr == 'paddle':
                ocr_result = analyze_with_paddleocr(screenshot_path, args.query)
            elif args.ocr == 'easy':
                ocr_result = analyze_with_easyocr(screenshot_path, args.query)
            elif args.ocr == 'tesseract':
                ocr_result = analyze_with_tesseract(screenshot_path, args.query)
            elif args.ocr == 'auto':
                # Try in order of accuracy
                ocr_result = (analyze_with_paddleocr(screenshot_path, args.query) or
                            analyze_with_easyocr(screenshot_path, args.query) or
                            analyze_with_tesseract(screenshot_path, args.query))
            
            if ocr_result:
                print(f"✓ OCR analysis complete using {ocr_result['method']}")
            
            # Cleanup
            if not args.keep_screenshot and not args.screenshot:
                try:
                    Path(screenshot_path).unlink()
                except:
                    pass
    
    # Combine results
    print("\nStep 4: Generating output...\n")
    
    if ocr_result:
        output = combine_text_and_ocr(text_content, ocr_result, args.query)
    else:
        # Text-only output
        output = f"""# {text_content['metadata']['title'] or 'Extracted Content'}

**Query:** {args.query}
**Source:** {text_content['metadata']['url']}

---

{text_content['markdown']}
"""
    
    # Save or print
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"✓ Saved to: {args.output}")
    else:
        print("="*80)
        print(output)
        print("="*80)
    
    print(f"\n{'='*80}")
    print("✓ EXTRACTION COMPLETE (100% FREE!)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
