#!/usr/bin/env python3
"""
Ultimate Demo - Compare All Three Extraction Methods
Shows: Text-Only, Free OCR, and Paid Claude Vision
"""

import sys
import time
from pathlib import Path


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def demo_text_only():
    """Demo: Text-only extraction (fastest, free)"""
    print_header("DEMO 1: Text-Only Extraction (FREE, FASTEST)")
    
    print("✨ Method: HTML text extraction only")
    print("⚡ Speed: 1-3 seconds")
    print("💰 Cost: $0")
    print("📊 Accuracy: 70-75%")
    print("🎯 Best for: Text-heavy content (blogs, articles, docs)\n")
    
    from web_content_extractor import extract_main_content, format_output
    
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    query = "Python programming language features syntax"
    
    print(f"URL: {url}")
    print(f"Query: {query}\n")
    
    proceed = input("Run demo? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Skipped.\n")
        return
    
    print("\nExtracting...")
    start = time.time()
    
    content = extract_main_content(url)
    if content:
        output = format_output(content, query, filter_by_query=True)
        duration = time.time() - start
        
        filename = "demo_1_text_only.md"
        with open(filename, 'w') as f:
            f.write(output)
        
        print(f"✅ Complete in {duration:.1f}s")
        print(f"📄 Output: {filename} ({len(output)} chars)")
        print(f"💰 Cost: $0")
        
        # Show preview
        print("\nPreview (first 500 chars):")
        print("-" * 80)
        print(output[:500] + "...")
        print("-" * 80)
    else:
        print("❌ Failed")


def demo_free_ocr():
    """Demo: Free OCR extraction (good accuracy, free)"""
    print_header("DEMO 2: Free OCR Extraction (FREE, GOOD ACCURACY)")
    
    print("✨ Method: HTML text + PaddleOCR screenshot analysis")
    print("⚡ Speed: 8-15 seconds")
    print("💰 Cost: $0")
    print("📊 Accuracy: 80-85%")
    print("🎯 Best for: Data tables, mixed content\n")
    
    from free_extractor import (
        capture_screenshot,
        analyze_with_paddleocr,
        extract_main_content,
        combine_text_and_ocr
    )
    
    url = "https://en.wikipedia.org/wiki/Los_Angeles_Lakers"
    query = "Lakers championships statistics"
    
    print(f"URL: {url}")
    print(f"Query: {query}\n")
    
    proceed = input("Run demo? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Skipped.\n")
        return
    
    print("\nExtracting...")
    start = time.time()
    
    # Capture screenshot
    print("  → Capturing screenshot...")
    screenshot = capture_screenshot(url, "demo_2_screenshot.png")
    
    if not screenshot:
        print("❌ Screenshot failed")
        return
    
    # OCR analysis
    print("  → Analyzing with PaddleOCR...")
    ocr_result = analyze_with_paddleocr(screenshot, query)
    
    # Text extraction
    print("  → Extracting HTML text...")
    text_content = extract_main_content(url)
    
    if ocr_result and text_content:
        output = combine_text_and_ocr(text_content, ocr_result, query)
        duration = time.time() - start
        
        filename = "demo_2_free_ocr.md"
        with open(filename, 'w') as f:
            f.write(output)
        
        print(f"✅ Complete in {duration:.1f}s")
        print(f"📄 Output: {filename} ({len(output)} chars)")
        print(f"📸 Screenshot: demo_2_screenshot.png")
        print(f"💰 Cost: $0")
        
        # Show preview
        print("\nPreview (first 500 chars):")
        print("-" * 80)
        print(output[:500] + "...")
        print("-" * 80)
    else:
        print("❌ Failed")


def demo_vision_api():
    """Demo: Claude Vision API (highest accuracy, paid)"""
    print_header("DEMO 3: Claude Vision API (PAID, HIGHEST ACCURACY)")
    
    print("✨ Method: HTML text + Claude Vision screenshot analysis")
    print("⚡ Speed: 10-20 seconds")
    print("💰 Cost: ~$0.02 per page")
    print("📊 Accuracy: 90-95%")
    print("🎯 Best for: Complex charts, mission-critical data\n")
    
    from enhanced_extractor import (
        capture_screenshot,
        analyze_screenshot_with_claude,
        extract_main_content,
        combine_analyses
    )
    
    url = "https://en.wikipedia.org/wiki/Basketball"
    query = "basketball court dimensions rules"
    
    print(f"URL: {url}")
    print(f"Query: {query}\n")
    
    proceed = input("Run demo? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Skipped.\n")
        return
    
    print("\nExtracting...")
    start = time.time()
    
    # Capture screenshot
    print("  → Capturing screenshot...")
    screenshot = capture_screenshot(url, "demo_3_screenshot.png")
    
    if not screenshot:
        print("❌ Screenshot failed")
        return
    
    # Vision analysis
    print("  → Analyzing with Claude Vision API...")
    visual_analysis = analyze_screenshot_with_claude(screenshot, query, url)
    
    # Text extraction
    print("  → Extracting HTML text...")
    text_content = extract_main_content(url)
    
    if visual_analysis and text_content:
        output = combine_analyses(text_content, visual_analysis, query)
        duration = time.time() - start
        
        filename = "demo_3_vision_api.md"
        with open(filename, 'w') as f:
            f.write(output)
        
        print(f"✅ Complete in {duration:.1f}s")
        print(f"📄 Output: {filename} ({len(output)} chars)")
        print(f"📸 Screenshot: demo_3_screenshot.png")
        print(f"💰 Cost: ~$0.02")
        
        # Show preview
        print("\nPreview (first 500 chars):")
        print("-" * 80)
        print(output[:500] + "...")
        print("-" * 80)
    else:
        print("❌ Failed")


def demo_comparison():
    """Demo: Side-by-side comparison"""
    print_header("DEMO 4: Side-by-Side Comparison")
    
    print("This demo runs the SAME URL with all three methods")
    print("and compares speed, cost, and output quality.\n")
    
    url = "https://en.wikipedia.org/wiki/National_Basketball_Association"
    query = "NBA teams statistics"
    
    print(f"URL: {url}")
    print(f"Query: {query}\n")
    
    proceed = input("Run comparison? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Skipped.\n")
        return
    
    results = {}
    
    # Method 1: Text-only
    print("\n[1/3] Running Text-Only...")
    from web_content_extractor import extract_main_content, format_output
    
    start = time.time()
    content1 = extract_main_content(url)
    if content1:
        output1 = format_output(content1, query, filter_by_query=True)
        with open("comparison_text_only.md", 'w') as f:
            f.write(output1)
        results['text_only'] = {
            'time': time.time() - start,
            'size': len(output1),
            'cost': 0,
            'file': 'comparison_text_only.md'
        }
        print(f"  ✅ Done in {results['text_only']['time']:.1f}s")
    
    # Method 2: Free OCR
    print("\n[2/3] Running Free OCR...")
    from free_extractor import (
        capture_screenshot,
        analyze_with_paddleocr,
        combine_text_and_ocr
    )
    
    start = time.time()
    screenshot2 = capture_screenshot(url, "comparison_ocr_screenshot.png")
    if screenshot2:
        ocr2 = analyze_with_paddleocr(screenshot2, query)
        content2 = extract_main_content(url)
        if ocr2 and content2:
            output2 = combine_text_and_ocr(content2, ocr2, query)
            with open("comparison_free_ocr.md", 'w') as f:
                f.write(output2)
            results['free_ocr'] = {
                'time': time.time() - start,
                'size': len(output2),
                'cost': 0,
                'file': 'comparison_free_ocr.md'
            }
            print(f"  ✅ Done in {results['free_ocr']['time']:.1f}s")
    
    # Method 3: Vision API
    print("\n[3/3] Running Claude Vision API...")
    from enhanced_extractor import (
        analyze_screenshot_with_claude,
        combine_analyses
    )
    
    start = time.time()
    screenshot3 = capture_screenshot(url, "comparison_vision_screenshot.png")
    if screenshot3:
        vision3 = analyze_screenshot_with_claude(screenshot3, query, url)
        content3 = extract_main_content(url)
        if vision3 and content3:
            output3 = combine_analyses(content3, vision3, query)
            with open("comparison_vision_api.md", 'w') as f:
                f.write(output3)
            results['vision_api'] = {
                'time': time.time() - start,
                'size': len(output3),
                'cost': 0.02,
                'file': 'comparison_vision_api.md'
            }
            print(f"  ✅ Done in {results['vision_api']['time']:.1f}s")
    
    # Display comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\n{'Method':<20} {'Time':<12} {'Size':<12} {'Cost':<10} {'File'}")
    print("-" * 80)
    
    for method, data in results.items():
        print(f"{method.replace('_', ' ').title():<20} "
              f"{data['time']:.1f}s{'':<8} "
              f"{data['size']:,} chars{'':<2} "
              f"${data['cost']:.2f}{'':<6} "
              f"{data['file']}")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS:")
    print("  • Text-Only: Fastest, free, good for text-heavy pages")
    print("  • Free OCR: Good balance, extracts visual data, still free")
    print("  • Vision API: Highest accuracy, understands context, small cost")
    print("="*80)


def interactive_mode():
    """Interactive mode - user chooses method"""
    print_header("INTERACTIVE MODE - Choose Your Method")
    
    url = input("Enter URL: ").strip()
    query = input("Enter query: ").strip()
    
    if not url or not query:
        print("Error: URL and query required")
        return
    
    print("\nChoose extraction method:")
    print("1. Text-Only (fastest, free)")
    print("2. Free OCR (good accuracy, free)")
    print("3. Claude Vision (best accuracy, ~$0.02)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    output_file = input("Output filename (default: output.md): ").strip() or "output.md"
    
    print("\nProcessing...")
    
    if choice == '1':
        from web_content_extractor import extract_main_content, format_output
        content = extract_main_content(url)
        if content:
            output = format_output(content, query, filter_by_query=True)
            with open(output_file, 'w') as f:
                f.write(output)
            print(f"✅ Saved to {output_file}")
    
    elif choice == '2':
        from free_extractor import (
            capture_screenshot,
            analyze_with_paddleocr,
            extract_main_content,
            combine_text_and_ocr
        )
        screenshot = capture_screenshot(url)
        if screenshot:
            ocr = analyze_with_paddleocr(screenshot, query)
            content = extract_main_content(url)
            if ocr and content:
                output = combine_text_and_ocr(content, ocr, query)
                with open(output_file, 'w') as f:
                    f.write(output)
                print(f"✅ Saved to {output_file}")
    
    elif choice == '3':
        from enhanced_extractor import (
            capture_screenshot,
            analyze_screenshot_with_claude,
            extract_main_content,
            combine_analyses
        )
        screenshot = capture_screenshot(url)
        if screenshot:
            vision = analyze_screenshot_with_claude(screenshot, query, url)
            content = extract_main_content(url)
            if vision and content:
                output = combine_analyses(content, vision, query)
                with open(output_file, 'w') as f:
                    f.write(output)
                print(f"✅ Saved to {output_file}")
    
    else:
        print("Invalid choice")


def main():
    print_header("ULTIMATE WEB CONTENT EXTRACTOR DEMO SUITE")
    
    print("This demo compares THREE extraction methods:\n")
    print("1. 📝 Text-Only (FREE, fastest)")
    print("2. 🔍 Free OCR (FREE, good accuracy)")
    print("3. 🤖 Claude Vision (PAID, best accuracy)")
    print("4. ⚖️  Side-by-Side Comparison")
    print("5. 🎮 Interactive Mode")
    print("6. 🎯 Run All Demos")
    print("0. Exit\n")
    
    choice = input("Enter choice (0-6): ").strip()
    
    if choice == '1':
        demo_text_only()
    elif choice == '2':
        demo_free_ocr()
    elif choice == '3':
        demo_vision_api()
    elif choice == '4':
        demo_comparison()
    elif choice == '5':
        interactive_mode()
    elif choice == '6':
        demo_text_only()
        demo_free_ocr()
        demo_vision_api()
        demo_comparison()
        print_header("ALL DEMOS COMPLETE!")
    elif choice == '0':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
