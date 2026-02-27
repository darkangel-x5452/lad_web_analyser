#!/usr/bin/env python3
"""
Enhanced Demo - Shows AI Vision Analysis + Text Extraction
"""

import sys
from pathlib import Path


def setup_and_import():
    """Setup dependencies and import modules"""
    print("Setting up dependencies...\n")
    
    # Setup dependencies
    from enhanced_extractor import setup_dependencies
    setup_dependencies()
    
    # Import after setup
    from enhanced_extractor import (
        capture_screenshot,
        extract_main_content,
        analyze_screenshot_with_claude,
        combine_analyses,
        filter_text_by_query
    )
    
    return {
        'capture_screenshot': capture_screenshot,
        'extract_main_content': extract_main_content,
        'analyze_screenshot_with_claude': analyze_screenshot_with_claude,
        'combine_analyses': combine_analyses,
        'filter_text_by_query': filter_text_by_query
    }


def demo_vision_analysis():
    """Demo: Full vision analysis + text extraction"""
    print("\n" + "="*80)
    print("DEMO 1: AI Vision Analysis + Text Extraction")
    print("="*80 + "\n")
    
    funcs = setup_and_import()
    
    # Example: Extract sports statistics
    url = "https://en.wikipedia.org/wiki/Los_Angeles_Lakers"
    query = "LA Lakers championships and recent statistics"
    
    print(f"URL: {url}")
    print(f"Query: {query}\n")
    
    print("This demo will:")
    print("1. Capture a screenshot of the webpage")
    print("2. Use Claude Vision API to analyze the screenshot")
    print("3. Extract text content using trafilatura")
    print("4. Combine both analyses into markdown\n")
    
    proceed = input("Proceed? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Skipped.")
        return
    
    # Extract text
    print("\nExtracting text content...")
    text_content = funcs['extract_main_content'](url)
    
    if not text_content:
        print("Failed to extract text")
        return
    
    print(f"✓ Extracted {len(text_content['text'])} characters")
    
    # Capture screenshot
    print("\nCapturing screenshot...")
    screenshot_path = "demo_lakers.png"
    screenshot_path = funcs['capture_screenshot'](url, screenshot_path)
    
    if not screenshot_path:
        print("Failed to capture screenshot")
        return
    
    # Analyze with Claude Vision
    print("\nAnalyzing with Claude Vision API...")
    visual_analysis = funcs['analyze_screenshot_with_claude'](screenshot_path, query, url)
    
    if not visual_analysis:
        print("Failed to analyze screenshot")
        return
    
    # Filter text
    print("\nFiltering text by query...")
    text_content['markdown'] = funcs['filter_text_by_query'](
        text_content['markdown'], 
        query
    )
    
    # Combine
    print("\nCombining analyses...")
    output = funcs['combine_analyses'](text_content, visual_analysis, query)
    
    # Save
    output_file = "demo_lakers_analysis.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Screenshot saved: {screenshot_path}")
    print(f"✓ Output saved: {output_file}")
    
    # Show preview
    print("\n" + "-"*80)
    print("Preview of output:")
    print("-"*80)
    preview = output[:800] + "..." if len(output) > 800 else output
    print(preview)
    print("-"*80)


def demo_text_only():
    """Demo: Text-only extraction (faster, no vision)"""
    print("\n" + "="*80)
    print("DEMO 2: Text-Only Extraction (Faster)")
    print("="*80 + "\n")
    
    funcs = setup_and_import()
    
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    query = "Python programming features syntax"
    
    print(f"URL: {url}")
    print(f"Query: {query}\n")
    print("This demo uses only text extraction (no screenshot/vision)\n")
    
    proceed = input("Proceed? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Skipped.")
        return
    
    print("\nExtracting text content...")
    text_content = funcs['extract_main_content'](url)
    
    if not text_content:
        print("Failed to extract")
        return
    
    print(f"✓ Extracted {len(text_content['text'])} characters")
    
    # Filter
    print("\nFiltering by query...")
    filtered = funcs['filter_text_by_query'](text_content['markdown'], query)
    
    # Create output
    output = f"""# {text_content['metadata']['title']}

**Query:** {query}
**Source:** {text_content['metadata']['url']}

---

{filtered}
"""
    
    output_file = "demo_python_text_only.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"\n✓ Output saved: {output_file}")
    
    # Preview
    print("\n" + "-"*80)
    print("Preview:")
    print("-"*80)
    preview = output[:600] + "..." if len(output) > 600 else output
    print(preview)


def demo_comparison():
    """Demo: Compare text-only vs vision-enhanced results"""
    print("\n" + "="*80)
    print("DEMO 3: Comparison - Text Only vs AI Vision Enhanced")
    print("="*80 + "\n")
    
    funcs = setup_and_import()
    
    url = "https://en.wikipedia.org/wiki/Basketball"
    query = "basketball court dimensions and scoring rules"
    
    print(f"URL: {url}")
    print(f"Query: {query}\n")
    print("This demo shows the difference between:")
    print("1. Text-only extraction")
    print("2. AI Vision-enhanced extraction\n")
    
    proceed = input("Proceed? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Skipped.")
        return
    
    # Extract text
    print("\nExtracting text content...")
    text_content = funcs['extract_main_content'](url)
    
    if not text_content:
        print("Failed to extract")
        return
    
    # Text-only version
    print("\nCreating text-only version...")
    text_only = funcs['filter_text_by_query'](text_content['markdown'], query)
    
    with open("demo_basketball_text_only.md", 'w') as f:
        f.write(f"# Text-Only Extraction\n\n{text_only}")
    
    print("✓ Text-only version saved")
    
    # Vision-enhanced version
    print("\nCapturing screenshot for vision analysis...")
    screenshot_path = funcs['capture_screenshot'](url, "demo_basketball.png")
    
    if screenshot_path:
        print("\nAnalyzing with Claude Vision...")
        visual_analysis = funcs['analyze_screenshot_with_claude'](screenshot_path, query, url)
        
        if visual_analysis:
            text_content['markdown'] = text_only
            combined = funcs['combine_analyses'](text_content, visual_analysis, query)
            
            with open("demo_basketball_vision_enhanced.md", 'w') as f:
                f.write(combined)
            
            print("✓ Vision-enhanced version saved")
            
            print("\n" + "="*80)
            print("COMPARISON RESULTS")
            print("="*80)
            print(f"\nText-only output: demo_basketball_text_only.md ({len(text_only)} chars)")
            print(f"Vision-enhanced: demo_basketball_vision_enhanced.md ({len(combined)} chars)")
            print("\nKey differences:")
            print("- Vision analysis can identify data in tables, charts, images")
            print("- Vision can understand visual layout and hierarchy")
            print("- Vision provides structured extraction of key statistics")
            print("- Text extraction is faster but may miss visual information")
        else:
            print("⚠ Vision analysis failed")
    else:
        print("⚠ Screenshot capture failed")


def interactive_mode():
    """Let user test with their own URL"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE - AI Vision Analysis")
    print("="*80 + "\n")
    
    funcs = setup_and_import()
    
    try:
        url = input("Enter URL: ").strip()
        query = input("Enter query: ").strip()
        
        if not url or not query:
            print("Error: Both URL and query required")
            return
        
        use_vision = input("\nUse AI vision analysis? (y/n): ").strip().lower() == 'y'
        
        print("\nProcessing...\n")
        
        # Extract text
        text_content = funcs['extract_main_content'](url)
        if not text_content:
            print("Failed to extract content")
            return
        
        print(f"✓ Text extracted")
        
        visual_analysis = None
        
        if use_vision:
            # Screenshot and analyze
            screenshot_path = "interactive_screenshot.png"
            screenshot_path = funcs['capture_screenshot'](url, screenshot_path)
            
            if screenshot_path:
                visual_analysis = funcs['analyze_screenshot_with_claude'](
                    screenshot_path, query, url
                )
                
                if visual_analysis:
                    print(f"✓ Visual analysis complete")
        
        # Filter and combine
        text_content['markdown'] = funcs['filter_text_by_query'](
            text_content['markdown'], query
        )
        
        if visual_analysis:
            output = funcs['combine_analyses'](text_content, visual_analysis, query)
        else:
            output = f"""# {text_content['metadata']['title']}

**Query:** {query}
**Source:** {url}

---

{text_content['markdown']}
"""
        
        # Save
        filename = input("\nSave to file (press Enter for default 'output.md'): ").strip() or "output.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"\n✓ Saved to: {filename}")
        
        # Preview
        show_preview = input("Show preview? (y/n): ").strip().lower() == 'y'
        if show_preview:
            print("\n" + "="*80)
            print(output[:1000] + "..." if len(output) > 1000 else output)
            print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nExiting...")


def main():
    print("="*80)
    print("ENHANCED WEB CONTENT EXTRACTOR - DEMO SUITE")
    print("AI Vision Analysis + Text Extraction")
    print("="*80)
    
    print("\nChoose a demo:")
    print("1. AI Vision Analysis + Text Extraction (comprehensive)")
    print("2. Text-Only Extraction (faster, no vision)")
    print("3. Comparison: Text-Only vs Vision-Enhanced")
    print("4. Interactive Mode (enter your own URL)")
    print("5. Run all demos")
    print("0. Exit")
    
    choice = input("\nEnter choice (1-5, 0 to exit): ").strip()
    
    if choice == '1':
        demo_vision_analysis()
    elif choice == '2':
        demo_text_only()
    elif choice == '3':
        demo_comparison()
    elif choice == '4':
        interactive_mode()
    elif choice == '5':
        demo_vision_analysis()
        demo_text_only()
        demo_comparison()
        print("\n" + "="*80)
        print("All demos completed!")
        print("="*80)
    elif choice == '0':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
