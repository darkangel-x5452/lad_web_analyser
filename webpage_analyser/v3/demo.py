#!/usr/bin/env python3
"""
Demo script showing how to use the web_content_extractor module
"""

import sys
from web_content_extractor import (
    setup_dependencies,
    extract_main_content,
    format_output
)

def demo_basic_extraction():
    """Basic example: Extract content from a URL with a query"""
    print("\n" + "="*80)
    print("DEMO 1: Basic Content Extraction")
    print("="*80 + "\n")
    
    # Example URLs you can test with
    examples = [
        {
            'url': 'https://en.wikipedia.org/wiki/Los_Angeles_Lakers',
            'query': 'LA Lakers championships and statistics'
        },
        {
            'url': 'https://en.wikipedia.org/wiki/Python_(programming_language)',
            'query': 'Python programming language features and history'
        },
        {
            'url': 'https://en.wikipedia.org/wiki/Climate_change',
            'query': 'climate change temperature increase statistics'
        }
    ]
    
    # Use the first example
    example = examples[0]
    
    print(f"URL: {example['url']}")
    print(f"Query: {example['query']}\n")
    print("Extracting content...\n")
    
    # Extract content
    content_dict = extract_main_content(example['url'])
    
    if content_dict:
        # Format and display
        output = format_output(content_dict, example['query'], filter_by_query=True)
        print(output)
        print("\n✓ Extraction successful!")
    else:
        print("✗ Failed to extract content")


def demo_custom_usage():
    """Show how to use the extractor programmatically"""
    print("\n" + "="*80)
    print("DEMO 2: Programmatic Usage")
    print("="*80 + "\n")
    
    # Custom URL and query
    url = "https://en.wikipedia.org/wiki/Basketball"
    query = "basketball court dimensions and rules"
    
    print(f"Extracting from: {url}")
    print(f"Looking for: {query}\n")
    
    # Extract
    content = extract_main_content(url)
    
    if content:
        print(f"Title: {content['metadata']['title']}")
        print(f"Text length: {len(content['text'])} characters")
        print(f"Markdown length: {len(content['markdown'])} characters")
        
        # Show preview
        print("\nPreview of extracted content:")
        print("-" * 80)
        preview = content['markdown'][:500] + "..." if len(content['markdown']) > 500 else content['markdown']
        print(preview)
        print("-" * 80)
        
        # Save to file
        output = format_output(content, query, filter_by_query=True)
        
        filename = "extracted_content_demo.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"\n✓ Full content saved to: {filename}")
    else:
        print("✗ Failed to extract content")


def demo_multiple_queries():
    """Extract content for multiple queries from the same source"""
    print("\n" + "="*80)
    print("DEMO 3: Multiple Queries on Same URL")
    print("="*80 + "\n")
    
    url = "https://en.wikipedia.org/wiki/National_Basketball_Association"
    
    queries = [
        "NBA teams and conferences",
        "NBA history and founding",
        "NBA championship statistics"
    ]
    
    print(f"Source: {url}\n")
    
    # Extract once
    content = extract_main_content(url)
    
    if content:
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 40)
            
            # Format for this specific query
            output = format_output(content, query, filter_by_query=True)
            
            # Save to separate file
            filename = f"query_{i}_output.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(output)
            
            print(f"✓ Saved to {filename}")
        
        print("\n✓ All queries processed!")
    else:
        print("✗ Failed to extract content")


def interactive_mode():
    """Let user input their own URL and query"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80 + "\n")
    
    try:
        url = input("Enter URL to extract from: ").strip()
        query = input("Enter your query: ").strip()
        
        if not url or not query:
            print("Error: Both URL and query are required")
            return
        
        print("\nExtracting content...\n")
        
        content = extract_main_content(url)
        
        if content:
            output = format_output(content, query, filter_by_query=True)
            
            # Ask if user wants to save
            save = input("\nSave to file? (y/n): ").strip().lower()
            
            if save == 'y':
                filename = input("Enter filename (default: output.md): ").strip() or "output.md"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(output)
                print(f"✓ Saved to {filename}")
            else:
                print("\nExtracted content:")
                print("=" * 80)
                print(output)
        else:
            print("✗ Failed to extract content")
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


def main():
    # Setup dependencies
    print("Setting up dependencies...")
    setup_dependencies()
    
    print("\n" + "="*80)
    print("WEB CONTENT EXTRACTOR - DEMO SUITE")
    print("="*80)
    
    # Menu
    print("\nChoose a demo:")
    print("1. Basic extraction example")
    print("2. Programmatic usage")
    print("3. Multiple queries on same URL")
    print("4. Interactive mode (enter your own URL)")
    print("5. Run all demos")
    print("0. Exit")
    
    choice = input("\nEnter choice (1-5, 0 to exit): ").strip()
    
    if choice == '1':
        demo_basic_extraction()
    elif choice == '2':
        demo_custom_usage()
    elif choice == '3':
        demo_multiple_queries()
    elif choice == '4':
        interactive_mode()
    elif choice == '5':
        demo_basic_extraction()
        demo_custom_usage()
        demo_multiple_queries()
        print("\n" + "="*80)
        print("All demos completed!")
        print("="*80)
    elif choice == '0':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice. Please run again.")


if __name__ == '__main__':
    main()
