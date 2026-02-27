#!/usr/bin/env python3
"""
Quick test script to validate the web content extractor installation
"""

import sys

def test_installation():
    """Test that all dependencies are installed and working"""
    print("Testing Web Content Extractor Installation...")
    print("=" * 60)
    
    # Test imports
    print("\n1. Testing dependencies...")
    try:
        import trafilatura
        print("   ✓ trafilatura installed")
    except ImportError:
        print("   ✗ trafilatura not found - run: pip install trafilatura --break-system-packages")
        return False
    
    try:
        import requests
        print("   ✓ requests installed")
    except ImportError:
        print("   ✗ requests not found - run: pip install requests --break-system-packages")
        return False
    
    try:
        from bs4 import BeautifulSoup
        print("   ✓ beautifulsoup4 installed")
    except ImportError:
        print("   ✗ beautifulsoup4 not found - run: pip install beautifulsoup4 --break-system-packages")
        return False
    
    try:
        import lxml
        print("   ✓ lxml installed")
    except ImportError:
        print("   ✗ lxml not found - run: pip install lxml --break-system-packages")
        return False
    
    # Test basic extraction
    print("\n2. Testing basic extraction...")
    try:
        # Simple test with a reliable public page
        test_url = "https://example.com"
        response = requests.get(test_url, timeout=5)
        
        if response.status_code == 200:
            print(f"   ✓ Successfully fetched {test_url}")
            
            # Test trafilatura extraction
            extracted = trafilatura.extract(response.text)
            if extracted and len(extracted) > 10:
                print(f"   ✓ Successfully extracted content ({len(extracted)} chars)")
            else:
                print("   ⚠ Extraction returned minimal content")
        else:
            print(f"   ✗ Failed to fetch URL (status: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"   ✗ Extraction test failed: {e}")
        return False
    
    # Test markdown conversion
    print("\n3. Testing markdown conversion...")
    try:
        markdown = trafilatura.extract(response.text, output_format='markdown')
        if markdown and '# ' in markdown or '## ' in markdown:
            print("   ✓ Markdown conversion working")
        else:
            print("   ⚠ Markdown conversion returned unexpected format")
    except Exception as e:
        print(f"   ✗ Markdown conversion failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! The extractor is ready to use.")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Run: python demo.py (for interactive examples)")
    print("  • Run: python web_content_extractor.py -u URL -q QUERY")
    print("  • Read: README.md (for full documentation)")
    
    return True


def quick_demo():
    """Run a quick demonstration"""
    print("\n" + "=" * 60)
    print("Quick Demo: Extracting from example.com")
    print("=" * 60 + "\n")
    
    try:
        from web_content_extractor import extract_main_content, format_output
        
        url = "https://example.com"
        query = "example domain information"
        
        print(f"URL: {url}")
        print(f"Query: {query}\n")
        print("Extracting...\n")
        
        content = extract_main_content(url)
        if content:
            output = format_output(content, query, filter_by_query=False)
            print(output)
            print("\n✅ Demo successful!")
        else:
            print("❌ Demo failed - could not extract content")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == '__main__':
    success = test_installation()
    
    if success:
        # Ask if user wants to run demo
        try:
            run_demo = input("\nRun quick demo? (y/n): ").strip().lower()
            if run_demo == 'y':
                quick_demo()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
    else:
        print("\n❌ Installation test failed. Please install missing dependencies.")
        print("Run: pip install -r requirements.txt --break-system-packages")
        sys.exit(1)
