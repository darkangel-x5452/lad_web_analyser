"""
Quick Test Script - Verify your setup is working
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    
    try:
        from playwright.sync_api import sync_playwright
        print("✓ Playwright installed")
    except ImportError:
        print("❌ Playwright not installed")
        print("   Run: pip install playwright --break-system-packages")
        print("   Then: playwright install chromium")
        return False
    
    try:
        import anthropic
        print("✓ Anthropic package installed")
    except ImportError:
        print("⚠️  Anthropic package not installed (needed for Claude)")
        print("   Run: pip install anthropic --break-system-packages")
    
    try:
        import google.generativeai as genai
        print("✓ Google GenAI package installed")
    except ImportError:
        print("⚠️  Google GenAI package not installed (needed for Gemini)")
        print("   Run: pip install google-generativeai --break-system-packages")
    
    try:
        import ollama
        print("✓ Ollama package installed")
    except ImportError:
        print("⚠️  Ollama package not installed (needed for local model)")
        print("   Run: pip install ollama --break-system-packages")
    
    try:
        from PIL import Image
        print("✓ Pillow (PIL) installed")
    except ImportError:
        print("❌ Pillow not installed")
        print("   Run: pip install Pillow --break-system-packages")
        return False
    
    return True

def test_api_keys():
    """Check if API keys are set"""
    import os
    
    print("\nChecking API keys...")
    
    has_claude = os.environ.get("ANTHROPIC_API_KEY")
    has_gemini = os.environ.get("GOOGLE_API_KEY")
    
    if has_claude:
        print("✓ ANTHROPIC_API_KEY is set")
    else:
        print("⚠️  ANTHROPIC_API_KEY not set (needed for Claude)")
        print("   Get key: https://console.anthropic.com/")
        print("   Set: export ANTHROPIC_API_KEY='your-key'")
    
    if has_gemini:
        print("✓ GOOGLE_API_KEY is set")
    else:
        print("⚠️  GOOGLE_API_KEY not set (needed for Gemini)")
        print("   Get key: https://aistudio.google.com/app/apikey")
        print("   Set: export GOOGLE_API_KEY='your-key'")
    
    if not has_claude and not has_gemini:
        print("\n⚠️  No API keys set. You need at least one:")
        print("   - Claude API (best accuracy)")
        print("   - Gemini API (fast & free)")
        print("   - Or use Ollama (local, no API needed)")
        return False
    
    return True

def test_screenshot():
    """Test screenshot functionality"""
    print("\nTesting screenshot capture...")
    
    try:
        from webpage_analyzer import WebpageAnalyzer
        
        # Pick model based on available API keys
        import os
        if os.environ.get("ANTHROPIC_API_KEY"):
            model = "claude"
        elif os.environ.get("GOOGLE_API_KEY"):
            model = "gemini"
        else:
            print("⚠️  No API keys set. Skipping screenshot test.")
            return True
        
        analyzer = WebpageAnalyzer(model=model)
        
        # Take a simple screenshot
        print(f"Taking test screenshot with {model}...")
        analyzer.capture_screenshot(
            "https://example.com",
            "test_screenshot.png",
            wait_time=1,
            full_page=False
        )
        
        print("✓ Screenshot test successful!")
        print("   Test file: test_screenshot.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Screenshot test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("  WEBPAGE ANALYZER - TEST SUITE")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Imports
    results.append(("Package Installation", test_imports()))
    print()
    
    # Test 2: API Keys
    results.append(("API Configuration", test_api_keys()))
    print()
    
    # Test 3: Screenshot
    results.append(("Screenshot Capture", test_screenshot()))
    print()
    
    # Summary
    print("=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! You're ready to use the analyzer.")
        print("\nTry running:")
        print("  python webpage_analyzer.py")
        print("  python example_usage.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nFor help, see README.md or MODEL_COMPARISON.md")
    
    print()

if __name__ == "__main__":
    main()
