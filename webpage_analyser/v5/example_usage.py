"""
Example Usage Scripts for Webpage Analyzer
Demonstrates different use cases and scenarios
"""

from webpage_analyzer import WebpageAnalyzer


def example_sports_stats():
    """Extract sports team statistics"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Sports Statistics")
    print("="*60)
    
    analyzer = WebpageAnalyzer(model="claude")  # Change as needed
    
    # NBA Lakers stats
    result = analyzer.analyze_webpage(
        url="https://www.nba.com/stats/team/1610612747",
        query="Get me the LA Lakers team statistics including record, points per game, and field goal percentage",
        screenshot_path="lakers_stats.png"
    )
    
    print(result)
    
    # Save to file
    with open("lakers_analysis.md", "w") as f:
        f.write(result)
    
    return result


def example_product_specs():
    """Extract product specifications"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Product Specifications")
    print("="*60)
    
    analyzer = WebpageAnalyzer(model="gemini")  # Gemini is fast for this
    
    # Product page
    result = analyzer.analyze_webpage(
        url="https://www.apple.com/iphone-16-pro/specs/",
        query="Extract the complete technical specifications including display, chip, camera, and battery details",
        screenshot_path="iphone_specs.png"
    )
    
    print(result)
    return result


def example_pricing_comparison():
    """Extract pricing tables"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Pricing Comparison")
    print("="*60)
    
    analyzer = WebpageAnalyzer(model="claude")
    
    result = analyzer.analyze_webpage(
        url="https://www.anthropic.com/pricing",
        query="Extract the pricing comparison table showing all plans, their features, and monthly costs",
        screenshot_path="pricing.png"
    )
    
    print(result)
    return result


def example_news_article():
    """Extract key points from news article"""
    print("\n" + "="*60)
    print("EXAMPLE 4: News Article Summary")
    print("="*60)
    
    analyzer = WebpageAnalyzer(model="gemini")
    
    result = analyzer.analyze_webpage(
        url="https://www.bbc.com/news",
        query="Extract the top 3 news headlines with brief summaries from the main content area",
        screenshot_path="news.png"
    )
    
    print(result)
    return result


def example_financial_data():
    """Extract financial data"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Financial Data")
    print("="*60)
    
    analyzer = WebpageAnalyzer(model="claude")
    
    result = analyzer.analyze_webpage(
        url="https://finance.yahoo.com/quote/AAPL",
        query="Extract the current stock price, market cap, P/E ratio, and 52-week range",
        screenshot_path="stock.png"
    )
    
    print(result)
    return result


def example_restaurant_menu():
    """Extract restaurant menu"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Restaurant Menu")
    print("="*60)
    
    analyzer = WebpageAnalyzer(model="gemini")
    
    result = analyzer.analyze_webpage(
        url="https://www.mcdonalds.com/us/en-us/full-menu.html",
        query="Extract the burger menu items with their names and descriptions",
        screenshot_path="menu.png"
    )
    
    print(result)
    return result


def example_batch_processing():
    """Process multiple pages"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Batch Processing")
    print("="*60)
    
    analyzer = WebpageAnalyzer(model="gemini")  # Gemini has generous free tier
    
    urls_and_queries = [
        ("https://www.nba.com/stats/team/1610612747", "Get Lakers season record"),
        ("https://www.nba.com/stats/team/1610612738", "Get Celtics season record"),
        ("https://www.nba.com/stats/team/1610612744", "Get Warriors season record"),
    ]
    
    results = []
    for i, (url, query) in enumerate(urls_and_queries):
        print(f"\nProcessing {i+1}/{len(urls_and_queries)}: {url}")
        result = analyzer.analyze_webpage(
            url=url,
            query=query,
            screenshot_path=f"batch_{i}.png"
        )
        results.append(result)
        print(result)
        print("-" * 60)
    
    # Save all results
    with open("batch_results.md", "w") as f:
        f.write("# Batch Analysis Results\n\n")
        for i, result in enumerate(results):
            f.write(f"## Result {i+1}\n\n")
            f.write(result)
            f.write("\n\n---\n\n")
    
    return results


def example_custom_screenshot():
    """Custom screenshot settings"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Custom Screenshot Settings")
    print("="*60)
    
    analyzer = WebpageAnalyzer(model="claude")
    
    # First capture screenshot with custom settings
    analyzer.capture_screenshot(
        url="https://www.wikipedia.org",
        output_path="wikipedia_custom.png",
        wait_time=5,  # Wait 5 seconds for full load
        full_page=False  # Just visible area
    )
    
    # Then analyze it
    result = analyzer.analyze_screenshot(
        image_path="wikipedia_custom.png",
        query="Extract the featured article title and first paragraph"
    )
    
    print(result)
    return result


def compare_models():
    """Compare different models on same task"""
    print("\n" + "="*60)
    print("EXAMPLE 9: Model Comparison")
    print("="*60)
    
    url = "https://www.nba.com/stats/team/1610612747"
    query = "Get the Lakers win-loss record"
    
    models = ["claude", "gemini"]  # Add "ollama" if you have it setup
    
    for model in models:
        try:
            print(f"\n--- Testing {model.upper()} ---")
            analyzer = WebpageAnalyzer(model=model)
            result = analyzer.analyze_webpage(url, query, 
                                            screenshot_path=f"test_{model}.png")
            print(result)
        except Exception as e:
            print(f"Error with {model}: {e}")


def main():
    """Run examples"""
    print("="*60)
    print("WEBPAGE ANALYZER - EXAMPLE USAGE")
    print("="*60)
    print("\nChoose an example to run:")
    print("1. Sports Statistics (Lakers)")
    print("2. Product Specifications (iPhone)")
    print("3. Pricing Comparison")
    print("4. News Headlines")
    print("5. Financial Data (Stock)")
    print("6. Restaurant Menu")
    print("7. Batch Processing (Multiple Teams)")
    print("8. Custom Screenshot Settings")
    print("9. Compare Models")
    print("0. Run All Examples")
    
    choice = input("\nEnter choice (0-9): ").strip()
    
    examples = {
        "1": example_sports_stats,
        "2": example_product_specs,
        "3": example_pricing_comparison,
        "4": example_news_article,
        "5": example_financial_data,
        "6": example_restaurant_menu,
        "7": example_batch_processing,
        "8": example_custom_screenshot,
        "9": compare_models,
    }
    
    if choice == "0":
        for func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"Error: {e}")
    elif choice in examples:
        examples[choice]()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    # Quick example
    print("\n🚀 Quick Start Example:")
    
    analyzer = WebpageAnalyzer(model="claude")  # Change to your preferred model
    
    result = analyzer.analyze_webpage(
        url="https://www.nba.com/stats/team/1610612747",
        query="Get me the LA Lakers team statistics"
    )
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(result)
    
    # Uncomment to run interactive menu
    # main()
