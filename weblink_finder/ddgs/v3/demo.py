#!/usr/bin/env python3
"""
Demo script showing various use cases of the Smart Link Scraper
Run this to see the scraper in action!
"""

import sys

# Check if dependencies are available
try:
    from duckduckgo_search import DDGS
    from advanced_scraper import AdvancedLinkScraper
    USE_ADVANCED = True
except ImportError:
    print("⚠️  Advanced dependencies not installed.")
    print("   Install with: pip install -r requirements.txt")
    print("   Falling back to basic scraper...\n")
    USE_ADVANCED = False


def demo_sports_teams():
    """Demo: Finding team statistics"""
    print("=" * 100)
    print("DEMO 1: Finding Specific Team Statistics")
    print("=" * 100)
    
    if not USE_ADVANCED:
        print("❌ Please install dependencies to run demos")
        return
    
    scraper = AdvancedLinkScraper(max_results=20)
    
    queries = [
        "LA Lakers team statistics NBA 2024",
        "Golden State Warriors roster NBA",
        "Boston Celtics schedule 2024 NBA"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 100)
        
        results = scraper.search(query)
        
        if results:
            print(f"✅ Found {len(results)} relevant links\n")
            
            # Show top 3 results
            for j, result in enumerate(results[:3], 1):
                print(f"{j}. {result['title']}")
                print(f"   🔗 {result['url']}")
                print(f"   📊 Score: {result['relevance_score']:.1f}\n")
        else:
            print("❌ No relevant results found\n")
        
        if i < len(queries):
            print("\n" + "▼" * 50 + "\n")


def demo_soccer_teams():
    """Demo: Finding soccer/football team info"""
    print("\n\n")
    print("=" * 100)
    print("DEMO 2: Finding Soccer Team Information")
    print("=" * 100)
    
    if not USE_ADVANCED:
        return
    
    scraper = AdvancedLinkScraper(max_results=20)
    
    queries = [
        "Manchester United squad roster Premier League 2024",
        "Real Madrid Champions League standings",
        "Barcelona player stats La Liga 2024"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 100)
        
        results = scraper.search(query)
        
        if results:
            print(f"✅ Found {len(results)} relevant links\n")
            
            # Show top 3 results
            for j, result in enumerate(results[:3], 1):
                print(f"{j}. {result['title']}")
                print(f"   🔗 {result['url']}")
                print(f"   📊 Score: {result['relevance_score']:.1f}\n")


def demo_comparison():
    """Demo: Show how filtering improves results"""
    print("\n\n")
    print("=" * 100)
    print("DEMO 3: Filtering Comparison - Generic vs Specific Links")
    print("=" * 100)
    
    if not USE_ADVANCED:
        return
    
    query = "LA Lakers NBA statistics"
    
    print(f"\n📋 Query: {query}")
    print("\nWithout Smart Filtering (typical search results):")
    print("-" * 100)
    
    # Simulate typical search results
    generic_results = [
        {"title": "NBA.com - Official Site", "url": "https://www.nba.com/", "score": 0},
        {"title": "Los Angeles Lakers", "url": "https://www.nba.com/lakers", "score": 0},
        {"title": "ESPN - NBA", "url": "https://www.espn.com/nba/", "score": 0},
        {"title": "Lakers Homepage", "url": "https://www.lakers.com/", "score": 0},
    ]
    
    for i, result in enumerate(generic_results, 1):
        print(f"{i}. {result['title']}")
        print(f"   🔗 {result['url']}")
        print(f"   ⚠️  Generic homepage - not specific data\n")
    
    print("\n" + "🎯" * 50)
    print("\nWith Smart Filtering (our scraper):")
    print("-" * 100)
    
    scraper = AdvancedLinkScraper(max_results=20)
    results = scraper.search(query)
    
    if results:
        for i, result in enumerate(results[:4], 1):
            print(f"{i}. {result['title']}")
            print(f"   🔗 {result['url']}")
            print(f"   ✅ Score: {result['relevance_score']:.1f} - Specific data page\n")


def demo_save_results():
    """Demo: Saving results to files"""
    print("\n\n")
    print("=" * 100)
    print("DEMO 4: Saving Results to Files")
    print("=" * 100)
    
    if not USE_ADVANCED:
        return
    
    scraper = AdvancedLinkScraper(max_results=20)
    query = "Golden State Warriors player stats NBA 2024"
    
    print(f"\n📋 Searching: {query}")
    results = scraper.search(query)
    
    if results:
        # Save as JSON
        json_file = "demo_results.json"
        scraper.save_results(results, query, json_file)
        print(f"✅ Saved to JSON: {json_file}")
        
        # Save as text
        txt_file = "demo_results.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Total Results: {len(results)}\n\n")
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {result['title']}\n")
                f.write(f"   {result['url']}\n")
                f.write(f"   Score: {result['relevance_score']:.1f}\n\n")
        print(f"✅ Saved to text: {txt_file}")
        
        print(f"\n📊 Summary:")
        print(f"   Total links found: {len(results)}")
        print(f"   Average relevance score: {sum(r['relevance_score'] for r in results) / len(results):.1f}")
        print(f"   Highest score: {max(r['relevance_score'] for r in results):.1f}")


def show_menu():
    """Show demo menu"""
    print("\n" + "=" * 100)
    print("🎯 SMART LINK SCRAPER - DEMO MENU")
    print("=" * 100)
    print("\nChoose a demo to run:")
    print("  [1] Basketball Teams (NBA)")
    print("  [2] Soccer Teams (Premier League, Champions League)")
    print("  [3] Comparison: Generic vs Specific Links")
    print("  [4] Saving Results to Files")
    print("  [5] Run All Demos")
    print("  [q] Quit")
    print("\n" + "=" * 100)


def main():
    """Main demo runner"""
    print("\n")
    print("🎯" * 50)
    print("\n         SMART LINK SCRAPER - DEMONSTRATION")
    print("         Finding Specific, Relevant Links")
    print("\n" + "🎯" * 50)
    
    if not USE_ADVANCED:
        print("\n❌ Please install dependencies first:")
        print("   pip install -r requirements.txt\n")
        sys.exit(1)
    
    while True:
        show_menu()
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == '1':
            demo_sports_teams()
        elif choice == '2':
            demo_soccer_teams()
        elif choice == '3':
            demo_comparison()
        elif choice == '4':
            demo_save_results()
        elif choice == '5':
            demo_sports_teams()
            demo_soccer_teams()
            demo_comparison()
            demo_save_results()
        elif choice in ['q', 'quit', 'exit']:
            print("\n👋 Thanks for trying Smart Link Scraper!")
            break
        else:
            print("\n❌ Invalid option. Please try again.")
        
        if choice != 'q':
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
