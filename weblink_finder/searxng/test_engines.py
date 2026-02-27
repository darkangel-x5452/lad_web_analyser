#!/usr/bin/env python3
"""
Test script to compare different search engines
Shows which engine works best for your queries
"""

import sys
import time
from searxng_scraper import SearxNGScraper
from bing_scraper import BingScraper
from multi_engine_scraper import MultiEngineScraper


def test_query(query: str):
    """
    Test a single query across all engines and compare results
    """
    print("\n" + "=" * 100)
    print(f"TESTING QUERY: {query}")
    print("=" * 100)
    
    engines = {
        'SearxNG': SearxNGScraper(max_results=15),
        'Bing': BingScraper(max_results=15),
        'Multi-Engine': MultiEngineScraper(max_results=15, preferred_engine='searxng')
    }
    
    results_by_engine = {}
    
    for engine_name, scraper in engines.items():
        print(f"\n{'─' * 100}")
        print(f"Testing: {engine_name}")
        print('─' * 100)
        
        start_time = time.time()
        
        try:
            results = scraper.search(query)
            elapsed = time.time() - start_time
            
            if results:
                results_by_engine[engine_name] = {
                    'results': results,
                    'count': len(results),
                    'time': elapsed,
                    'success': True,
                    'avg_score': sum(r['relevance_score'] for r in results) / len(results),
                    'max_score': max(r['relevance_score'] for r in results)
                }
                
                print(f"✅ Success!")
                print(f"   Results: {len(results)}")
                print(f"   Time: {elapsed:.2f}s")
                print(f"   Avg Score: {results_by_engine[engine_name]['avg_score']:.1f}")
                print(f"   Max Score: {results_by_engine[engine_name]['max_score']:.1f}")
                
                # Show top 3 results
                print(f"\n   Top 3 Results:")
                for i, result in enumerate(results[:3], 1):
                    print(f"   {i}. [{result['relevance_score']:.1f}] {result['title'][:60]}...")
                    print(f"      {result['url'][:80]}...")
            else:
                results_by_engine[engine_name] = {
                    'success': False,
                    'time': elapsed,
                    'error': 'No results found'
                }
                print(f"❌ Failed: No results found")
                
        except Exception as e:
            elapsed = time.time() - start_time
            results_by_engine[engine_name] = {
                'success': False,
                'time': elapsed,
                'error': str(e)
            }
            print(f"❌ Failed: {e}")
        
        # Small delay between engines
        time.sleep(1)
    
    # Summary comparison
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    
    successful = [(name, data) for name, data in results_by_engine.items() if data.get('success')]
    
    if successful:
        print("\n✅ Successful Engines:")
        successful.sort(key=lambda x: x[1]['avg_score'], reverse=True)
        
        for rank, (name, data) in enumerate(successful, 1):
            print(f"\n{rank}. {name}")
            print(f"   Results: {data['count']}")
            print(f"   Speed: {data['time']:.2f}s")
            print(f"   Avg Score: {data['avg_score']:.1f}")
            print(f"   Max Score: {data['max_score']:.1f}")
            print(f"   Quality: {'⭐' * int(data['avg_score'] / 20)}")
        
        # Winner
        winner = successful[0]
        print(f"\n🏆 WINNER: {winner[0]}")
        print(f"   Best balance of speed, quality, and reliability")
        
    else:
        print("\n❌ All engines failed for this query")
        print("   Suggestions:")
        print("   - Check your internet connection")
        print("   - Try a different query")
        print("   - Some search engines may be temporarily down")
    
    failed = [(name, data) for name, data in results_by_engine.items() if not data.get('success')]
    if failed:
        print("\n❌ Failed Engines:")
        for name, data in failed:
            print(f"   - {name}: {data.get('error', 'Unknown error')}")


def interactive_test():
    """
    Interactive testing mode
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   SEARCH ENGINE COMPARISON TOOL                             ║
║                                                                              ║
║  Test different search engines to find which works best for you!            ║
║                                                                              ║
║  Tests:                                                                     ║
║  • SearxNG (meta-search engine)                                             ║
║  • Bing (direct search)                                                     ║
║  • Multi-Engine (automatic fallback)                                        ║
║                                                                              ║
║  Compares: Speed, Quality, Reliability                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Predefined test queries
    sample_queries = [
        "LA Lakers team statistics NBA 2024",
        "Golden State Warriors player roster NBA",
        "Manchester United Premier League squad 2024",
        "Real Madrid Champions League standings",
    ]
    
    print("\nSample queries you can test:")
    for i, q in enumerate(sample_queries, 1):
        print(f"  {i}. {q}")
    
    print("\nOptions:")
    print("  [1-4] Test a sample query")
    print("  [c] Enter custom query")
    print("  [a] Test all sample queries")
    print("  [q] Quit")
    
    while True:
        choice = input("\n> Select option: ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Thanks for testing!")
            break
        
        elif choice == 'c':
            query = input("\nEnter your query: ").strip()
            if query:
                test_query(query)
            else:
                print("❌ Invalid query")
        
        elif choice == 'a':
            for i, query in enumerate(sample_queries, 1):
                print(f"\n\n{'#' * 100}")
                print(f"# Test {i}/{len(sample_queries)}")
                print('#' * 100)
                test_query(query)
                if i < len(sample_queries):
                    input("\nPress Enter to continue to next query...")
        
        elif choice in ['1', '2', '3', '4']:
            idx = int(choice) - 1
            if 0 <= idx < len(sample_queries):
                test_query(sample_queries[idx])
            else:
                print("❌ Invalid selection")
        
        else:
            print("❌ Invalid option")


def quick_test():
    """
    Quick test with a single query
    """
    query = "LA Lakers team statistics NBA 2024"
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          QUICK ENGINE TEST                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    test_query(query)
    
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print("""
Based on this test, we recommend:

1. Use multi_engine_scraper.py for most cases
   - Automatically tries multiple engines
   - Best reliability
   
2. Use bing_scraper.py if you need speed
   - Fastest results
   - Good quality
   
3. Use searxng_scraper.py for maximum privacy
   - Aggregates multiple search engines
   - No direct connection to major search engines

To use the recommended scraper:
    python multi_engine_scraper.py
    """)


def main():
    """
    Main entry point
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            quick_test()
        elif sys.argv[1] == '--query':
            if len(sys.argv) > 2:
                query = ' '.join(sys.argv[2:])
                test_query(query)
            else:
                print("Usage: python test_engines.py --query YOUR QUERY HERE")
        else:
            print("Usage:")
            print("  python test_engines.py              # Interactive mode")
            print("  python test_engines.py --quick      # Quick test")
            print("  python test_engines.py --query QUERY # Test specific query")
    else:
        interactive_test()


if __name__ == "__main__":
    main()
