#!/usr/bin/env python3
"""
Interactive CLI for Smart Link Scraper
Allows users to input custom queries and get results
"""

import sys
try:
    from duckduckgo_search import DDGS
    USE_ADVANCED = True
except ImportError:
    USE_ADVANCED = False
    print("⚠️  duckduckgo-search not installed. Using basic scraper.")
    print("   Install with: pip install duckduckgo-search")

from urllib.parse import urlparse
import re
from typing import List, Dict
import json


class InteractiveScraper:
    def __init__(self, max_results: int = 20):
        self.max_results = max_results
        
    def calculate_relevance_score(self, result: Dict[str, str], query: str) -> float:
        """Calculate relevance score for a search result"""
        url = result['url'].lower()
        title = result['title'].lower()
        snippet = result.get('snippet', '').lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Extract keywords
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        stop_words = {'the', 'a', 'an', 'for', 'from', 'get', 'to', 'of', 'in', 'on', 'at', 'by'}
        query_words -= stop_words
        
        # Parse URL
        parsed = urlparse(url)
        path = parsed.path.lower()
        path_parts = [p for p in path.strip('/').split('/') if p]
        depth = len(path_parts)
        
        # Scoring factors
        score += depth * 4
        
        if path in ['', '/'] and not parsed.query:
            score -= 50
        
        url_matches = sum(1 for word in query_words if word in url)
        score += url_matches * 15
        
        path_matches = sum(1 for word in query_words if word in path)
        score += path_matches * 10
        
        title_matches = sum(1 for word in query_words if word in title)
        score += title_matches * 12
        
        if query_lower in title:
            score += 20
        
        snippet_matches = sum(1 for word in query_words if word in snippet)
        score += snippet_matches * 4
        
        sports_indicators = [
            'stats', 'statistics', 'roster', 'schedule', 'standings', 'team',
            'player', 'season', 'results', 'scores', 'game', 'match',
            'league', 'championship', 'playoff', 'record', 'ranking'
        ]
        
        url_indicators = sum(1 for indicator in sports_indicators if indicator in url)
        score += url_indicators * 8
        
        authoritative_sites = {
            'espn.com': 20, 'nba.com': 20, 'nfl.com': 20, 'mlb.com': 20,
            'basketball-reference.com': 25, 'sports-reference.com': 25,
            'statmuse.com': 18, 'ncaa.com': 15, 'teamrankings.com': 15,
        }
        
        for site, bonus in authoritative_sites.items():
            if site in parsed.netloc and depth >= 2:
                score += bonus
                break
        
        return score
    
    def search_ddgs(self, query: str) -> List[Dict[str, str]]:
        """Search using duckduckgo-search library"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region='us-en',
                    safesearch='moderate',
                    max_results=self.max_results
                ))
                
                formatted_results = []
                for r in results:
                    formatted_results.append({
                        'url': r.get('href', r.get('link', '')),
                        'title': r.get('title', ''),
                        'snippet': r.get('body', r.get('snippet', ''))
                    })
                
                return formatted_results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def filter_and_rank(self, results: List[Dict[str, str]], 
                       query: str,
                       min_score: float = 15) -> List[Dict[str, str]]:
        """Filter and rank results by relevance"""
        scored_results = []
        for result in results:
            score = self.calculate_relevance_score(result, query)
            result['relevance_score'] = score
            scored_results.append(result)
        
        scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        filtered = [r for r in scored_results if r['relevance_score'] >= min_score]
        
        # Deduplicate by domain
        domain_best = {}
        for result in filtered:
            domain = urlparse(result['url']).netloc
            if domain not in domain_best or result['relevance_score'] > domain_best[domain]['relevance_score']:
                domain_best[domain] = result
        
        final_results = list(domain_best.values())
        final_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return final_results
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """Main search method"""
        if not USE_ADVANCED:
            print("❌ Advanced search library not available")
            print("   Please install: pip install duckduckgo-search")
            return []
        
        print(f"\n🔍 Searching for: {query}")
        results = self.search_ddgs(query)
        
        if not results:
            print("⚠️  No results found.")
            return []
        
        print(f"📊 Found {len(results)} raw results, filtering...")
        filtered_results = self.filter_and_rank(results, query)
        
        return filtered_results


def print_results(results: List[Dict[str, str]], show_all: bool = False):
    """Print results in a nice format"""
    if not results:
        print("\n❌ No highly relevant results found.")
        print("💡 Tip: Try refining your query with more specific keywords")
        return
    
    max_display = len(results) if show_all else min(10, len(results))
    
    print(f"\n✅ Found {len(results)} highly relevant links")
    print(f"📋 Showing top {max_display} results:\n")
    print("=" * 100)
    
    for i, result in enumerate(results[:max_display], 1):
        print(f"\n{i}. 📌 {result['title']}")
        print(f"   🔗 {result['url']}")
        print(f"   📊 Score: {result['relevance_score']:.1f}")
        
        if result.get('snippet'):
            snippet = result['snippet']
            if len(snippet) > 150:
                snippet = snippet[:150] + "..."
            print(f"   💬 {snippet}")
        
        print("-" * 100)
    
    if len(results) > max_display:
        print(f"\n💡 {len(results) - max_display} more results available")


def save_results_to_file(results: List[Dict[str, str]], query: str, filename: str):
    """Save results to a file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"SEARCH QUERY: {query}\n")
        f.write(f"TOTAL RESULTS: {len(results)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result['title']}\n")
            f.write(f"   URL: {result['url']}\n")
            f.write(f"   Relevance Score: {result['relevance_score']:.1f}\n")
            if result.get('snippet'):
                f.write(f"   Description: {result['snippet'][:200]}\n")
            f.write("\n")
    
    print(f"\n💾 Results saved to: {filename}")


def interactive_mode():
    """Run interactive CLI mode"""
    print("=" * 100)
    print("🎯 SMART LINK SCRAPER - Interactive Mode")
    print("=" * 100)
    print("\nFinds highly relevant, specific links (not generic homepages)")
    print("Perfect for sports stats, team info, player data, and more!\n")
    
    if not USE_ADVANCED:
        print("⚠️  Warning: Advanced features not available")
        print("   Install dependencies: pip install -r requirements.txt\n")
        return
    
    scraper = InteractiveScraper(max_results=25)
    
    while True:
        print("\n" + "-" * 100)
        query = input("\n🔍 Enter your search query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using Smart Link Scraper!")
            break
        
        if not query:
            print("❌ Please enter a valid query")
            continue
        
        # Search
        results = scraper.search(query)
        
        # Display results
        print_results(results, show_all=False)
        
        # Options menu
        if results:
            print("\n📋 Options:")
            print("  [s] Save results to file")
            print("  [a] Show all results")
            print("  [n] New search")
            
            choice = input("\nSelect option (or press Enter for new search): ").strip().lower()
            
            if choice == 's':
                filename = input("Enter filename (default: results.txt): ").strip()
                if not filename:
                    filename = 'results.txt'
                save_results_to_file(results, query, filename)
            
            elif choice == 'a':
                print_results(results, show_all=True)
            


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Command-line mode
        query = ' '.join(sys.argv[1:])
        
        if not USE_ADVANCED:
            print("❌ Please install dependencies: pip install -r requirements.txt")
            sys.exit(1)
        
        scraper = InteractiveScraper(max_results=25)
        results = scraper.search(query)
        print_results(results, show_all=True)
        
        # Auto-save
        save_results_to_file(results, query, 'search_results.txt')
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
