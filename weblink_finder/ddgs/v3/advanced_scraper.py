#!/usr/bin/env python3
"""
Advanced Smart Link Scraper - Uses duckduckgo-search library
More reliable and feature-rich version
"""

from duckduckgo_search import DDGS
from urllib.parse import urlparse
import re
from typing import List, Dict
import json


class AdvancedLinkScraper:
    def __init__(self, max_results: int = 20):
        self.max_results = max_results
        
    def search_with_ddgs(self, query: str, region: str = 'us-en') -> List[Dict[str, str]]:
        """
        Search using duckduckgo-search library
        
        Args:
            query: Search query
            region: Region code (e.g., 'us-en', 'uk-en')
        
        Returns:
            List of search results
        """
        print(f"🔍 Searching with DuckDuckGo API for: {query}")
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region=region,
                    safesearch='moderate',
                    max_results=self.max_results
                ))
                
                # Convert to consistent format
                formatted_results = []
                for r in results:
                    formatted_results.append({
                        'url': r.get('href', r.get('link', '')),
                        'title': r.get('title', ''),
                        'snippet': r.get('body', r.get('snippet', ''))
                    })
                
                return formatted_results
                
        except Exception as e:
            print(f"❌ DuckDuckGo search error: {e}")
            return []
    
    def calculate_relevance_score(self, result: Dict[str, str], query: str) -> float:
        """
        Enhanced relevance scoring algorithm
        """
        url = result['url'].lower()
        title = result['title'].lower()
        snippet = result['snippet'].lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Extract and clean query keywords
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'for', 'from', 'get', 'to', 'of', 'in', 'on', 'at', 'by'}
        query_words -= stop_words
        
        # Parse URL components
        parsed = urlparse(url)
        path = parsed.path.lower()
        path_parts = [p for p in path.strip('/').split('/') if p]
        depth = len(path_parts)
        
        # 1. URL DEPTH ANALYSIS (Deeper = More Specific)
        score += depth * 4
        
        # Heavily penalize root/homepage URLs
        if path in ['', '/'] and not parsed.query:
            score -= 50
        
        # 2. KEYWORD MATCHING IN URL (Most Important)
        url_matches = sum(1 for word in query_words if word in url)
        score += url_matches * 15
        
        # Bonus for keywords in path segments (not just domain)
        path_matches = sum(1 for word in query_words if word in path)
        score += path_matches * 10
        
        # 3. KEYWORD MATCHING IN TITLE
        title_matches = sum(1 for word in query_words if word in title)
        score += title_matches * 12
        
        # Exact phrase match in title
        if query_lower in title:
            score += 20
        
        # 4. KEYWORD MATCHING IN SNIPPET
        snippet_matches = sum(1 for word in query_words if word in snippet)
        score += snippet_matches * 4
        
        # 5. SPECIFIC CONTENT INDICATORS
        # Sports-specific indicators
        sports_indicators = [
            'stats', 'statistics', 'roster', 'schedule', 'standings', 'team',
            'player', 'season', 'results', 'scores', 'game', 'match',
            'league', 'championship', 'playoff', 'record', 'ranking',
            'performance', 'analysis', 'recap', 'news', 'depth-chart'
        ]
        
        url_indicators = sum(1 for indicator in sports_indicators if indicator in url)
        score += url_indicators * 8
        
        title_indicators = sum(1 for indicator in sports_indicators if indicator in title)
        score += title_indicators * 5
        
        # 6. AUTHORITATIVE SPORTS SOURCES
        authoritative_sites = {
            'espn.com': 20,
            'nba.com': 20,
            'nfl.com': 20,
            'mlb.com': 20,
            'nhl.com': 20,
            'basketball-reference.com': 25,
            'sports-reference.com': 25,
            'statmuse.com': 18,
            'ncaa.com': 15,
            'bleacherreport.com': 12,
            'teamrankings.com': 15,
            'flashscore.com': 15,
            'sofascore.com': 15,
        }
        
        for site, bonus in authoritative_sites.items():
            if site in parsed.netloc and depth >= 2:  # Must have subpages
                score += bonus
                break
        
        # 7. URL STRUCTURE QUALITY
        # Reward clean URL structures
        if any(ext in url for ext in ['.html', '.php', '.aspx', '.jsp']):
            score += 3
        
        # Penalize URLs with excessive parameters (often search/filter pages)
        if parsed.query and len(parsed.query) > 50:
            score -= 5
        
        # 8. CONTENT TYPE INDICATORS
        # Detect if URL suggests actual data pages
        data_patterns = [
            r'/stats?/',
            r'/roster',
            r'/schedule',
            r'/standings',
            r'/team/\w+',
            r'/player/\w+',
            r'\d{4}-\d{2}',  # Year-season pattern
            r'/season/\d{4}',
        ]
        
        for pattern in data_patterns:
            if re.search(pattern, url):
                score += 6
        
        # 9. PENALIZE GENERIC/AGGREGATOR PAGES
        generic_patterns = ['/news', '/blog', '/articles', '/videos', '/photos']
        if any(pattern in path and depth <= 2 for pattern in generic_patterns):
            score -= 8
        
        # 10. TITLE QUALITY
        # Penalize overly generic titles
        generic_title_words = ['home', 'homepage', 'official site', 'welcome', 'main page']
        if any(word in title for word in generic_title_words):
            score -= 15
        
        return score
    
    def filter_and_rank_results(self, results: List[Dict[str, str]], 
                                query: str,
                                min_score: float = 15) -> List[Dict[str, str]]:
        """
        Advanced filtering and ranking
        """
        # Score all results
        scored_results = []
        for result in results:
            score = self.calculate_relevance_score(result, query)
            result['relevance_score'] = score
            scored_results.append(result)
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Filter by minimum score
        filtered = [r for r in scored_results if r['relevance_score'] >= min_score]
        
        # Smart deduplication: Keep best result per domain
        domain_best = {}
        for result in filtered:
            domain = urlparse(result['url']).netloc
            if domain not in domain_best or result['relevance_score'] > domain_best[domain]['relevance_score']:
                domain_best[domain] = result
        
        # Sort again and return
        final_results = list(domain_best.values())
        final_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return final_results
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Main search interface
        """
        results = self.search_with_ddgs(query)
        
        if not results:
            print("⚠️  No results found.")
            return []
        
        filtered_results = self.filter_and_rank_results(results, query)
        return filtered_results
    
    def save_results(self, results: List[Dict[str, str]], 
                    query: str, 
                    filename: str = 'results.json'):
        """
        Save results to JSON file
        """
        data = {
            'query': query,
            'total_results': len(results),
            'results': results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {filename}")


def print_results(results: List[Dict[str, str]], max_display: int = 10):
    """
    Pretty print results
    """
    if not results:
        print("\n❌ No relevant results found.")
        return
    
    print(f"\n✅ Found {len(results)} highly relevant links:\n")
    print("=" * 100)
    
    for i, result in enumerate(results[:max_display], 1):
        print(f"\n{i}. {result['title']}")
        print(f"   🔗 {result['url']}")
        print(f"   📊 Relevance Score: {result['relevance_score']:.1f}")
        if result['snippet']:
            snippet = result['snippet'][:200] + "..." if len(result['snippet']) > 200 else result['snippet']
            print(f"   📝 {snippet}")
        print("-" * 100)


def main():
    """
    Example usage with multiple queries
    """
    scraper = AdvancedLinkScraper(max_results=25)
    
    # Example queries
    queries = [
        "Get the statistics for the team LA Lakers from competition NBA, sport Basketball",
        "Manchester United Premier League squad roster 2024 2025",
        "Real Madrid Champions League standings 2024",
        "Golden State Warriors player stats NBA 2024"
    ]
    
    for query in queries[:1]:  # Process first query as demo
        print("=" * 100)
        print("ADVANCED SMART LINK SCRAPER")
        print("=" * 100)
        print(f"\n📋 Query: {query}\n")
        
        results = scraper.search(query)
        print_results(results, max_display=10)
        
        # Save to files
        if results:
            scraper.save_results(results, query, 'search_results.json')
            
            # Also save as text
            with open('search_results.txt', 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Found {len(results)} relevant links\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"{i}. {result['title']}\n")
                    f.write(f"   URL: {result['url']}\n")
                    f.write(f"   Relevance Score: {result['relevance_score']:.1f}\n")
                    f.write(f"   Snippet: {result['snippet'][:150]}...\n\n")
            
            print("💾 Results also saved to search_results.txt")
        
        print("\n")


if __name__ == "__main__":
    main()
