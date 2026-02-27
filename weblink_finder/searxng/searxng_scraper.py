#!/usr/bin/env python3
"""
Smart Link Scraper using SearxNG - Free Meta-Search Engine
No API key needed, aggregates results from multiple search engines
"""

import requests
from urllib.parse import urlparse, quote_plus
import re
from typing import List, Dict
import json
import random


class SearxNGScraper:
    """
    Uses SearxNG public instances to search
    SearxNG is a free, open-source meta-search engine
    """
    
    # Public SearxNG instances (regularly updated list)
    SEARXNG_INSTANCES = [
        "https://searx.be",
        "https://search.sapti.me",
        "https://searx.work",
        "https://search.bus-hit.me",
        "https://searx.tiekoetter.com",
        "https://search.incogniweb.net",
        "https://searx.prvcy.eu",
        "https://search.im-in.space",
        "https://searx.be",
        "https://baresearch.org",
    ]
    
    def __init__(self, max_results: int = 20):
        self.max_results = max_results
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def search_searxng(self, query: str, instance_url: str = None) -> List[Dict[str, str]]:
        """
        Search using SearxNG instance
        
        Args:
            query: Search query
            instance_url: Specific instance to use (optional)
        
        Returns:
            List of search results
        """
        if instance_url is None:
            instance_url = random.choice(self.SEARXNG_INSTANCES)
        
        print(f"🔍 Searching via SearxNG ({instance_url}): {query}")
        
        # SearxNG JSON API endpoint
        search_url = f"{instance_url}/search"
        
        params = {
            'q': query,
            'format': 'json',
            'categories': 'general',
            'language': 'en',
            'pageno': 1
        }
        
        try:
            response = requests.get(
                search_url,
                params=params,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for result in data.get('results', [])[:self.max_results]:
                    results.append({
                        'url': result.get('url', ''),
                        'title': result.get('title', ''),
                        'snippet': result.get('content', '')
                    })
                
                print(f"✅ Found {len(results)} results from SearxNG")
                return results
            else:
                print(f"⚠️  Status code: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ SearxNG search error: {e}")
            return []
    
    def search_with_fallback(self, query: str) -> List[Dict[str, str]]:
        """
        Try multiple SearxNG instances until one works
        """
        instances = self.SEARXNG_INSTANCES.copy()
        random.shuffle(instances)
        
        for instance in instances[:3]:  # Try up to 3 instances
            results = self.search_searxng(query, instance)
            if results:
                return results
            print(f"⚠️  Instance {instance} failed, trying another...")
        
        print("❌ All SearxNG instances failed")
        return []
    
    def calculate_relevance_score(self, result: Dict[str, str], query: str) -> float:
        """Calculate relevance score"""
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
        
        # Scoring
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
            'sofascore.com': 15, 'flashscore.com': 15,
        }
        
        for site, bonus in authoritative_sites.items():
            if site in parsed.netloc and depth >= 2:
                score += bonus
                break
        
        return score
    
    def filter_and_rank(self, results: List[Dict[str, str]], 
                       query: str,
                       min_score: float = 15) -> List[Dict[str, str]]:
        """Filter and rank results"""
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
        results = self.search_with_fallback(query)
        
        if not results:
            return []
        
        filtered_results = self.filter_and_rank(results, query)
        return filtered_results


def print_results(results: List[Dict[str, str]], max_display: int = 10):
    """Pretty print results"""
    if not results:
        print("\n❌ No relevant results found.")
        return
    
    print(f"\n✅ Found {len(results)} highly relevant links:\n")
    print("=" * 100)
    
    for i, result in enumerate(results[:max_display], 1):
        print(f"\n{i}. {result['title']}")
        print(f"   🔗 {result['url']}")
        print(f"   📊 Relevance Score: {result['relevance_score']:.1f}")
        if result.get('snippet'):
            snippet = result['snippet'][:200] + "..." if len(result['snippet']) > 200 else result['snippet']
            print(f"   📝 {snippet}")
        print("-" * 100)


def main():
    """Example usage"""
    scraper = SearxNGScraper(max_results=25)
    
    queries = [
        "Get the statistics for the team LA Lakers from competition NBA, sport Basketball",
        "Manchester United Premier League squad roster 2024",
        "Golden State Warriors player stats NBA 2024"
    ]
    
    query = queries[0]
    
    print("=" * 100)
    print("SMART LINK SCRAPER - Using SearxNG Meta-Search Engine")
    print("=" * 100)
    print(f"\n📋 Query: {query}\n")
    
    results = scraper.search(query)
    print_results(results, max_display=10)
    
    if results:
        # Save results
        with open('searxng_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'query': query,
                'total_results': len(results),
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print("\n💾 Results saved to searxng_results.json")


if __name__ == "__main__":
    main()
