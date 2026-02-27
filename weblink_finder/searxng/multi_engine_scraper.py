#!/usr/bin/env python3
"""
Multi-Engine Smart Link Scraper
Automatically tries multiple free search engines with fallback
No API keys required!

Supported engines:
1. SearxNG (meta-search, best option)
2. Bing (reliable fallback)
3. Brave Search (alternative)
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus
import re
from typing import List, Dict, Optional
import json
import random
import time


class MultiEngineScraper:
    """
    Smart scraper that tries multiple search engines with automatic fallback
    """
    
    # SearxNG public instances
    SEARXNG_INSTANCES = [
        "https://searx.be",
        "https://search.sapti.me",
        "https://searx.work",
        "https://search.bus-hit.me",
        "https://searx.tiekoetter.com",
        "https://search.incogniweb.net",
    ]
    
    def __init__(self, max_results: int = 20, preferred_engine: str = 'searxng'):
        """
        Initialize scraper
        
        Args:
            max_results: Maximum number of results to retrieve
            preferred_engine: 'searxng', 'bing', or 'brave'
        """
        self.max_results = max_results
        self.preferred_engine = preferred_engine
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
    
    def search_searxng(self, query: str) -> Optional[List[Dict[str, str]]]:
        """Search using SearxNG"""
        instances = self.SEARXNG_INSTANCES.copy()
        random.shuffle(instances)
        
        for instance in instances[:2]:  # Try 2 instances
            try:
                print(f"🔍 Trying SearxNG: {instance}")
                
                search_url = f"{instance}/search"
                params = {
                    'q': query,
                    'format': 'json',
                    'categories': 'general',
                    'language': 'en',
                }
                
                response = requests.get(
                    search_url,
                    params=params,
                    headers=self.headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for result in data.get('results', [])[:self.max_results]:
                        results.append({
                            'url': result.get('url', ''),
                            'title': result.get('title', ''),
                            'snippet': result.get('content', ''),
                            'engine': 'searxng'
                        })
                    
                    if results:
                        print(f"✅ SearxNG returned {len(results)} results")
                        return results
                        
            except Exception as e:
                print(f"⚠️  SearxNG instance failed: {e}")
                continue
        
        return None
    
    def search_bing(self, query: str) -> Optional[List[Dict[str, str]]]:
        """Search using Bing"""
        try:
            print(f"🔍 Trying Bing Search")
            
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}&count=50"
            
            response = requests.get(search_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for result in soup.find_all('li', class_='b_algo'):
                h2_tag = result.find('h2')
                if not h2_tag:
                    continue
                
                a_tag = h2_tag.find('a')
                if not a_tag:
                    continue
                
                url = a_tag.get('href', '')
                title = a_tag.get_text(strip=True)
                
                snippet = ''
                p_tag = result.find('p')
                if p_tag:
                    snippet = p_tag.get_text(strip=True)
                
                if url and title and url.startswith('http'):
                    results.append({
                        'url': url,
                        'title': title,
                        'snippet': snippet,
                        'engine': 'bing'
                    })
                
                if len(results) >= self.max_results:
                    break
            
            if results:
                print(f"✅ Bing returned {len(results)} results")
                return results
            
        except Exception as e:
            print(f"⚠️  Bing search failed: {e}")
        
        return None
    
    def search_brave(self, query: str) -> Optional[List[Dict[str, str]]]:
        """Search using Brave (HTML scraping)"""
        try:
            print(f"🔍 Trying Brave Search")
            
            search_url = f"https://search.brave.com/search?q={quote_plus(query)}"
            
            response = requests.get(search_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Brave uses different structure - look for data-type="web"
            for result in soup.find_all('div', attrs={'data-type': 'web'}):
                # Find title link
                title_div = result.find('div', class_='title')
                if not title_div:
                    continue
                
                a_tag = title_div.find('a')
                if not a_tag:
                    continue
                
                url = a_tag.get('href', '')
                title = a_tag.get_text(strip=True)
                
                # Find snippet
                snippet = ''
                snippet_div = result.find('div', class_='snippet')
                if snippet_div:
                    snippet = snippet_div.get_text(strip=True)
                
                if url and title and url.startswith('http'):
                    results.append({
                        'url': url,
                        'title': title,
                        'snippet': snippet,
                        'engine': 'brave'
                    })
                
                if len(results) >= self.max_results:
                    break
            
            if results:
                print(f"✅ Brave returned {len(results)} results")
                return results
                
        except Exception as e:
            print(f"⚠️  Brave search failed: {e}")
        
        return None
    
    def search_with_fallback(self, query: str) -> List[Dict[str, str]]:
        """
        Try multiple search engines in order until one succeeds
        """
        engines = {
            'searxng': self.search_searxng,
            'bing': self.search_bing,
            'brave': self.search_brave,
        }
        
        # Try preferred engine first
        if self.preferred_engine in engines:
            results = engines[self.preferred_engine](query)
            if results:
                return results
        
        # Try remaining engines
        for engine_name, engine_func in engines.items():
            if engine_name == self.preferred_engine:
                continue  # Already tried
            
            time.sleep(1)  # Be polite with rate limiting
            results = engine_func(query)
            if results:
                return results
        
        print("❌ All search engines failed")
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
        
        # Scoring algorithm
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
        
        # Deduplicate
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
        print(f"\n{'='*100}")
        print(f"🔍 Searching for: {query}")
        print(f"{'='*100}\n")
        
        results = self.search_with_fallback(query)
        
        if not results:
            return []
        
        print(f"\n📊 Filtering and ranking {len(results)} results...")
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
        engine = result.get('engine', 'unknown')
        print(f"\n{i}. {result['title']}")
        print(f"   🔗 {result['url']}")
        print(f"   📊 Score: {result['relevance_score']:.1f} | Engine: {engine}")
        if result.get('snippet'):
            snippet = result['snippet'][:200] + "..." if len(result['snippet']) > 200 else result['snippet']
            print(f"   📝 {snippet}")
        print("-" * 100)


def main():
    """Example usage"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MULTI-ENGINE SMART LINK SCRAPER                          ║
║                                                                              ║
║  Uses multiple free search engines with automatic fallback:                 ║
║  • SearxNG (meta-search engine)                                             ║
║  • Bing (Microsoft search)                                                  ║
║  • Brave Search                                                             ║
║                                                                              ║
║  No API keys required!                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create scraper (try SearxNG first, then Bing, then Brave)
    scraper = MultiEngineScraper(max_results=25, preferred_engine='searxng')
    
    # Example queries
    queries = [
        "Get the statistics for the team LA Lakers from competition NBA, sport Basketball",
        "Golden State Warriors player stats NBA 2024",
        "Manchester United Premier League roster 2024"
    ]
    
    for query in queries[:1]:  # Run first query as demo
        results = scraper.search(query)
        print_results(results, max_display=10)
        
        if results:
            # Save results
            output_file = 'multi_engine_results.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': query,
                    'total_results': len(results),
                    'engines_used': list(set(r.get('engine', 'unknown') for r in results)),
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()
