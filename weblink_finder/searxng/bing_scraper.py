#!/usr/bin/env python3
"""
Smart Link Scraper using Bing Search
Free HTML scraping, no API key needed
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus, unquote
import re
from typing import List, Dict
import json
import time


class BingScraper:
    """
    Scrapes Bing search results
    Bing is generally more lenient than Google for scraping
    """
    
    def __init__(self, max_results: int = 20):
        self.max_results = max_results
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
    def search_bing(self, query: str) -> List[Dict[str, str]]:
        """
        Search Bing and extract results
        
        Args:
            query: Search query
        
        Returns:
            List of search results
        """
        print(f"🔍 Searching Bing for: {query}")
        
        search_url = f"https://www.bing.com/search?q={quote_plus(query)}&count=50"
        
        try:
            response = requests.get(search_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find organic search results
            # Bing uses li.b_algo for organic results
            for result in soup.find_all('li', class_='b_algo'):
                # Extract title and URL
                h2_tag = result.find('h2')
                if not h2_tag:
                    continue
                
                a_tag = h2_tag.find('a')
                if not a_tag:
                    continue
                
                url = a_tag.get('href', '')
                title = a_tag.get_text(strip=True)
                
                # Extract snippet
                snippet = ''
                p_tag = result.find('p')
                if p_tag:
                    snippet = p_tag.get_text(strip=True)
                else:
                    # Try div with class b_caption
                    caption = result.find('div', class_='b_caption')
                    if caption:
                        p_tag = caption.find('p')
                        if p_tag:
                            snippet = p_tag.get_text(strip=True)
                
                if url and title and url.startswith('http'):
                    results.append({
                        'url': url,
                        'title': title,
                        'snippet': snippet
                    })
                
                if len(results) >= self.max_results:
                    break
            
            print(f"✅ Found {len(results)} results from Bing")
            return results
            
        except Exception as e:
            print(f"❌ Bing search error: {e}")
            return []
    
    def calculate_relevance_score(self, result: Dict[str, str], query: str) -> float:
        """Calculate relevance score for a result"""
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
        
        # 1. URL Depth
        score += depth * 4
        
        # Penalize root URLs
        if path in ['', '/'] and not parsed.query:
            score -= 50
        
        # 2. Keyword matching in URL
        url_matches = sum(1 for word in query_words if word in url)
        score += url_matches * 15
        
        path_matches = sum(1 for word in query_words if word in path)
        score += path_matches * 10
        
        # 3. Keyword matching in title
        title_matches = sum(1 for word in query_words if word in title)
        score += title_matches * 12
        
        # Exact phrase match
        if query_lower in title:
            score += 20
        
        # 4. Keyword matching in snippet
        snippet_matches = sum(1 for word in query_words if word in snippet)
        score += snippet_matches * 4
        
        # 5. Content indicators
        sports_indicators = [
            'stats', 'statistics', 'roster', 'schedule', 'standings', 'team',
            'player', 'season', 'results', 'scores', 'game', 'match',
            'league', 'championship', 'playoff', 'record', 'ranking',
            'performance', 'analysis', 'recap', 'depth-chart'
        ]
        
        url_indicators = sum(1 for indicator in sports_indicators if indicator in url)
        score += url_indicators * 8
        
        title_indicators = sum(1 for indicator in sports_indicators if indicator in title)
        score += title_indicators * 5
        
        # 6. Authoritative sports sites
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
            if site in parsed.netloc and depth >= 2:
                score += bonus
                break
        
        # 7. URL structure quality
        if any(ext in url for ext in ['.html', '.php', '.aspx']):
            score += 3
        
        # 8. Data patterns
        data_patterns = [
            r'/stats?/',
            r'/roster',
            r'/schedule',
            r'/standings',
            r'/team/\w+',
            r'/player/\w+',
            r'\d{4}-\d{2}',
            r'/season/\d{4}',
        ]
        
        for pattern in data_patterns:
            if re.search(pattern, url):
                score += 6
        
        return score
    
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
        """Main search method with rate limiting"""
        # Add small delay to be respectful
        time.sleep(1)
        
        results = self.search_bing(query)
        
        if not results:
            print("⚠️  No results found")
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
    scraper = BingScraper(max_results=25)
    
    queries = [
        "Get the statistics for the team LA Lakers from competition NBA, sport Basketball",
        "Golden State Warriors player stats NBA 2024",
        "Manchester United Premier League roster 2024"
    ]
    
    query = queries[0]
    
    print("=" * 100)
    print("SMART LINK SCRAPER - Using Bing Search")
    print("=" * 100)
    print(f"\n📋 Query: {query}\n")
    
    results = scraper.search(query)
    print_results(results, max_display=10)
    
    if results:
        # Save results
        with open('bing_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'query': query,
                'total_results': len(results),
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        with open('bing_results.txt', 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Total Results: {len(results)}\n\n")
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {result['title']}\n")
                f.write(f"   {result['url']}\n")
                f.write(f"   Score: {result['relevance_score']:.1f}\n\n")
        
        print("\n💾 Results saved to bing_results.json and bing_results.txt")


if __name__ == "__main__":
    main()
