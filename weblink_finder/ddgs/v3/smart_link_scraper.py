#!/usr/bin/env python3
"""
Smart Link Scraper - Finds highly relevant, specific subpage links for queries
Uses free tools and intelligent filtering to avoid generic homepage results
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, quote_plus
import re
from typing import List, Dict, Tuple
import time


class SmartLinkScraper:
    def __init__(self, max_results: int = 15):
        self.max_results = max_results
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """Search using DuckDuckGo HTML scraping (free, no API key needed)"""
        print(f"🔍 Searching DuckDuckGo for: {query}")
        
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        try:
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find all result divs
            for result in soup.find_all('div', class_='result'):
                title_tag = result.find('a', class_='result__a')
                snippet_tag = result.find('a', class_='result__snippet')
                
                if title_tag:
                    url = title_tag.get('href', '')
                    title = title_tag.get_text(strip=True)
                    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                    
                    if url and title:
                        results.append({
                            'url': url,
                            'title': title,
                            'snippet': snippet
                        })
                        
                if len(results) >= self.max_results:
                    break
                    
            return results
            
        except Exception as e:
            print(f"❌ DuckDuckGo search error: {e}")
            return []
    
    def search_google_scrape(self, query: str) -> List[Dict[str, str]]:
        """Backup: Scrape Google search results (use sparingly)"""
        print(f"🔍 Searching Google for: {query}")
        
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num=20"
        
        try:
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find search result divs
            for g in soup.find_all('div', class_='g'):
                anchor = g.find('a')
                title_tag = g.find('h3')
                snippet_tag = g.find('div', class_=['VwiC3b', 'yXK7lf'])
                
                if anchor and title_tag:
                    url = anchor.get('href', '')
                    title = title_tag.get_text(strip=True)
                    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                    
                    if url.startswith('http'):
                        results.append({
                            'url': url,
                            'title': title,
                            'snippet': snippet
                        })
                        
                if len(results) >= self.max_results:
                    break
                    
            return results
            
        except Exception as e:
            print(f"❌ Google search error: {e}")
            return []
    
    def calculate_relevance_score(self, result: Dict[str, str], query: str) -> float:
        """
        Calculate relevance score based on multiple factors
        Higher score = more specific and relevant
        """
        url = result['url'].lower()
        title = result['title'].lower()
        snippet = result['snippet'].lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Extract query keywords
        query_words = set(re.findall(r'\w+', query_lower))
        query_words.discard('the')
        query_words.discard('get')
        query_words.discard('from')
        
        # 1. URL depth score (deeper = more specific)
        path_parts = urlparse(url).path.strip('/').split('/')
        depth = len([p for p in path_parts if p])
        score += depth * 3  # 3 points per level
        
        # 2. Penalize generic/root URLs heavily
        if url.endswith('/') and depth <= 1:
            score -= 20
        
        # 3. Keyword presence in URL (very important)
        url_keyword_matches = sum(1 for word in query_words if word in url)
        score += url_keyword_matches * 10
        
        # 4. Keyword presence in title
        title_keyword_matches = sum(1 for word in query_words if word in title)
        score += title_keyword_matches * 8
        
        # 5. Keyword presence in snippet
        snippet_keyword_matches = sum(1 for word in query_words if word in snippet)
        score += snippet_keyword_matches * 3
        
        # 6. Reward URLs with specific indicators (stats, roster, schedule, etc.)
        specific_indicators = ['stats', 'statistics', 'roster', 'schedule', 'standings', 
                              'team', 'player', 'season', 'results', 'scores', 'news',
                              'analysis', 'recap', 'game', 'match']
        indicator_matches = sum(1 for indicator in specific_indicators if indicator in url)
        score += indicator_matches * 5
        
        # 7. Penalize very generic domains without subpages
        parsed = urlparse(url)
        if parsed.path in ['', '/'] and not parsed.query:
            score -= 30
        
        # 8. Reward authoritative sports sites with specific pages
        sports_sites = ['espn.com', 'nba.com', 'basketball-reference.com', 
                       'sports-reference.com', 'ncaa.com', 'bleacherreport.com',
                       'statmuse.com', 'teamrankings.com']
        if any(site in url for site in sports_sites) and depth >= 2:
            score += 15
        
        # 9. Check for file extensions that indicate specific content
        if any(ext in url for ext in ['.html', '.php', '.aspx']):
            score += 3
            
        # 10. Check title for exact query phrase matches
        if query_lower in title:
            score += 12
            
        return score
    
    def filter_and_rank_results(self, results: List[Dict[str, str]], 
                                query: str) -> List[Dict[str, str]]:
        """Filter out generic links and rank by relevance"""
        
        # Calculate scores for all results
        scored_results = []
        for result in results:
            score = self.calculate_relevance_score(result, query)
            result['relevance_score'] = score
            scored_results.append(result)
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Filter out low-scoring results (likely generic pages)
        min_score_threshold = 10
        filtered_results = [r for r in scored_results if r['relevance_score'] >= min_score_threshold]
        
        # Remove duplicate domains (keep only the highest scoring from each)
        seen_domains = {}
        deduplicated = []
        
        for result in filtered_results:
            domain = urlparse(result['url']).netloc
            if domain not in seen_domains or result['relevance_score'] > seen_domains[domain]['relevance_score']:
                if domain in seen_domains:
                    # Replace with higher scoring one
                    deduplicated = [r for r in deduplicated if urlparse(r['url']).netloc != domain]
                seen_domains[domain] = result
                deduplicated.append(result)
        
        return deduplicated
    
    def search(self, query: str, method: str = 'duckduckgo') -> List[Dict[str, str]]:
        """
        Main search method
        
        Args:
            query: Search query
            method: 'duckduckgo' or 'google'
        
        Returns:
            List of relevant links with metadata
        """
        # Get raw search results
        if method == 'duckduckgo':
            results = self.search_duckduckgo(query)
        else:
            results = self.search_google_scrape(query)
            
        if not results:
            print("⚠️  No results found. Trying alternate search method...")
            # Try the other method
            if method == 'duckduckgo':
                results = self.search_google_scrape(query)
            else:
                results = self.search_duckduckgo(query)
        
        # Filter and rank
        filtered_results = self.filter_and_rank_results(results, query)
        
        return filtered_results


def print_results(results: List[Dict[str, str]], max_display: int = 10):
    """Pretty print the results"""
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
            snippet = result['snippet'][:150] + "..." if len(result['snippet']) > 150 else result['snippet']
            print(f"   📝 {snippet}")
        print("-" * 100)


def main():
    """Example usage"""
    scraper = SmartLinkScraper(max_results=20)
    
    # Example queries
    queries = [
        "Get the statistics for the team LA Lakers from competition NBA, sport Basketball",
        "Manchester United Premier League squad roster 2024",
        "Real Madrid Champions League match results"
    ]
    
    # Use the first query as demonstration
    query = queries[0]
    
    print("=" * 100)
    print("SMART LINK SCRAPER - Finding Specific, Relevant Links")
    print("=" * 100)
    print(f"\nQuery: {query}\n")
    
    results = scraper.search(query, method='duckduckgo')
    print_results(results, max_display=10)
    
    # Save results to file
    if results:
        print("\n💾 Saving results to 'search_results.txt'...")
        with open('search_results.txt', 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Found {len(results)} relevant links\n\n")
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {result['title']}\n")
                f.write(f"   URL: {result['url']}\n")
                f.write(f"   Score: {result['relevance_score']:.1f}\n\n")
        print("✅ Results saved!")


if __name__ == "__main__":
    main()
