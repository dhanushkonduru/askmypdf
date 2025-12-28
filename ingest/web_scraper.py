"""
Web scraping module for crawling and extracting content from websites.
Supports static sites (BeautifulSoup + requests) and JavaScript-heavy sites (Playwright).
"""
import asyncio
import time
import re
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
import requests
from bs4 import BeautifulSoup
import trafilatura

# Optional Playwright import (fallback for JS-heavy sites)
try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Rate limiting: minimum delay between requests (seconds)
MIN_REQUEST_DELAY = 1.0
REQUEST_TIMEOUT = 30
MAX_PAGES = 50  # Maximum number of pages to crawl (prevents infinite loops)


class WebScraper:
    """Web scraper with robots.txt checking, rate limiting, and content extraction."""
    
    def __init__(self, max_depth: int = 3, same_domain_only: bool = True, max_pages: int = MAX_PAGES):
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.robots_cache: Dict[str, Optional[RobotFileParser]] = {}
        self.last_request_time: Dict[str, float] = {}
        self.playwright_browser = None
        self.playwright_instance = None
        
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL to prevent duplicates (remove fragments, query params, trailing slashes)."""
        parsed = urlparse(url)
        # Normalize path: remove trailing slash (except for root)
        path = parsed.path.rstrip('/') or '/'
        # Reconstruct URL without fragment and query
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        return normalized.lower()  # Case-insensitive comparison
    
    def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        domain = self._get_domain(url)
        
        # Check cache first
        if domain not in self.robots_cache:
            try:
                robots_url = urljoin(domain, '/robots.txt')
                rp = RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                self.robots_cache[domain] = rp
            except Exception as e:
                # If robots.txt is inaccessible, allow by default
                print(f"⚠️ Could not fetch robots.txt for {domain}: {e}")
                self.robots_cache[domain] = None
        
        rp = self.robots_cache[domain]
        if rp is None:
            return True  # Allow if robots.txt unavailable
        
        return rp.can_fetch('*', url)
    
    def _rate_limit(self, url: str):
        """Enforce rate limiting (minimum delay between requests)."""
        domain = self._get_domain(url)
        last_time = self.last_request_time.get(domain, 0)
        elapsed = time.time() - last_time
        
        if elapsed < MIN_REQUEST_DELAY:
            time.sleep(MIN_REQUEST_DELAY - elapsed)
        
        self.last_request_time[domain] = time.time()
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract same-domain links from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        base_domain = self._get_domain(base_url)
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(base_url, href)
            
            # Normalize URL to prevent duplicates
            normalized_url = self._normalize_url(absolute_url)
            
            # Only include same-domain links if enabled
            if self.same_domain_only:
                if self._get_domain(normalized_url) == base_domain:
                    links.append(normalized_url)
            else:
                links.append(normalized_url)
        
        # Deduplicate and return
        return list(set(links))
    
    def _extract_text_static(self, url: str) -> Optional[Dict[str, str]]:
        """Extract text from static HTML using requests + trafilatura."""
        try:
            self._rate_limit(url)
            response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            # Use trafilatura for clean text extraction
            text = trafilatura.extract(response.text, url=url)
            
            if not text or len(text.strip()) < 100:  # Skip very short pages
                return None
            
            # Get title from BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else url
            
            return {
                'url': url,
                'title': title,
                'text': text,
                'html': response.text
            }
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching {url}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error extracting text from {url}: {e}")
            return None
    
    async def _extract_text_playwright(self, url: str) -> Optional[Dict[str, str]]:
        """Extract text from JavaScript-heavy pages using Playwright."""
        if not PLAYWRIGHT_AVAILABLE:
            return None
        
        try:
            if self.playwright_browser is None:
                self.playwright_instance = await async_playwright().start()
                self.playwright_browser = await self.playwright_instance.chromium.launch(headless=True)
            
            page = await self.playwright_browser.new_page()
            await page.goto(url, wait_until='networkidle', timeout=REQUEST_TIMEOUT * 1000)
            
            # Get page content
            html = await page.content()
            title = await page.title()
            
            # Extract text using trafilatura
            text = trafilatura.extract(html, url=url)
            
            await page.close()
            
            if not text or len(text.strip()) < 100:
                return None
            
            return {
                'url': url,
                'title': title,
                'text': text,
                'html': html
            }
        except PlaywrightTimeoutError:
            print(f"⏱️ Timeout fetching {url} with Playwright")
            return None
        except Exception as e:
            print(f"❌ Playwright error for {url}: {e}")
            return None
    
    def _extract_text(self, url: str, use_playwright: bool = False) -> Optional[Dict[str, str]]:
        """Extract text from URL, with Playwright fallback if needed."""
        # Try static extraction first
        result = self._extract_text_static(url)
        
        # If static extraction fails or returns minimal content, try Playwright
        if (result is None or len(result.get('text', '')) < 200) and use_playwright and PLAYWRIGHT_AVAILABLE:
            print(f"🔄 Trying Playwright for {url}...")
            try:
                # Create new event loop for this thread
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(self._extract_text_playwright(url))
            except Exception as e:
                print(f"⚠️ Playwright extraction failed: {e}")
        
        return result
    
    def crawl(self, start_urls: List[str], progress_callback=None) -> List[Dict[str, str]]:
        """
        Crawl websites starting from given URLs.
        
        Args:
            start_urls: List of starting URLs
            progress_callback: Optional callback function(url, status, depth) for progress updates
        
        Returns:
            List of extracted page data with keys: url, title, text, crawl_date, domain
        """
        self.visited_urls.clear()
        results = []
        # Normalize starting URLs
        normalized_start_urls = [self._normalize_url(url) for url in start_urls]
        to_visit = [(url, 0) for url in normalized_start_urls]  # (url, depth)
        pages_crawled = 0
        
        while to_visit and pages_crawled < self.max_pages:
            url, depth = to_visit.pop(0)
            
            # Normalize URL before checking
            normalized_url = self._normalize_url(url)
            
            # Skip if already visited, exceeds max depth, or max pages reached
            if normalized_url in self.visited_urls or depth > self.max_depth:
                continue
            
            # Check robots.txt
            if not self._check_robots_txt(normalized_url):
                if progress_callback:
                    progress_callback(normalized_url, "blocked_by_robots", depth)
                continue
            
            # Mark as visited BEFORE crawling to prevent re-queuing
            self.visited_urls.add(normalized_url)
            pages_crawled += 1
            
            if progress_callback:
                progress_callback(normalized_url, "crawling", depth)
            
            # Extract content (use original URL for fetching, but store normalized)
            result = self._extract_text(normalized_url, use_playwright=True)
            
            if result:
                # Update result with normalized URL
                result['url'] = normalized_url
                # Add metadata
                result['crawl_date'] = datetime.now().isoformat()
                result['domain'] = self._get_domain(normalized_url)
                result['source_type'] = 'website'
                
                results.append(result)
                
                if progress_callback:
                    progress_callback(normalized_url, "success", depth)
                
                # Extract links for next depth level (only if under max pages and depth)
                if depth < self.max_depth and pages_crawled < self.max_pages:
                    links = self._extract_links(result.get('html', ''), normalized_url)
                    for link in links:
                        # Normalize link before checking
                        normalized_link = self._normalize_url(link)
                        if normalized_link not in self.visited_urls:
                            to_visit.append((normalized_link, depth + 1))
            else:
                if progress_callback:
                    progress_callback(normalized_url, "failed", depth)
        
        # Warn if max pages reached
        if pages_crawled >= self.max_pages:
            print(f"⚠️ Reached maximum page limit ({self.max_pages}). Stopping crawl.")
        
        # Cleanup Playwright if used
        if self.playwright_browser or self.playwright_instance:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def cleanup():
                if self.playwright_browser:
                    await self.playwright_browser.close()
                if self.playwright_instance:
                    await self.playwright_instance.stop()
            
            try:
                loop.run_until_complete(cleanup())
            except Exception as e:
                print(f"⚠️ Error closing Playwright: {e}")
            finally:
                self.playwright_browser = None
                self.playwright_instance = None
        
        return results


def crawl_urls(urls: List[str], max_depth: int = 3, max_pages: int = MAX_PAGES, progress_callback=None) -> List[Dict[str, str]]:
    """
    Convenience function to crawl multiple URLs.
    
    Args:
        urls: List of URLs to crawl
        max_depth: Maximum crawl depth (default: 3)
        max_pages: Maximum number of pages to crawl (default: 50, prevents infinite loops)
        progress_callback: Optional callback for progress updates
    
    Returns:
        List of extracted page data
    """
    scraper = WebScraper(max_depth=max_depth, same_domain_only=True, max_pages=max_pages)
    return scraper.crawl(urls, progress_callback=progress_callback)

