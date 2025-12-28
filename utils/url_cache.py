"""
URL caching system to avoid re-crawling the same URLs.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

class URLCache:
    """Cache for crawled URL content to avoid redundant requests."""
    
    def __init__(self, cache_dir: str = '.cache', ttl_hours: int = 24):
        """
        Initialize URL cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours (default: 24)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for URL."""
        key = self._get_cache_key(url)
        return self.cache_dir / f"{key}.json"
    
    def is_cached(self, url: str) -> bool:
        """Check if URL is cached and not stale."""
        cache_path = self._get_cache_path(url)
        
        if not cache_path.exists():
            return False
        
        try:
            # Check if cache is stale
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time > self.ttl:
                return False
            
            return True
        except Exception:
            return False
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached data for URL."""
        if not self.is_cached(url):
            return None
        
        try:
            cache_path = self._get_cache_path(url)
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def set(self, url: str, data: Dict[str, Any]):
        """Cache data for URL."""
        cache_path = self._get_cache_path(url)
        
        cache_entry = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_entry, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to cache URL {url}: {e}")
    
    def invalidate(self, url: str):
        """Remove cache for URL."""
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            try:
                cache_path.unlink()
            except Exception as e:
                print(f"⚠️ Failed to invalidate cache for {url}: {e}")
    
    def clear_all(self):
        """Clear entire cache."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        except Exception as e:
            print(f"⚠️ Failed to clear cache: {e}")

