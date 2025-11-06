"""
Cache service for LLM responses and intermediate results.

Implements multi-level caching:
- Level 1: In-memory cache (LRU, fast)
- Level 2: Disk cache (persistent, slower)
"""

import logging
from pathlib import Path
from typing import Any, Optional, Dict
from collections import OrderedDict
import diskcache
import hashlib
import json
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MultiLevelCache:
    """
    Multi-level cache with in-memory and disk storage.
    
    Strategy:
    1. Check memory cache first (fast, ~0ms)
    2. If miss, check disk cache (slower, ~1-10ms)
    3. If hit, promote to memory cache
    4. Write new entries to both levels
    """
    
    def __init__(
        self,
        cache_dir: Path,
        memory_size: int = 100,  # Max entries in memory
        disk_size_gb: int = 2,
        memory_ttl_hours: int = 24  # TTL for memory entries
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Level 1: In-memory cache (LRU)
        self.memory_cache: OrderedDict = OrderedDict()
        self.memory_size = memory_size
        self.memory_ttl = timedelta(hours=memory_ttl_hours)
        self.memory_lock = threading.RLock()
        
        # Cache statistics counters
        self._total_requests = 0
        self._memory_hits = 0
        self._disk_hits = 0
        
        # Level 2: Disk cache (persistent)
        size_limit_bytes = disk_size_gb * 1024 * 1024 * 1024
        self.disk_cache = diskcache.Cache(str(self.cache_dir), size_limit=size_limit_bytes)
        
        logger.info(
            f"Multi-level cache initialized: "
            f"Memory={memory_size} entries, Disk={disk_size_gb}GB"
        )
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (memory first, then disk).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Increment total requests counter (within lock for thread safety)
        with self.memory_lock:
            self._total_requests += 1
        
        # Level 1: Check memory cache first (fast)
        with self.memory_lock:
            if key in self.memory_cache:
                value, timestamp = self.memory_cache[key]
                
                # Check TTL
                if datetime.now() - timestamp < self.memory_ttl:
                    # Move to end (most recently used)
                    self.memory_cache.move_to_end(key)
                    logger.debug(f"Memory cache HIT for key: {key[:16]}...")
                    self._memory_hits += 1  # Increment memory hit counter
                    return value
                else:
                    # Expired, remove from memory
                    del self.memory_cache[key]
                    logger.debug(f"Memory cache entry expired for key: {key[:16]}...")
        
        # Level 2: Check disk cache
        disk_value = self.disk_cache.get(key)
        if disk_value is not None:
            # Promote to memory cache
            with self.memory_lock:
                self._add_to_memory(key, disk_value)
                self._disk_hits += 1  # Increment disk hit counter
            logger.debug(f"Disk cache HIT (promoted to memory) for key: {key[:16]}...")
            return disk_value
        
        logger.debug(f"Cache MISS for key: {key[:16]}...")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in both cache levels.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Level 2: Write to disk (persistent)
        try:
            self.disk_cache[key] = value
        except Exception as e:
            logger.warning(f"Error writing to disk cache: {e}")
        
        # Level 1: Write to memory (fast)
        with self.memory_lock:
            self._add_to_memory(key, value)
    
    def _add_to_memory(self, key: str, value: Any) -> None:
        """Add entry to memory cache with LRU eviction."""
        with self.memory_lock:
            # Remove if exists (will add at end)
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            # Add at end (most recently used)
            self.memory_cache[key] = (value, datetime.now())
            
            # Evict oldest if over limit
            if len(self.memory_cache) > self.memory_size:
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]
                logger.debug(f"Memory cache evicted oldest entry: {oldest_key[:16]}...")
    
    def delete(self, key: str) -> None:
        """Delete key from both cache levels."""
        with self.memory_lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
        
        if key in self.disk_cache:
            del self.disk_cache[key]
    
    def clear(self) -> None:
        """Clear both cache levels."""
        with self.memory_lock:
            self.memory_cache.clear()
        self.disk_cache.clear()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (dictionary-like interface)."""
        # Check memory first
        with self.memory_lock:
            if key in self.memory_cache:
                value, timestamp = self.memory_cache[key]
                if datetime.now() - timestamp < self.memory_ttl:
                    return True
        
        # Check disk
        return key in self.disk_cache
    
    def __getitem__(self, key: str) -> Any:
        """Get value by key (dictionary-like interface)."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set value by key (dictionary-like interface)."""
        self.set(key, value)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.memory_lock:
            memory_size = len(self.memory_cache)
            total_requests = self._total_requests
            memory_hits = self._memory_hits
            disk_hits = self._disk_hits
        
        disk_size = len(self.disk_cache)
        
        return {
            'memory_entries': memory_size,
            'memory_max': self.memory_size,
            'disk_entries': disk_size,
            'total_requests': total_requests,
            'memory_hits': memory_hits,
            'disk_hits': disk_hits,
            'memory_hit_rate': memory_hits / max(total_requests, 1),
            'disk_hit_rate': disk_hits / max(total_requests, 1)
        }


class CacheService:
    """Service for managing multi-level caches."""
    
    def __init__(self, cache_dir: Path, size_limit_gb: int = 2):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Use multi-level cache
        self.cache = MultiLevelCache(
            cache_dir=cache_dir,
            memory_size=100,  # Keep 100 most recent entries in memory
            disk_size_gb=size_limit_gb,
            memory_ttl_hours=24
        )
        
        logger.info(f"Cache service initialized at: {self.cache_dir} (multi-level)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self.cache.set(key, value)
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self.cache.delete(key)
    
    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()

