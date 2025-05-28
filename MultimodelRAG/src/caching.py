"""
Caching utilities for the MultimodalRAG system.
Provides caching for expensive operations like embeddings and LLM responses.
"""

import os
import json
import hashlib
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union, Tuple
from functools import wraps
from datetime import datetime, timedelta
import pickle

# Configure logging
logger = logging.getLogger('multimodal_rag.caching')

# Type variable for function return type
T = TypeVar('T')

class Cache:
    """Base cache class"""
    
    def __init__(self, name: str, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the cache.
        
        Args:
            name: Cache name
            max_size: Maximum number of items in cache
            ttl_seconds: Time-to-live in seconds
        """
        self.name = name
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached item or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Check if item has expired
                timestamp, value = self.cache[key]
                if time.time() - timestamp <= self.ttl_seconds:
                    # Update access time
                    self.access_times[key] = time.time()
                    self.hits += 1
                    return value
                else:
                    # Remove expired item
                    del self.cache[key]
                    del self.access_times[key]
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Evict items if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict()
            
            # Add item to cache
            self.cache[key] = (time.time(), value)
            self.access_times[key] = time.time()
    
    def _evict(self) -> None:
        """Evict least recently used items from cache"""
        if not self.access_times:
            return
        
        # Find least recently used item
        oldest_key = min(self.access_times, key=self.access_times.get)
        
        # Remove item
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self) -> None:
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "name": self.name,
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }

class MemoryCache(Cache):
    """In-memory cache implementation"""
    pass

class DiskCache(Cache):
    """Disk-based cache implementation"""
    
    def __init__(self, name: str, cache_dir: str = "cache", max_size: int = 1000, 
                ttl_seconds: int = 3600):
        """
        Initialize the disk cache.
        
        Args:
            name: Cache name
            cache_dir: Directory to store cache files
            max_size: Maximum number of items in cache
            ttl_seconds: Time-to-live in seconds
        """
        super().__init__(name, max_size, ttl_seconds)
        
        self.cache_dir = os.path.join(cache_dir, name)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load cache metadata
        self.metadata_path = os.path.join(self.cache_dir, "metadata.json")
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.cache = {k: (v[0], None) for k, v in metadata.get("cache", {}).items()}
                    self.access_times = metadata.get("access_times", {})
                    self.hits = metadata.get("hits", 0)
                    self.misses = metadata.get("misses", 0)
            except Exception as e:
                logger.error(f"Error loading cache metadata: {str(e)}")
                self.cache = {}
                self.access_times = {}
                self.hits = 0
                self.misses = 0
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_path, 'w') as f:
                metadata = {
                    "cache": {k: [v[0], None] for k, v in self.cache.items()},
                    "access_times": self.access_times,
                    "hits": self.hits,
                    "misses": self.misses
                }
                json.dump(metadata, f)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
    
    def _get_cache_path(self, key: str) -> str:
        """
        Get path to cache file for key.
        
        Args:
            key: Cache key
        
        Returns:
            Path to cache file
        """
        # Use hash of key as filename to avoid invalid characters
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.pkl")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from disk cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached item or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Check if item has expired
                timestamp, _ = self.cache[key]
                if time.time() - timestamp <= self.ttl_seconds:
                    # Load value from disk
                    try:
                        cache_path = self._get_cache_path(key)
                        if os.path.exists(cache_path):
                            with open(cache_path, 'rb') as f:
                                value = pickle.load(f)
                            
                            # Update access time
                            self.access_times[key] = time.time()
                            self.hits += 1
                            return value
                    except Exception as e:
                        logger.error(f"Error loading cached item: {str(e)}")
                
                # Remove expired or invalid item
                self._remove_item(key)
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set item in disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Evict items if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict()
            
            # Save value to disk
            try:
                cache_path = self._get_cache_path(key)
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Update metadata
                self.cache[key] = (time.time(), None)
                self.access_times[key] = time.time()
                self._save_metadata()
            except Exception as e:
                logger.error(f"Error caching item: {str(e)}")
    
    def _remove_item(self, key: str) -> None:
        """
        Remove item from disk cache.
        
        Args:
            key: Cache key
        """
        try:
            # Remove cache file
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            
            # Update metadata
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            
            self._save_metadata()
        except Exception as e:
            logger.error(f"Error removing cached item: {str(e)}")
    
    def clear(self) -> None:
        """Clear the disk cache"""
        with self.lock:
            # Remove all cache files
            for key in list(self.cache.keys()):
                self._remove_item(key)
            
            # Reset metadata
            self.cache.clear()
            self.access_times.clear()
            self._save_metadata()

class CacheManager:
    """Manages multiple caches"""
    
    def __init__(self):
        """Initialize the cache manager"""
        self.caches = {}
    
    def get_cache(self, name: str, cache_type: str = "memory", **kwargs) -> Cache:
        """
        Get or create a cache.
        
        Args:
            name: Cache name
            cache_type: Type of cache ("memory" or "disk")
            **kwargs: Additional cache parameters
        
        Returns:
            Cache instance
        """
        if name not in self.caches:
            if cache_type == "disk":
                self.caches[name] = DiskCache(name, **kwargs)
            else:
                self.caches[name] = MemoryCache(name, **kwargs)
        
        return self.caches[name]
    
    def clear_all(self) -> None:
        """Clear all caches"""
        for cache in self.caches.values():
            cache.clear()
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.
        
        Returns:
            Dictionary with cache statistics
        """
        return {name: cache.get_stats() for name, cache in self.caches.items()}

# Cache decorators

def cache_result(cache_name: str = "default", key_fn: Optional[Callable] = None, 
                ttl_seconds: int = 3600, cache_type: str = "memory"):
    """
    Decorator to cache function results.
    
    Args:
        cache_name: Name of cache to use
        key_fn: Function to generate cache key from arguments
        ttl_seconds: Time-to-live in seconds
        cache_type: Type of cache ("memory" or "disk")
    
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get cache
            cache_manager = CacheManager()
            cache = cache_manager.get_cache(
                cache_name, 
                cache_type=cache_type, 
                ttl_seconds=ttl_seconds
            )
            
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Call function and