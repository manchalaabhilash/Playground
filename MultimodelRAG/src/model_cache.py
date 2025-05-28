import os
import time
import logging
import threading
from typing import Dict, Any, Tuple, Optional, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelCache:
    """
    Cache for loaded models to avoid reloading
    """
    
    def __init__(self, max_size: int = 3, ttl_seconds: int = 3600):
        """
        Initialize the model cache
        
        Args:
            max_size: Maximum number of models to cache
            ttl_seconds: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}  # model_key -> (model, timestamp)
        self.lock = threading.RLock()
        self.cleanup_thread = None
        self.running = False
    
    def _generate_key(self, model_path: str, model_type: str, **kwargs) -> str:
        """
        Generate a cache key
        
        Args:
            model_path: Path to the model
            model_type: Type of model
            **kwargs: Additional parameters
        
        Returns:
            Cache key
        """
        # Sort kwargs to ensure consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        
        # Create key components
        key_components = [model_path, model_type]
        
        # Add kwargs
        for k, v in sorted_kwargs:
            key_components.append(f"{k}={v}")
        
        # Join components
        return "|".join(key_components)
    
    def get(self, model_path: str, model_type: str, **kwargs) -> Optional[Any]:
        """
        Get a model from the cache
        
        Args:
            model_path: Path to the model
            model_type: Type of model
            **kwargs: Additional parameters
        
        Returns:
            Cached model or None if not found
        """
        with self.lock:
            # Generate key
            key = self._generate_key(model_path, model_type, **kwargs)
            
            # Check if model is in cache
            if key in self.cache:
                model, timestamp = self.cache[key]
                
                # Check if model is still valid
                if time.time() - timestamp <= self.ttl_seconds:
                    logger.info(f"Cache hit for {key}")
                    
                    # Update timestamp
                    self.cache[key] = (model, time.time())
                    
                    return model
                else:
                    # Model expired
                    logger.info(f"Cache expired for {key}")
                    del self.cache[key]
            
            # Model not found or expired
            return None
    
    def put(self, model_path: str, model_type: str, model: Any, **kwargs) -> None:
        """
        Put a model in the cache
        
        Args:
            model_path: Path to the model
            model_type: Type of model
            model: Model to cache
            **kwargs: Additional parameters
        """
        with self.lock:
            # Generate key
            key = self._generate_key(model_path, model_type, **kwargs)
            
            # Check if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest model
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                logger.info(f"Cache full, removing {oldest_key}")
                del self.cache[oldest_key]
            
            # Add model to cache
            self.cache[key] = (model, time.time())
            logger.info(f"Added model to cache: {key}")
    
    def remove(self, model_path: str, model_type: str, **kwargs) -> None:
        """
        Remove a model from the cache
        
        Args:
            model_path: Path to the model
            model_type: Type of model
            **kwargs: Additional parameters
        """
        with self.lock:
            # Generate key
            key = self._generate_key(model_path, model_type, **kwargs)
            
            # Remove model from cache
            if key in self.cache:
                del self.cache[key]
                logger.info(f"Removed model from cache: {key}")
    
    def clear(self) -> None:
        """Clear the cache"""
        with self.lock:
            self.cache = {}
            logger.info("Cleared model cache")
    
    def start_cleanup_thread(self, interval_seconds: int = 300) -> None:
        """
        Start a thread to clean up expired models
        
        Args:
            interval_seconds: Cleanup interval in seconds
        """
        if self.cleanup_thread is not None and self.cleanup_thread.is_alive():
            logger.warning("Cleanup thread already running")
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, args=(interval_seconds,), daemon=True)
        self.cleanup_thread.start()
        logger.info("Started model cache cleanup thread")
    
    def stop_cleanup_thread(self) -> None:
        """Stop the cleanup thread"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            return
        
        self.running = False
        self.cleanup_thread.join(timeout=5.0)
        self.cleanup_thread = None
        logger.info("Stopped model cache cleanup thread")
    
    def _cleanup_loop(self, interval_seconds: int) -> None:
        """
        Cleanup loop
        
        Args:
            interval_seconds: Cleanup interval in seconds
        """
        while self.running:
            try:
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cleanup thread: {str(e)}")
            
            # Sleep until next cleanup
            time.sleep(interval_seconds)
    
    def _cleanup_expired(self) -> None:
        """Clean up expired models"""
        with self.lock:
            # Get current time
            current_time = time.time()
            
            # Find expired models
            expired_keys = []
            for key, (_, timestamp) in self.cache.items():
                if current_time - timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            # Remove expired models
            for key in expired_keys:
                del self.cache[key]
                logger.info(f"Removed expired model from cache: {key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Cache statistics
        """
        with self.lock:
            stats = {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "models": []
            }
            
            # Add model info
            for key, (_, timestamp) in self.cache.items():
                age_seconds = time.time() - timestamp
                expires_in = max(0, self.ttl_seconds - age_seconds)
                
                stats["models"].append({
                    "key": key,
                    "age_seconds": age_seconds,
                    "expires_in_seconds": expires_in
                })
            
            return stats

# Create a global instance with environment variable configuration
model_cache = ModelCache(
    max_size=int(os.environ.get("MODEL_CACHE_SIZE", "3")),
    ttl_seconds=int(os.environ.get("MODEL_CACHE_TTL", "3600"))
)

# Auto-start cleanup thread if enabled
if os.environ.get("MODEL_CACHE_CLEANUP_ENABLED", "true").lower() in ("true", "1", "yes"):
    model_cache.start_cleanup_thread(
        interval_seconds=int(os.environ.get("MODEL_CACHE_CLEANUP_INTERVAL", "300"))
    )
