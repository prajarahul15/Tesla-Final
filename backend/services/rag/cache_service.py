"""
Cache Service for RAG
Caches retrieval results and embeddings to improve performance
"""

import logging
import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis not available, using in-memory cache")

logger = logging.getLogger(__name__)


class CacheService:
    """Service for caching RAG retrieval results"""
    
    def __init__(self, use_redis: bool = False, redis_host: str = "localhost", redis_port: int = 6379):
        """
        Initialize cache service
        
        Args:
            use_redis: Whether to use Redis (if available)
            redis_host: Redis host
            redis_port: Redis port
        """
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.cache_ttl = 3600  # 1 hour default
        
        if self.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                self.redis_client.ping()  # Test connection
                logger.info("✅ Redis cache initialized")
            except Exception as e:
                logger.warning(f"⚠️ Redis not available, using in-memory cache: {e}")
                self.use_redis = False
                self.memory_cache = {}
        else:
            self.memory_cache = {}
            logger.info("📦 In-memory cache initialized")
    
    def _generate_cache_key(self, prefix: str, key_data: Any) -> str:
        """
        Generate cache key from data
        
        Args:
            prefix: Key prefix
            key_data: Data to hash
            
        Returns:
            Cache key string
        """
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"rag:{prefix}:{key_hash}"
    
    def get(self, prefix: str, key_data: Any) -> Optional[Any]:
        """
        Get cached value
        
        Args:
            prefix: Key prefix
            key_data: Data to generate key from
            
        Returns:
            Cached value or None
        """
        cache_key = self._generate_cache_key(prefix, key_data)
        
        try:
            if self.use_redis:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            else:
                if cache_key in self.memory_cache:
                    entry = self.memory_cache[cache_key]
                    # Check expiration
                    if datetime.now() < entry['expires_at']:
                        return entry['value']
                    else:
                        # Expired, remove it
                        del self.memory_cache[cache_key]
        
        except Exception as e:
            logger.warning(f"⚠️ Cache get error: {e}")
        
        return None
    
    def set(self, prefix: str, key_data: Any, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Cache a value
        
        Args:
            prefix: Key prefix
            key_data: Data to generate key from
            value: Value to cache
            ttl: Time to live in seconds (default: self.cache_ttl)
            
        Returns:
            True if successful
        """
        cache_key = self._generate_cache_key(prefix, key_data)
        ttl = ttl or self.cache_ttl
        
        try:
            if self.use_redis:
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(value)
                )
            else:
                self.memory_cache[cache_key] = {
                    'value': value,
                    'expires_at': datetime.now() + timedelta(seconds=ttl)
                }
            
            logger.debug(f"💾 Cached: {prefix}")
            return True
        
        except Exception as e:
            logger.warning(f"⚠️ Cache set error: {e}")
            return False
    
    def invalidate(self, prefix: str, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries
        
        Args:
            prefix: Key prefix
            pattern: Optional pattern to match
            
        Returns:
            Number of keys invalidated
        """
        try:
            if self.use_redis:
                # Find keys matching pattern
                keys = list(self.redis_client.scan_iter(match=f"rag:{prefix}:*"))
                if pattern:
                    keys = [k for k in keys if pattern in k]
                
                if keys:
                    self.redis_client.delete(*keys)
                    return len(keys)
            else:
                # In-memory cache invalidation
                keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(f"rag:{prefix}:")]
                if pattern:
                    keys_to_delete = [k for k in keys_to_delete if pattern in k]
                
                for key in keys_to_delete:
                    del self.memory_cache[key]
                
                return len(keys_to_delete)
        
        except Exception as e:
            logger.warning(f"⚠️ Cache invalidation error: {e}")
        
        return 0
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Cache statistics dictionary
        """
        try:
            if self.use_redis:
                info = self.redis_client.info('memory')
                return {
                    "type": "redis",
                    "used_memory": info.get('used_memory_human', 'N/A'),
                    "keys": self.redis_client.dbsize()
                }
            else:
                # Clean expired entries
                now = datetime.now()
                expired = [k for k, v in self.memory_cache.items() if v['expires_at'] < now]
                for key in expired:
                    del self.memory_cache[key]
                
                return {
                    "type": "memory",
                    "entries": len(self.memory_cache),
                    "expired_cleaned": len(expired)
                }
        
        except Exception as e:
            logger.warning(f"⚠️ Cache stats error: {e}")
            return {"error": str(e)}


