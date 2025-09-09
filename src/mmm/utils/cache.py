"""
Caching utilities for Redis integration.
"""
import json
import hashlib
from typing import Any, Optional, Dict, List
import structlog
import redis.asyncio as redis

from mmm.config.settings import settings

logger = structlog.get_logger()


class CacheManager:
    """Manages caching operations with Redis fallback to in-memory."""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache: Dict[str, Any] = {}
        
    async def initialize(self, redis_client: Optional[redis.Redis] = None):
        """Initialize cache manager with Redis client."""
        self.redis_client = redis_client
        if self.redis_client:
            logger.info("Cache manager initialized with Redis")
        else:
            logger.warning("Cache manager using in-memory fallback")
    
    def _generate_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate a consistent cache key."""
        # Sort kwargs for consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        key_data = f"{prefix}:{':'.join(f'{k}={v}' for k, v in sorted_kwargs)}"
        
        # Hash long keys to avoid Redis key length limits
        if len(key_data) > 250:
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            return f"{prefix}:hash:{key_hash}"
        
        return key_data
    
    async def get(self, cache_key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.redis_client:
                cached_value = await self.redis_client.get(cache_key)
                if cached_value:
                    return json.loads(cached_value)
            else:
                # Fallback to memory cache
                return self.memory_cache.get(cache_key)
        except Exception as e:
            logger.warning("Cache get failed", key=cache_key, error=str(e))
        
        return None
    
    async def set(self, cache_key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        try:
            ttl = ttl or settings.database.cache_ttl
            serialized_value = json.dumps(value, default=str)
            
            if self.redis_client:
                await self.redis_client.setex(cache_key, ttl, serialized_value)
            else:
                # Fallback to memory cache (without TTL)
                self.memory_cache[cache_key] = value
            
            return True
        except Exception as e:
            logger.warning("Cache set failed", key=cache_key, error=str(e))
            return False
    
    async def delete(self, cache_key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.redis_client:
                await self.redis_client.delete(cache_key)
            else:
                self.memory_cache.pop(cache_key, None)
            
            return True
        except Exception as e:
            logger.warning("Cache delete failed", key=cache_key, error=str(e))
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    deleted = await self.redis_client.delete(*keys)
                    return deleted
            else:
                # Memory cache pattern matching
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern.replace('*', '') in k]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                return len(keys_to_delete)
        except Exception as e:
            logger.warning("Cache clear pattern failed", pattern=pattern, error=str(e))
        
        return 0
    
    # Specialized caching methods for MMM
    
    async def cache_training_progress(self, run_id: str, progress_data: Dict[str, Any]) -> bool:
        """Cache training progress data."""
        cache_key = f"training_progress:{run_id}"
        return await self.set(cache_key, progress_data, ttl=300)  # 5 minutes
    
    async def get_training_progress(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get cached training progress."""
        cache_key = f"training_progress:{run_id}"
        return await self.get(cache_key)
    
    async def cache_response_curve(self, run_id: str, channel: str, 
                                 curve_data: Dict[str, Any]) -> bool:
        """Cache response curve data."""
        cache_key = self._generate_cache_key(
            "response_curve", 
            run_id=run_id, 
            channel=channel
        )
        ttl = settings.database.response_curve_cache_ttl
        return await self.set(cache_key, curve_data, ttl=ttl)
    
    async def get_response_curve(self, run_id: str, channel: str) -> Optional[Dict[str, Any]]:
        """Get cached response curve."""
        cache_key = self._generate_cache_key(
            "response_curve", 
            run_id=run_id, 
            channel=channel
        )
        return await self.get(cache_key)
    
    async def cache_optimization_result(self, run_id: str, budget: float,
                                      constraints: List[Dict], result: Dict[str, Any]) -> bool:
        """Cache optimization results."""
        # Create a hash of constraints for cache key
        constraints_hash = hashlib.md5(json.dumps(constraints, sort_keys=True).encode()).hexdigest()[:8]
        
        cache_key = self._generate_cache_key(
            "optimization_result",
            run_id=run_id,
            budget=int(budget),
            constraints_hash=constraints_hash
        )
        
        return await self.set(cache_key, result, ttl=7200)  # 2 hours
    
    async def get_optimization_result(self, run_id: str, budget: float,
                                    constraints: List[Dict]) -> Optional[Dict[str, Any]]:
        """Get cached optimization result."""
        constraints_hash = hashlib.md5(json.dumps(constraints, sort_keys=True).encode()).hexdigest()[:8]
        
        cache_key = self._generate_cache_key(
            "optimization_result",
            run_id=run_id,
            budget=int(budget),
            constraints_hash=constraints_hash
        )
        
        return await self.get(cache_key)
    
    async def cache_model_results(self, run_id: str, results: Dict[str, Any]) -> bool:
        """Cache complete model results."""
        cache_key = f"model_results:{run_id}"
        return await self.set(cache_key, results, ttl=86400)  # 24 hours
    
    async def get_model_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get cached model results."""
        cache_key = f"model_results:{run_id}"
        return await self.get(cache_key)
    
    async def invalidate_run_cache(self, run_id: str) -> int:
        """Invalidate all cache entries for a training run."""
        patterns = [
            f"training_progress:{run_id}",
            f"response_curve:*run_id={run_id}*",
            f"optimization_result:*run_id={run_id}*",
            f"model_results:{run_id}"
        ]
        
        total_deleted = 0
        for pattern in patterns:
            deleted = await self.clear_pattern(pattern)
            total_deleted += deleted
        
        logger.info("Cache invalidated for run", run_id=run_id, deleted_keys=total_deleted)
        return total_deleted


# Global cache manager instance
cache_manager = CacheManager()