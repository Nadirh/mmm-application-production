"""
Database connection management for SQLite and Redis.
"""
import asyncio
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
import redis.asyncio as redis
import structlog

from mmm.config.settings import settings

logger = structlog.get_logger()


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        self.engine = None
        self.async_session_maker = None
        self.redis_pool = None
        
    async def initialize(self):
        """Initialize database connections."""
        await self._setup_database()
        await self._setup_redis()
        
    async def _setup_database(self):
        """Setup database async engine."""
        database_url = settings.database.url
        
        # Convert database URLs for async support
        if database_url.startswith("sqlite://"):
            database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
        elif database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        
        self.engine = create_async_engine(
            database_url,
            echo=settings.database.echo,
            pool_pre_ping=True,
            # Add production optimizations
            pool_size=20,
            max_overflow=30,
            pool_timeout=30,
            pool_recycle=3600,
            # Add query timeout protection
            connect_args={
                "server_settings": {
                    "statement_timeout": "30s",
                    "lock_timeout": "10s"
                }
            } if "postgresql" in database_url else {}
        )
        
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("Database initialized", url=database_url)
        
    async def _setup_redis(self):
        """Setup Redis connection pool."""
        try:
            redis_url = getattr(settings.database, 'redis_url', 'redis://localhost:6379/0')
            
            # Check if this is a cluster configuration (rediss:// with cluster endpoint)
            if 'clustercfg.' in redis_url or redis_url.startswith('rediss://'):
                from redis.asyncio.cluster import RedisCluster
                import ssl
                
                # Extract host and port from the URL
                if redis_url.startswith('rediss://'):
                    # Remove rediss:// prefix and extract host:port
                    host_port = redis_url.replace('rediss://', '').split('/')[0]
                    if ':' in host_port:
                        host, port = host_port.split(':')
                        port = int(port)
                    else:
                        host = host_port
                        port = 6379
                else:
                    # Fallback parsing
                    host, port = redis_url.split('/')[-1].split(':')
                    port = int(port)
                
                # Setup SSL context for TLS
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                self.redis_pool = RedisCluster(
                    host=host,
                    port=port,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=10,
                    ssl=ssl_context,
                    health_check_interval=30
                )
                
                # Test cluster connection with timeout
                await asyncio.wait_for(self.redis_pool.ping(), timeout=10.0)
                logger.info("Redis cluster connection established", url=redis_url)
                
            else:
                # Standard Redis connection (non-cluster)
                self.redis_pool = redis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=20,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    health_check_interval=30
                )
                
                # Test connection with timeout
                await asyncio.wait_for(self.redis_pool.ping(), timeout=5.0)
                logger.info("Redis connection established", url=redis_url)
            
        except Exception as e:
            logger.warning("Redis connection failed, using in-memory fallback", error=str(e))
            self.redis_pool = None
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session."""
        if not self.async_session_maker:
            await self.initialize()
            
        async with self.async_session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def get_redis(self) -> Optional[redis.Redis]:
        """Get Redis connection."""
        if not self.redis_pool:
            await self._setup_redis()
        return self.redis_pool
    
    async def close(self):
        """Close all database connections."""
        if self.engine:
            await self.engine.dispose()
            
        if self.redis_pool:
            await self.redis_pool.aclose()
        
        logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session."""
    async for session in db_manager.get_session():
        yield session


async def get_redis() -> Optional[redis.Redis]:
    """Dependency for getting Redis connection."""
    return await db_manager.get_redis()