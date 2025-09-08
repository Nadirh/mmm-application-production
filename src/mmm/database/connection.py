"""
Database connection management for SQLite and Redis.
"""
import asyncio
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
import aioredis
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
        await self._setup_sqlite()
        await self._setup_redis()
        
    async def _setup_sqlite(self):
        """Setup SQLite async engine."""
        database_url = settings.database.url
        
        # Convert sqlite:// to sqlite+aiosqlite:// for async support
        if database_url.startswith("sqlite://"):
            database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
        
        self.engine = create_async_engine(
            database_url,
            echo=settings.database.echo,
            pool_pre_ping=True,
        )
        
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("SQLite database initialized", url=database_url)
        
    async def _setup_redis(self):
        """Setup Redis connection pool."""
        try:
            redis_url = getattr(settings.database, 'redis_url', 'redis://localhost:6379/0')
            self.redis_pool = await aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            # Test connection
            await self.redis_pool.ping()
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
    
    async def get_redis(self) -> Optional[aioredis.Redis]:
        """Get Redis connection."""
        if not self.redis_pool:
            await self._setup_redis()
        return self.redis_pool
    
    async def close(self):
        """Close all database connections."""
        if self.engine:
            await self.engine.dispose()
            
        if self.redis_pool:
            await self.redis_pool.close()
        
        logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session."""
    async for session in db_manager.get_session():
        yield session


async def get_redis() -> Optional[aioredis.Redis]:
    """Dependency for getting Redis connection."""
    return await db_manager.get_redis()