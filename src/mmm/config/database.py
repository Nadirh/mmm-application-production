"""
Database configuration for PostgreSQL with multi-client support
"""
import os
import asyncio
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData, event
from sqlalchemy.pool import NullPool
import structlog

logger = structlog.get_logger()

class Base(DeclarativeBase):
    """Base class for all database models"""
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s"
        }
    )

class DatabaseManager:
    """Manages database connections for multi-client deployment"""
    
    def __init__(self):
        self.engines: dict[str, AsyncEngine] = {}
        self.session_makers: dict[str, async_sessionmaker[AsyncSession]] = {}
        
    def get_database_url(self, client_id: Optional[str] = None) -> str:
        """Get database URL for specific client or default"""
        if client_id:
            # Multi-client production setup
            url = os.getenv(f"DATABASE_URL_{client_id.upper()}")
            if not url:
                # Fallback to parameterized URL
                base_url = os.getenv("DATABASE_URL")
                if base_url and "postgresql" in base_url:
                    # Replace database name with client-specific name
                    url = base_url.replace("/mmm", f"/mmm_{client_id.replace('-', '_')}")
                else:
                    url = base_url
        else:
            url = os.getenv("DATABASE_URL")
            
        if not url:
            raise ValueError("DATABASE_URL environment variable not set")
            
        # Convert psycopg2 URL to asyncpg for async support
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            
        return url
    
    def get_engine(self, client_id: Optional[str] = None) -> AsyncEngine:
        """Get or create database engine for client"""
        cache_key = client_id or "default"
        
        if cache_key not in self.engines:
            database_url = self.get_database_url(client_id)
            
            # Engine configuration
            engine_kwargs = {
                "echo": os.getenv("DATABASE_ECHO", "false").lower() == "true",
                "pool_size": int(os.getenv("DATABASE_POOL_SIZE", "10")),
                "max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
                "pool_timeout": int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
                "pool_recycle": int(os.getenv("DATABASE_POOL_RECYCLE", "3600")),
                "pool_pre_ping": True,
            }
            
            # For SQLite, use NullPool to avoid threading issues
            if "sqlite" in database_url:
                engine_kwargs["poolclass"] = NullPool
                # Remove PostgreSQL-specific options
                engine_kwargs.pop("pool_size", None)
                engine_kwargs.pop("max_overflow", None)
                engine_kwargs.pop("pool_timeout", None)
                engine_kwargs.pop("pool_recycle", None)
                engine_kwargs.pop("pool_pre_ping", None)
            
            self.engines[cache_key] = create_async_engine(database_url, **engine_kwargs)
            
            # Create session maker
            self.session_makers[cache_key] = async_sessionmaker(
                self.engines[cache_key],
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info(
                "Created database engine",
                client_id=client_id,
                database_type="postgresql" if "postgresql" in database_url else "sqlite"
            )
        
        return self.engines[cache_key]
    
    def get_session_maker(self, client_id: Optional[str] = None) -> async_sessionmaker[AsyncSession]:
        """Get session maker for client"""
        cache_key = client_id or "default"
        
        # Ensure engine exists
        self.get_engine(client_id)
        
        return self.session_makers[cache_key]
    
    async def create_database_if_not_exists(self, client_id: Optional[str] = None):
        """Create database if it doesn't exist (PostgreSQL only)"""
        database_url = self.get_database_url(client_id)
        
        if "postgresql" in database_url:
            try:
                from sqlalchemy.ext.asyncio import create_async_engine
                from sqlalchemy import text
                
                # Connect to postgres database to create client database
                admin_url = database_url.rsplit("/", 1)[0] + "/postgres"
                admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
                
                db_name = database_url.rsplit("/", 1)[1].split("?")[0]
                
                async with admin_engine.begin() as conn:
                    # Check if database exists
                    result = await conn.execute(
                        text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                        {"db_name": db_name}
                    )
                    
                    if not result.fetchone():
                        # Create database
                        await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                        logger.info("Created database", database=db_name, client_id=client_id)
                
                await admin_engine.dispose()
                
            except Exception as e:
                logger.warning(
                    "Could not create database",
                    database=db_name,
                    client_id=client_id,
                    error=str(e)
                )
    
    async def initialize_tables(self, client_id: Optional[str] = None):
        """Initialize database tables for client"""
        engine = self.get_engine(client_id)
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Initialized database tables", client_id=client_id)
    
    async def close_all(self):
        """Close all database connections"""
        for engine in self.engines.values():
            await engine.dispose()
        
        self.engines.clear()
        self.session_makers.clear()
        
        logger.info("Closed all database connections")

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions
def get_engine(client_id: Optional[str] = None) -> AsyncEngine:
    """Get database engine for client"""
    return db_manager.get_engine(client_id)

def get_session_maker(client_id: Optional[str] = None) -> async_sessionmaker[AsyncSession]:
    """Get session maker for client"""
    return db_manager.get_session_maker(client_id)

async def get_session(client_id: Optional[str] = None) -> AsyncSession:
    """Get database session for client"""
    session_maker = get_session_maker(client_id)
    return session_maker()

# Context manager for database sessions
class DatabaseSession:
    """Context manager for database sessions"""
    
    def __init__(self, client_id: Optional[str] = None):
        self.client_id = client_id
        self.session: Optional[AsyncSession] = None
    
    async def __aenter__(self) -> AsyncSession:
        self.session = await get_session(self.client_id)
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                await self.session.rollback()
            else:
                await self.session.commit()
            await self.session.close()

# Migration support
async def run_migrations(client_id: Optional[str] = None):
    """Run database migrations for client"""
    try:
        # Create database if it doesn't exist
        await db_manager.create_database_if_not_exists(client_id)
        
        # Initialize tables
        await db_manager.initialize_tables(client_id)
        
        logger.info("Completed database migrations", client_id=client_id)
        
    except Exception as e:
        logger.error(
            "Failed to run migrations",
            client_id=client_id,
            error=str(e)
        )
        raise