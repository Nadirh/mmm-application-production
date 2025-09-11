import asyncio
import os
import sys
sys.path.insert(0, 'src')

from sqlalchemy.ext.asyncio import create_async_engine
from mmm.database.connection import Base
from mmm.database.models import UploadSession

async def create_tables():
    """Create all database tables."""
    database_url = "postgresql+asyncpg://mmm_admin:6UDU4M4:IUSp-)N6@mmm-mmm-demo-production.cl4y84wusrzb.us-east-2.rds.amazonaws.com:5432/mmm_mmm_demo"
    
    engine = create_async_engine(database_url)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    await engine.dispose()
    print("Database tables created successfully!")

if __name__ == "__main__":
    asyncio.run(create_tables())
