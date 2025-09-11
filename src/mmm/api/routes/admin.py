"""
Admin endpoints for database initialization and maintenance.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import structlog

from mmm.database.connection import db_manager, Base

router = APIRouter()
logger = structlog.get_logger()


@router.post("/init-database")
async def initialize_database() -> Dict[str, Any]:
    """
    Initialize database tables.
    
    This endpoint creates all required database tables for the MMM application.
    Should only be called once during initial deployment.
    """
    try:
        if not db_manager.engine:
            await db_manager.initialize()
        
        async with db_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables initialized successfully")
        
        return {
            "status": "success",
            "message": "Database tables created successfully",
            "tables_created": [
                "upload_sessions",
                "training_runs", 
                "model_results",
                "optimization_runs"
            ]
        }
        
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Database initialization failed: {str(e)}"
        )


@router.get("/database-status")
async def get_database_status() -> Dict[str, Any]:
    """Check database connection and table status."""
    try:
        if not db_manager.engine:
            await db_manager.initialize()
            
        # Test connection
        async with db_manager.engine.connect() as conn:
            result = await conn.execute("SELECT version()")
            db_version = result.scalar()
            
        return {
            "status": "connected",
            "database_version": db_version,
            "engine_status": "initialized"
        }
        
    except Exception as e:
        logger.error("Database status check failed", error=str(e))
        return {
            "status": "error",
            "error": str(e)
        }
