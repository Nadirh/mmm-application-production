"""
Main FastAPI application for MMM.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import structlog
import traceback
import time
import json
import numpy as np
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager

from mmm.config.settings import settings
from mmm.api.routes import data, model, optimization, health, websocket, admin


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting MMM API server", environment=settings.env.value)
    
    # Create necessary directories
    settings.setup_directories()
    
    # Initialize database connections
    from mmm.database.connection import db_manager
    from mmm.utils.cache import cache_manager
    
    await db_manager.initialize()
    
    # Initialize cache with Redis
    redis_client = await db_manager.get_redis()
    await cache_manager.initialize(redis_client)
    
    logger.info("MMM API server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MMM API server")
    
    try:
        await cache_manager.close()
        await db_manager.close()
        logger.info("MMM API server shutdown completed")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="Media Mix Modeling API",
    description="API for Media Mix Modeling application to optimize marketing budget allocation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development() else None,
    redoc_url="/redoc" if settings.is_development() else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=settings.api.cors_methods,
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info(
        "HTTP request processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        exc_info=exc,
        method=request.method,
        url=str(request.url),
        traceback=traceback.format_exc()
    )
    
    if settings.is_development():
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "traceback": traceback.format_exc()
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred"
            }
        )



# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(model.router, prefix="/api/model", tags=["model"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["optimization"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend application."""
    try:
        with open("static/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>MMM Dashboard</title></head>
        <body>
            <h1>Media Mix Modeling API</h1>
            <p>Version: 1.0.0</p>
            <p>Environment: production</p>
            <p>Frontend not found. Please check static files.</p>
        </body>
        </html>
        """)

@app.get("/api/info")
async def api_info():
    """API information endpoint."""
    return {
        "message": "Media Mix Modeling API",
        "version": "1.0.0",
        "environment": settings.env.value
    }


if __name__ == "__main__":
    import uvicorn
    import time
    
    uvicorn.run(
        "mmm.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.logging.level.lower()
    )