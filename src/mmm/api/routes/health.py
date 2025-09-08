"""
Health check endpoints.
"""
from fastapi import APIRouter
from typing import Dict, Any
import time
from datetime import datetime

from mmm.config.settings import settings

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.env.value,
        "version": "1.0.0"
    }


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with system information."""
    import psutil
    import sys
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.env.value,
        "version": "1.0.0",
        "system": {
            "python_version": sys.version,
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent
        },
        "configuration": {
            "training_window_days": settings.model.training_window_days,
            "test_window_days": settings.model.test_window_days,
            "optimization_days": settings.optimization.default_optimization_days
        }
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """Kubernetes readiness probe endpoint."""
    # Add any readiness checks here (database connectivity, etc.)
    return {"status": "ready"}


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}