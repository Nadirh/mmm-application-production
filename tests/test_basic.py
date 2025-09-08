"""
Basic tests to verify the application structure.
"""
import pytest
from fastapi.testclient import TestClient


def test_app_import():
    """Test that the app can be imported."""
    from mmm.api.main import app
    assert app is not None


def test_health_endpoint():
    """Test the health check endpoint."""
    from mmm.api.main import app
    client = TestClient(app)
    
    response = client.get("/api/health/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_root_endpoint():
    """Test the root endpoint."""
    from mmm.api.main import app
    client = TestClient(app)
    
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_docs_endpoint():
    """Test that docs are available in development."""
    from mmm.api.main import app
    client = TestClient(app)
    
    response = client.get("/docs")
    # Should either return docs or redirect, not 404
    assert response.status_code in [200, 307, 308]