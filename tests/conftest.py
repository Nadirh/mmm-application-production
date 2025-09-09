"""
Pytest configuration and fixtures for MMM application tests.
"""
import pytest
import pytest_asyncio
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
import tempfile
import os
from typing import Dict, Any, AsyncGenerator

from mmm.api.main import app
from mmm.database.connection import Base, get_db
from mmm.database.models import UploadSession, TrainingRun, OptimizationRun
from mmm.config.settings import settings
from mmm.data.validator import DataSummary, BusinessTier
from mmm.data.processor import ChannelInfo, ChannelType


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_db():
    """Create a test database."""
    # Use in-memory SQLite for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    async_session_maker = async_sessionmaker(
        engine, expire_on_commit=False
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async def get_test_db():
        async with async_session_maker() as session:
            yield session
    
    app.dependency_overrides[get_db] = get_test_db
    
    yield async_session_maker
    
    await engine.dispose()
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_csv_data():
    """Generate sample CSV data for testing."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    np.random.seed(42)  # For reproducible tests
    
    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'profit': np.random.normal(10000, 2000, len(dates)).clip(min=1000),
        'search_brand': np.random.normal(5000, 1000, len(dates)).clip(min=0),
        'search_nonbrand': np.random.normal(8000, 1500, len(dates)).clip(min=0),
        'social_facebook': np.random.normal(3000, 800, len(dates)).clip(min=0),
        'display_programmatic': np.random.normal(4000, 1200, len(dates)).clip(min=0),
        'tv_video': np.random.normal(6000, 2000, len(dates)).clip(min=0),
    }
    
    # Add some seasonality and trends
    for i, date in enumerate(dates):
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
        trend_factor = 1 + 0.001 * i  # Small upward trend
        
        data['profit'][i] *= seasonal_factor * trend_factor
        for channel in ['search_brand', 'search_nonbrand', 'social_facebook', 
                       'display_programmatic', 'tv_video']:
            data[channel][i] *= seasonal_factor * trend_factor
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(sample_csv_data, tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "sample_data.csv"
    sample_csv_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def invalid_csv_data():
    """Generate invalid CSV data for testing edge cases."""
    return pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'profit': [1000, -500, 2000],  # Negative profit
        'search_brand': [100, 200, -150],  # Negative spend
        'social': [0, 0, 0],  # All zero spend
    })


@pytest.fixture
def invalid_csv_file(invalid_csv_data, tmp_path):
    """Create a temporary CSV file with invalid data."""
    csv_file = tmp_path / "invalid_data.csv"
    invalid_csv_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def minimal_csv_data():
    """Generate minimal valid CSV data."""
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')  # 6 months
    
    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'profit': np.random.normal(5000, 1000, len(dates)).clip(min=1000),
        'search': np.random.normal(2000, 500, len(dates)).clip(min=100),
        'social': np.random.normal(1500, 400, len(dates)).clip(min=100),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def minimal_csv_file(minimal_csv_data, tmp_path):
    """Create a temporary CSV file with minimal data."""
    csv_file = tmp_path / "minimal_data.csv"
    minimal_csv_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def mock_upload_session(sample_csv_data):
    """Create a mock upload session in the in-memory storage."""
    from mmm.api.routes.data import upload_sessions
    
    upload_id = "test-upload-123"
    
    # Create mock data summary
    data_summary = DataSummary(
        total_days=len(sample_csv_data),
        total_profit=float(sample_csv_data['profit'].sum()),
        total_annual_spend=float(sample_csv_data[['search_brand', 'search_nonbrand', 
                                      'social_facebook', 'display_programmatic', 
                                      'tv_video']].sum().sum()),
        channel_count=5,
        date_range=(pd.to_datetime(sample_csv_data['date'].iloc[0]),
                   pd.to_datetime(sample_csv_data['date'].iloc[-1])),
        business_tier=BusinessTier.ENTERPRISE,
        data_quality_score=95.0
    )
    
    # Create mock channel info
    channel_info = {
        "search_brand": ChannelInfo(
            name="search_brand",
            type=ChannelType.SEARCH_BRAND,
            total_spend=15000.0,
            spend_share=0.25,
            days_active=300
        )
    }
    
    # Add to in-memory storage
    upload_sessions[upload_id] = {
        "filename": "test_data.csv",
        "upload_time": datetime(2023, 1, 1),
        "file_path": "/tmp/test_data.csv",
        "data_summary": data_summary,
        "validation_errors": [],
        "channel_info": channel_info,
        "processed_df": sample_csv_data,
        "status": "validated"
    }
    
    # Create a mock object that has the attributes the tests expect
    class MockUploadSession:
        def __init__(self):
            self.id = upload_id
            self.filename = "test_data.csv"
            self.status = "validated"
    
    mock_session = MockUploadSession()
    
    yield mock_session
    
    # Clean up
    if upload_id in upload_sessions:
        del upload_sessions[upload_id]


@pytest_asyncio.fixture
async def uploaded_session(test_db, sample_csv_data):
    """Create a sample upload session in the database."""
    async with test_db() as session:
        upload_session = UploadSession(
            id="test-upload-123",
            filename="test_data.csv",
            file_path="/tmp/test_data.csv",
            total_days=int(len(sample_csv_data)),
            total_profit=float(sample_csv_data['profit'].sum()),
            total_annual_spend=float(sample_csv_data[['search_brand', 'search_nonbrand', 
                                              'social_facebook', 'display_programmatic', 
                                              'tv_video']].sum().sum()),
            channel_count=5,
            date_range_start=pd.to_datetime(sample_csv_data['date'].iloc[0]),
            date_range_end=pd.to_datetime(sample_csv_data['date'].iloc[-1]),
            business_tier="enterprise",
            data_quality_score=95.0,
            validation_errors=[],
            channel_info={
                "search_brand": {"type": "search_brand", "total_spend": 1000000},
                "search_nonbrand": {"type": "search_non_brand", "total_spend": 1500000},
                "social_facebook": {"type": "social", "total_spend": 800000},
                "display_programmatic": {"type": "display", "total_spend": 1200000},
                "tv_video": {"type": "tv_video", "total_spend": 1800000}
            },
            status="validated"
        )
        
        session.add(upload_session)
        await session.commit()
        await session.refresh(upload_session)
        
        return upload_session


@pytest_asyncio.fixture
async def completed_training_run(test_db, uploaded_session):
    """Create a completed training run in the database."""
    async with test_db() as session:
        training_run = TrainingRun(
            id="test-run-456",
            upload_session_id=uploaded_session.id,
            status="completed",
            completion_time=datetime.now(UTC),
            training_config={
                "training_window_days": 126,
                "test_window_days": 14,
                "n_bootstrap": 100
            },
            model_parameters={
                "alpha_baseline": 1000.0,
                "alpha_trend": 0.5,
                "channel_alphas": {
                    "search_brand": 0.8,
                    "search_nonbrand": 1.2,
                    "social_facebook": 0.6,
                    "display_programmatic": 0.9,
                    "tv_video": 1.1
                },
                "channel_betas": {
                    "search_brand": 0.6,
                    "search_nonbrand": 0.7,
                    "social_facebook": 0.5,
                    "display_programmatic": 0.6,
                    "tv_video": 0.4
                },
                "channel_rs": {
                    "search_brand": 0.1,
                    "search_nonbrand": 0.2,
                    "social_facebook": 0.4,
                    "display_programmatic": 0.3,
                    "tv_video": 0.6
                }
            },
            model_performance={
                "cv_mape": 18.5,
                "r_squared": 0.82,
                "mape": 16.2
            },
            diagnostics={
                "media_attribution_percentage": 65.0,
                "channel_attributions": {
                    "search_brand": 800000,
                    "search_nonbrand": 1200000,
                    "social_facebook": 600000,
                    "display_programmatic": 900000,
                    "tv_video": 1100000
                }
            },
            confidence_intervals={
                "search_brand": [720000, 880000],
                "search_nonbrand": [1080000, 1320000],
                "social_facebook": [540000, 660000],
                "display_programmatic": [810000, 990000],
                "tv_video": [990000, 1210000]
            }
        )
        
        session.add(training_run)
        await session.commit()
        await session.refresh(training_run)
        
        return training_run


@pytest.fixture
def optimization_request_data():
    """Sample optimization request data."""
    return {
        "total_budget": 1000000,
        "current_spend": {
            "search_brand": 200000,
            "search_nonbrand": 300000,
            "social_facebook": 150000,
            "display_programmatic": 200000,
            "tv_video": 150000
        },
        "constraints": [
            {
                "channel": "search_brand",
                "type": "floor",
                "value": 100000,
                "description": "Minimum brand spend"
            },
            {
                "channel": "tv_video",
                "type": "cap",
                "value": 300000,
                "description": "Maximum TV spend"
            }
        ],
        "optimization_window_days": 365
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        async def get(self, key):
            return self.data.get(key)
        
        async def setex(self, key, ttl, value):
            self.data[key] = value
        
        async def delete(self, key):
            self.data.pop(key, None)
        
        async def keys(self, pattern):
            if pattern.endswith('*'):
                prefix = pattern[:-1]
                return [k for k in self.data.keys() if k.startswith(prefix)]
            return [k for k in self.data.keys() if pattern in k]
        
        async def ping(self):
            return True
    
    return MockRedis()


@pytest.fixture
def auth_headers():
    """Authentication headers for testing."""
    return {"Authorization": "Bearer test-token"}


# Test data constants
TEST_UPLOAD_ID = "test-upload-123"
TEST_RUN_ID = "test-run-456"
TEST_SESSION_ID = "test-session-789"


# Performance test constants
LOAD_TEST_CONCURRENT_REQUESTS = 10
LOAD_TEST_TOTAL_REQUESTS = 100
PERFORMANCE_THRESHOLD_MS = 5000  # 5 seconds for optimization endpoints