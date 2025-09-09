"""
Performance and load testing for the MMM application.

These tests validate system performance under various load conditions:
- API endpoint response times
- Concurrent request handling
- Memory usage during model training
- Database query performance
- WebSocket connection scalability
"""
import pytest
import asyncio
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import statistics
from dataclasses import dataclass

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from mmm.api.main import app
from mmm.api.websocket import connection_manager
from mmm.model.mmm_model import MMMModel
from mmm.optimization.optimizer import BudgetOptimizer


@dataclass
class PerformanceMetrics:
    """Container for performance test results."""
    response_times: List[float]
    success_rate: float
    error_rate: float
    throughput: float  # requests per second
    memory_usage_mb: float
    cpu_usage_percent: float
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0
    
    @property
    def p95_response_time(self) -> float:
        return statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else 0
    
    @property
    def p99_response_time(self) -> float:
        return statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else 0


class LoadTestRunner:
    """Utility class for running load tests."""
    
    def __init__(self, client: TestClient, max_workers: int = 10):
        self.client = client
        self.max_workers = max_workers
        self.results: List[Dict[str, Any]] = []
    
    def run_load_test(self, 
                     test_func, 
                     num_requests: int, 
                     concurrent_requests: int = 5,
                     **kwargs) -> PerformanceMetrics:
        """Run a load test with specified parameters."""
        start_time = time.time()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with ThreadPoolExecutor(max_workers=min(concurrent_requests, self.max_workers)) as executor:
            futures = []
            
            # Submit requests
            for i in range(num_requests):
                future = executor.submit(self._timed_request, test_func, **kwargs)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    response_time, success = future.result()
                    response_times.append(response_time)
                    if success:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                except Exception as e:
                    failed_requests += 1
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Final resource measurements
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = process.cpu_percent()
        
        return PerformanceMetrics(
            response_times=response_times,
            success_rate=successful_requests / num_requests if num_requests > 0 else 0,
            error_rate=failed_requests / num_requests if num_requests > 0 else 0,
            throughput=num_requests / total_duration if total_duration > 0 else 0,
            memory_usage_mb=final_memory - initial_memory,
            cpu_usage_percent=cpu_percent
        )
    
    def _timed_request(self, test_func, **kwargs) -> Tuple[float, bool]:
        """Execute a request and measure its response time."""
        start_time = time.time()
        try:
            result = test_func(**kwargs)
            end_time = time.time()
            return end_time - start_time, True
        except Exception as e:
            end_time = time.time()
            return end_time - start_time, False


@pytest.fixture
def load_test_runner(client):
    """Create a load test runner instance."""
    return LoadTestRunner(client)


@pytest.fixture
def performance_dataset(tmp_path):
    """Create a large dataset for performance testing."""
    # Generate 3 years of daily data
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    np.random.seed(42)
    
    # Many channels for stress testing
    channels = {
        f'channel_{i:02d}': {
            'base': np.random.uniform(1000, 10000),
            'std': np.random.uniform(200, 2000)
        } for i in range(20)  # 20 channels
    }
    
    data = {'date': dates.strftime('%Y-%m-%d')}
    
    # Generate complex profit model
    base_profit = 50000
    seasonal_component = 10000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    trend_component = 5 * np.arange(len(dates))
    
    media_contribution = np.zeros(len(dates))
    for channel, params in channels.items():
        spend = np.random.normal(params['base'], params['std'], len(dates)).clip(min=50)
        data[channel] = spend
        media_contribution += spend * np.random.uniform(0.05, 0.15)
    
    noise = np.random.normal(0, 5000, len(dates))
    data['profit'] = (base_profit + seasonal_component + trend_component + 
                     media_contribution + noise).clip(min=10000)
    
    df = pd.DataFrame(data)
    csv_file = tmp_path / "performance_dataset.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.mark.performance
class TestAPIPerformance:
    """Test API endpoint performance."""
    
    def test_upload_endpoint_performance(self, load_test_runner, sample_csv_file):
        """Test upload endpoint under load."""
        def upload_request():
            with open(sample_csv_file, 'rb') as f:
                response = load_test_runner.client.post(
                    "/api/v1/upload",
                    files={"file": ("test_data.csv", f, "text/csv")},
                    data={"business_tier": "enterprise"}
                )
            return response.status_code == 200
        
        metrics = load_test_runner.run_load_test(
            upload_request,
            num_requests=20,
            concurrent_requests=5
        )
        
        # Performance assertions
        assert metrics.success_rate >= 0.9  # 90% success rate
        assert metrics.avg_response_time < 5.0  # Average under 5 seconds
        assert metrics.p95_response_time < 10.0  # 95th percentile under 10 seconds
        assert metrics.throughput > 0.5  # At least 0.5 requests per second
    
    def test_training_status_endpoint_performance(self, load_test_runner, completed_training_run):
        """Test training status endpoint performance."""
        run_id = completed_training_run.id
        
        def status_request():
            response = load_test_runner.client.get(f"/api/v1/training/status/{run_id}")
            return response.status_code == 200
        
        metrics = load_test_runner.run_load_test(
            status_request,
            num_requests=100,
            concurrent_requests=10
        )
        
        # Status endpoint should be very fast
        assert metrics.success_rate >= 0.95
        assert metrics.avg_response_time < 0.5  # Average under 500ms
        assert metrics.p95_response_time < 1.0  # 95th percentile under 1 second
        assert metrics.throughput > 10  # At least 10 requests per second
    
    def test_optimization_endpoint_performance(self, load_test_runner, completed_training_run,
                                             optimization_request_data):
        """Test optimization endpoint performance."""
        run_id = completed_training_run.id
        
        def optimization_request():
            response = load_test_runner.client.post(
                f"/api/v1/optimization/{run_id}/start",
                json=optimization_request_data
            )
            return response.status_code == 200
        
        metrics = load_test_runner.run_load_test(
            optimization_request,
            num_requests=10,
            concurrent_requests=3  # Lower concurrency for CPU-intensive task
        )
        
        # Optimization is CPU-intensive but should still be reasonable
        assert metrics.success_rate >= 0.8
        assert metrics.avg_response_time < 30.0  # Average under 30 seconds
        assert metrics.p95_response_time < 60.0  # 95th percentile under 1 minute


@pytest.mark.performance
class TestModelPerformance:
    """Test mathematical model performance."""
    
    def test_mmm_model_training_performance(self, sample_csv_data):
        """Test MMM model training performance with various data sizes."""
        model = MMMModel()
        
        # Test with different data sizes
        data_sizes = [100, 365, 730]  # 100 days, 1 year, 2 years
        
        for size in data_sizes:
            # Prepare data subset
            data_subset = sample_csv_data.head(size).copy()
            
            start_time = time.time()
            
            # Train model
            model.fit(
                data_subset,
                profit_column='profit',
                date_column='date',
                training_window_days=min(90, size - 14),
                test_window_days=14,
                n_bootstrap=10  # Reduced for performance testing
            )
            
            training_time = time.time() - start_time
            
            # Performance assertions based on data size
            if size <= 365:
                assert training_time < 30  # Under 30 seconds for 1 year
            else:
                assert training_time < 120  # Under 2 minutes for 2 years
    
    def test_model_prediction_performance(self, completed_training_run, sample_csv_data):
        """Test model prediction performance."""
        model = MMMModel()
        
        # Load model parameters
        model.model_parameters = completed_training_run.model_parameters
        
        # Prepare spend scenarios
        spend_scenarios = []
        channels = ['search_brand', 'search_nonbrand', 'social_facebook', 
                   'display_programmatic', 'tv_video']
        
        for i in range(100):  # 100 scenarios
            scenario = {channel: np.random.uniform(1000, 10000) for channel in channels}
            spend_scenarios.append(scenario)
        
        start_time = time.time()
        
        # Run predictions
        predictions = []
        for scenario in spend_scenarios:
            prediction = model.predict_scenario(scenario, days=365)
            predictions.append(prediction)
        
        prediction_time = time.time() - start_time
        
        # Performance assertions
        assert prediction_time < 10  # Under 10 seconds for 100 predictions
        assert len(predictions) == 100
        avg_time_per_prediction = prediction_time / 100
        assert avg_time_per_prediction < 0.1  # Under 100ms per prediction


@pytest.mark.performance
class TestOptimizationPerformance:
    """Test budget optimization performance."""
    
    def test_budget_optimizer_performance(self, completed_training_run):
        """Test budget optimizer performance."""
        optimizer = BudgetOptimizer()
        
        # Prepare optimization problem
        total_budget = 2000000
        current_spend = {
            "search_brand": 300000,
            "search_nonbrand": 400000,
            "social_facebook": 200000,
            "display_programmatic": 250000,
            "tv_video": 350000
        }
        
        constraints = [
            {"channel": "search_brand", "type": "floor", "value": 200000},
            {"channel": "tv_video", "type": "cap", "value": 500000}
        ]
        
        start_time = time.time()
        
        # Run optimization
        result = optimizer.optimize_budget(
            model_parameters=completed_training_run.model_parameters,
            total_budget=total_budget,
            current_spend=current_spend,
            constraints=constraints,
            optimization_window_days=365
        )
        
        optimization_time = time.time() - start_time
        
        # Performance assertions
        assert optimization_time < 60  # Under 1 minute
        assert result is not None
        assert "optimal_spend" in result
        assert "expected_profit" in result


@pytest.mark.performance
class TestConcurrencyPerformance:
    """Test system performance under concurrent load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_training_runs(self, test_db, sample_csv_file):
        """Test handling multiple concurrent training runs."""
        from mmm.api.routes.training import start_training_run
        from mmm.database.models import UploadSession
        
        # Create multiple upload sessions
        upload_sessions = []
        async with test_db() as session:
            for i in range(3):
                upload_session = UploadSession(
                    id=f"concurrent-test-{i}",
                    filename=f"test_data_{i}.csv",
                    file_path=sample_csv_file,
                    status="validated",
                    business_tier="enterprise"
                )
                session.add(upload_session)
                upload_sessions.append(upload_session)
            
            await session.commit()
        
        # Start concurrent training runs
        training_config = {
            "training_window_days": 60,
            "test_window_days": 7,
            "n_bootstrap": 5  # Very small for testing
        }
        
        start_time = time.time()
        
        tasks = []
        for upload_session in upload_sessions:
            task = asyncio.create_task(
                start_training_run(upload_session.id, training_config, test_db())
            )
            tasks.append(task)
        
        # Wait for all training runs to start
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert total_time < 10  # All should start within 10 seconds
        successful_starts = sum(1 for r in results if not isinstance(r, Exception))
        assert successful_starts >= 2  # At least 2 should succeed
    
    def test_websocket_connection_scalability(self, client):
        """Test WebSocket connection handling under load."""
        successful_connections = 0
        failed_connections = 0
        connection_times = []
        
        def connect_websocket(connection_id):
            try:
                start_time = time.time()
                with client.websocket_connect(f"/ws/training/test-{connection_id}") as websocket:
                    connection_time = time.time() - start_time
                    connection_times.append(connection_time)
                    
                    # Send a test message
                    websocket.send_json({"type": "ping"})
                    
                    # Brief hold to simulate real usage
                    time.sleep(0.1)
                    
                    return True
            except Exception as e:
                return False
        
        # Test concurrent WebSocket connections
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(50):  # Try 50 concurrent connections
                future = executor.submit(connect_websocket, i)
                futures.append(future)
            
            for future in as_completed(futures):
                if future.result():
                    successful_connections += 1
                else:
                    failed_connections += 1
        
        # Performance assertions
        connection_success_rate = successful_connections / 50
        assert connection_success_rate >= 0.7  # At least 70% should succeed
        
        if connection_times:
            avg_connection_time = statistics.mean(connection_times)
            assert avg_connection_time < 1.0  # Average under 1 second


@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory usage and performance."""
    
    def test_large_dataset_memory_usage(self, performance_dataset):
        """Test memory usage with large datasets."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load large dataset
        large_data = pd.read_csv(performance_dataset)
        
        after_load_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = after_load_memory - initial_memory
        
        # Memory assertions
        assert memory_increase < 500  # Should use less than 500MB for dataset
        
        # Test model training memory usage
        model = MMMModel()
        
        try:
            model.fit(
                large_data,
                profit_column='profit',
                date_column='date',
                training_window_days=120,
                test_window_days=14,
                n_bootstrap=5  # Reduced for memory testing
            )
            
            after_training_memory = process.memory_info().rss / 1024 / 1024  # MB
            training_memory_increase = after_training_memory - after_load_memory
            
            # Training should not use excessive memory
            assert training_memory_increase < 1000  # Under 1GB additional
            
        except MemoryError:
            pytest.skip("Insufficient memory for large dataset testing")
    
    def test_memory_cleanup_after_training(self, sample_csv_data):
        """Test that memory is properly cleaned up after training."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple training cycles
        for i in range(3):
            model = MMMModel()
            model.fit(
                sample_csv_data,
                profit_column='profit',
                date_column='date',
                training_window_days=90,
                test_window_days=14,
                n_bootstrap=10
            )
            
            # Force cleanup
            del model
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not continuously grow
        assert memory_increase < 100  # Under 100MB permanent increase


@pytest.mark.performance
class TestDatabasePerformance:
    """Test database query performance."""
    
    async def test_upload_session_query_performance(self, test_db):
        """Test upload session query performance."""
        from mmm.database.models import UploadSession
        
        # Create multiple upload sessions
        async with test_db() as session:
            upload_sessions = []
            for i in range(100):
                upload_session = UploadSession(
                    id=f"perf-test-{i:03d}",
                    filename=f"test_data_{i}.csv",
                    file_path=f"/tmp/test_{i}.csv",
                    status="validated" if i % 2 == 0 else "processing",
                    business_tier=["enterprise", "mid_market", "small_business"][i % 3]
                )
                upload_sessions.append(upload_session)
                session.add(upload_session)
            
            await session.commit()
        
        # Test query performance
        async with test_db() as session:
            start_time = time.time()
            
            # Query all sessions
            from sqlalchemy import select
            result = await session.execute(select(UploadSession))
            all_sessions = result.scalars().all()
            
            query_time = time.time() - start_time
            
            # Performance assertions
            assert len(all_sessions) == 100
            assert query_time < 1.0  # Under 1 second for 100 records
            
            # Test filtered queries
            start_time = time.time()
            result = await session.execute(
                select(UploadSession).where(UploadSession.status == "validated")
            )
            validated_sessions = result.scalars().all()
            
            filtered_query_time = time.time() - start_time
            
            assert len(validated_sessions) == 50
            assert filtered_query_time < 0.5  # Under 500ms for filtered query


# Performance test configuration
PERFORMANCE_THRESHOLDS = {
    "api_response_time_avg": 5.0,  # seconds
    "api_response_time_p95": 10.0,  # seconds
    "model_training_time": 120.0,  # seconds for 2 years data
    "optimization_time": 60.0,  # seconds
    "memory_usage": 1000.0,  # MB
    "success_rate_min": 0.9,  # 90%
    "throughput_min": 0.5,  # requests per second
}


def performance_report(metrics: PerformanceMetrics) -> str:
    """Generate a performance test report."""
    return f"""
Performance Test Report:
========================
Success Rate: {metrics.success_rate:.2%}
Error Rate: {metrics.error_rate:.2%}
Average Response Time: {metrics.avg_response_time:.3f}s
95th Percentile Response Time: {metrics.p95_response_time:.3f}s
99th Percentile Response Time: {metrics.p99_response_time:.3f}s
Throughput: {metrics.throughput:.2f} requests/second
Memory Usage: {metrics.memory_usage_mb:.1f} MB
CPU Usage: {metrics.cpu_usage_percent:.1f}%
"""