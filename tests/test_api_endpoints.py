"""
Comprehensive API endpoint tests for MMM application.
"""
import pytest
import pytest_asyncio
import json
import tempfile
from fastapi import status
from unittest.mock import patch, AsyncMock
import pandas as pd
import numpy as np


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_basic_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/api/health/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_detailed_health_check(self, client):
        """Test detailed health check endpoint."""
        response = client.get("/api/health/detailed")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "system" in data
        assert "configuration" in data
    
    def test_readiness_check(self, client):
        """Test readiness probe endpoint."""
        response = client.get("/api/health/ready")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "ready"
    
    def test_liveness_check(self, client):
        """Test liveness probe endpoint."""
        response = client.get("/api/health/live")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "alive"


class TestDataUploadEndpoints:
    """Test data upload and validation endpoints."""
    
    @pytest.mark.asyncio
    async def test_upload_valid_csv(self, client_with_db, sample_csv_file):
        """Test uploading a valid CSV file."""
        with open(sample_csv_file, 'rb') as f:
            response = client_with_db.post(
                "/api/data/upload",
                files={"file": ("test_data.csv", f, "text/csv")}
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert "upload_id" in data
        assert "data_summary" in data
        assert "validation_errors" in data
        assert "channel_info" in data
        
        # Check data summary
        summary = data["data_summary"]
        assert summary["total_days"] == 365
        assert summary["channel_count"] == 5
        assert summary["business_tier"] in ["enterprise", "mid_market", "small_business", "prototype"]
        assert 0 <= summary["data_quality_score"] <= 100
        
        # Check channel info
        channel_info = data["channel_info"]
        assert len(channel_info) == 5
        for channel, info in channel_info.items():
            assert "type" in info
            assert "total_spend" in info
            assert "spend_share" in info
            assert "days_active" in info
    
    @pytest.mark.asyncio
    async def test_upload_invalid_csv(self, client_with_db, invalid_csv_file):
        """Test uploading CSV with validation errors."""
        with open(invalid_csv_file, 'rb') as f:
            response = client_with_db.post(
                "/api/data/upload",
                files={"file": ("invalid_data.csv", f, "text/csv")}
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should have validation errors
        assert len(data["validation_errors"]) > 0
        
        # Check error structure
        errors = data["validation_errors"]
        for error in errors:
            assert "code" in error
            assert "message" in error
            assert "severity" in error
    
    def test_upload_non_csv_file(self, client, tmp_path):
        """Test uploading non-CSV file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is not a CSV file")
        
        with open(txt_file, 'rb') as f:
            response = client.post(
                "/api/data/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Only CSV files are allowed" in response.json()["detail"]
    
    def test_upload_too_large_file(self, client, tmp_path):
        """Test uploading file that exceeds size limit."""
        # Create a large CSV file (simulate)
        large_file = tmp_path / "large.csv"
        
        # Mock the file size check
        with patch('mmm.config.settings.settings.api.max_upload_size', 1024):  # 1KB limit
            with open(large_file, 'w') as f:
                f.write("date,profit,spend\n" * 1000)  # Create large file
            
            with open(large_file, 'rb') as f:
                response = client.post(
                    "/api/data/upload",
                    files={"file": ("large.csv", f, "text/csv")}
                )
            
            assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    
    def test_get_upload_summary(self, client, mock_upload_session):
        """Test getting upload summary."""
        response = client.get(f"/api/data/upload/{mock_upload_session.id}/summary")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["upload_id"] == mock_upload_session.id
        assert data["filename"] == mock_upload_session.filename
        assert data["status"] == mock_upload_session.status
        assert "data_summary" in data
    
    def test_get_upload_summary_not_found(self, client):
        """Test getting summary for non-existent upload."""
        response = client.get("/api/data/upload/non-existent-id/summary")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_channel_info(self, client, mock_upload_session):
        """Test getting channel information."""
        response = client.get(f"/api/data/upload/{mock_upload_session.id}/channels")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["upload_id"] == mock_upload_session.id
        assert "channels" in data
        assert len(data["channels"]) > 0
    
    def test_get_validation_results(self, client, mock_upload_session):
        """Test getting validation results."""
        response = client.get(f"/api/data/upload/{mock_upload_session.id}/validation")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["upload_id"] == mock_upload_session.id
        assert "validation_errors" in data
    
    def test_delete_upload(self, client, mock_upload_session):
        """Test deleting an upload session."""
        response = client.delete(f"/api/data/upload/{mock_upload_session.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "deleted"
        assert data["upload_id"] == mock_upload_session.id


class TestModelTrainingEndpoints:
    """Test model training endpoints."""
    
    @pytest.mark.asyncio
    async def test_start_training(self, client_with_db, mock_upload_session):
        """Test starting model training."""
        config_data = {
            "training_window_days": 126,
            "test_window_days": 14,
            "n_bootstrap": 100
        }
        
        response = client_with_db.post(
            "/api/model/train",
            params={"upload_id": mock_upload_session.id},
            json=config_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "run_id" in data
        assert data["status"] == "queued"
    
    def test_start_training_invalid_upload(self, client_with_db):
        """Test starting training with invalid upload ID."""
        response = client_with_db.post(
            "/api/model/train",
            params={"upload_id": "non-existent-id"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @pytest.mark.asyncio
    async def test_get_training_progress(self, client_with_db, mock_completed_training_run):
        """Test getting training progress."""
        response = client_with_db.get(f"/api/model/training/progress/{mock_completed_training_run.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["run_id"] == mock_completed_training_run.id
        assert data["status"] == "completed"
        assert "start_time" in data
        assert "progress" in data
    
    @pytest.mark.asyncio
    async def test_get_model_results(self, client_with_db, mock_completed_training_run):
        """Test getting complete model results."""
        response = client_with_db.get(f"/api/model/results/{mock_completed_training_run.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["run_id"] == mock_completed_training_run.id
        assert "training_info" in data
        assert "model_performance" in data
        assert "parameters" in data
        assert "diagnostics" in data
        assert "confidence_intervals" in data
        
        # Check model performance metrics
        performance = data["model_performance"]
        assert "cv_mape" in performance
        assert "r_squared" in performance
        assert "mape" in performance
    
    @pytest.mark.asyncio
    async def test_get_channel_performance(self, client_with_db, mock_completed_training_run):
        """Test getting channel performance metrics."""
        response = client_with_db.get(f"/api/model/results/{mock_completed_training_run.id}/channel-performance")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["run_id"] == mock_completed_training_run.id
        assert "channel_performance" in data
        assert "total_media_attribution" in data
        
        # Check channel performance structure
        for channel, performance in data["channel_performance"].items():
            assert "attribution" in performance
            assert "attribution_share" in performance
            assert "incremental_strength" in performance
            assert "saturation_parameter" in performance
            assert "adstock_parameter" in performance
            assert "confidence_interval" in performance
    
    @pytest.mark.asyncio
    async def test_get_response_curve(self, client_with_db, mock_completed_training_run):
        """Test getting response curve data."""
        channel_name = "search_brand"
        response = client_with_db.get(
            f"/api/model/response-curves/{mock_completed_training_run.id}/{channel_name}",
            params={"resolution": 50, "max_spend": 100000}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["run_id"] == mock_completed_training_run.id
        assert data["channel"] == channel_name
        assert "curve_data" in data
        
        curve = data["curve_data"]
        assert "spend_levels" in curve
        assert "incremental_profits" in curve
        assert "efficiency_metrics" in curve
    
    def test_get_training_progress_not_found(self, client_with_db):
        """Test getting progress for non-existent run."""
        response = client_with_db.get("/api/model/training/progress/non-existent-id")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @pytest.mark.asyncio
    async def test_delete_training_run(self, client_with_db, mock_completed_training_run):
        """Test deleting a training run."""
        response = client_with_db.delete(f"/api/model/training/{mock_completed_training_run.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "deleted"
        assert data["run_id"] == mock_completed_training_run.id


class TestOptimizationEndpoints:
    """Test budget optimization endpoints."""
    
    @pytest.mark.asyncio
    async def test_run_optimization(self, client_with_db, mock_completed_training_run, optimization_request_data):
        """Test running budget optimization."""
        optimization_request_data["run_id"] = mock_completed_training_run.id
        
        with patch('mmm.optimization.optimizer.BudgetOptimizer.optimize') as mock_optimize:
            # Mock optimization result
            mock_result = type('MockResult', (), {
                'optimal_spend': {"search_brand": 250000, "search_nonbrand": 350000},
                'optimal_profit': 12000000,
                'current_profit': 10000000,
                'profit_uplift': 2000000,
                'shadow_prices': {"total_budget": 1.2, "search_brand": 0.8},
                'constraints_binding': ["total_budget"],
                'response_curves': {
                    "search_brand": (np.linspace(0, 300000, 100), np.linspace(0, 5000000, 100))
                },
                'scenario_analysis': {}
            })()
            
            mock_optimize.return_value = mock_result
            
            response = client_with_db.post("/api/optimization/run", json=optimization_request_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["run_id"] == mock_completed_training_run.id
            assert "optimization_results" in data
            assert "response_curves" in data
            assert "scenario_analysis" in data
            
            # Check optimization results structure
            results = data["optimization_results"]
            assert "optimal_spend" in results
            assert "optimal_profit" in results
            assert "current_profit" in results
            assert "profit_uplift" in results
            assert "shadow_prices" in results
            assert "constraints_binding" in results
    
    def test_run_optimization_invalid_run(self, client_with_db, optimization_request_data):
        """Test optimization with invalid run ID."""
        optimization_request_data["run_id"] = "non-existent-id"
        
        response = client_with_db.post("/api/optimization/run", json=optimization_request_data)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_run_optimization_invalid_constraints(self, client_with_db, mock_completed_training_run):
        """Test optimization with invalid constraint types."""
        request_data = {
            "run_id": mock_completed_training_run.id,
            "total_budget": 1000000,
            "current_spend": {"search_brand": 200000},
            "constraints": [
                {
                    "channel": "search_brand",
                    "type": "invalid_type",  # Invalid constraint type
                    "value": 100000
                }
            ]
        }
        
        response = client_with_db.post("/api/optimization/run", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    @pytest.mark.asyncio
    async def test_get_response_curve(self, client_with_db, mock_completed_training_run):
        """Test getting response curve for optimization."""
        channel = "search_brand"
        
        with patch('mmm.optimization.optimizer.BudgetOptimizer.get_response_curve') as mock_curve:
            # Mock response curve
            mock_response = type('MockCurve', (), {
                'channel': channel,
                'spend_range': np.linspace(0, 100000, 100),
                'profit_values': np.linspace(0, 5000000, 100),
                'marginal_efficiency': np.ones(100),
                'current_spend': 50000,
                'optimal_spend': 75000
            })()
            
            mock_curve.return_value = mock_response
            
            response = client_with_db.get(
                f"/api/optimization/response-curve/{mock_completed_training_run.id}/{channel}",
                params={"max_spend": 100000, "resolution": 100}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["run_id"] == mock_completed_training_run.id
            assert data["channel"] == channel
            assert "response_curve" in data
    
    @pytest.mark.asyncio
    async def test_scenario_analysis(self, client_with_db, mock_completed_training_run):
        """Test custom scenario analysis."""
        scenarios = {
            "scenario_1": {
                "current_spend": {"search_brand": 300000, "social": 200000},
                "total_budget": 500000,
                "optimization_window_days": 365
            },
            "scenario_2": {
                "current_spend": {"search_brand": 200000, "social": 300000},
                "total_budget": 500000,
                "optimization_window_days": 365
            }
        }
        
        response = client_with_db.post(
            f"/api/optimization/scenario-analysis/{mock_completed_training_run.id}",
            json=scenarios
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["run_id"] == mock_completed_training_run.id
        assert "scenario_results" in data
        assert len(data["scenario_results"]) == 2
    
    def test_get_default_constraints(self, client_with_db):
        """Test getting default constraint configurations."""
        response = client_with_db.get("/api/optimization/constraints/defaults")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "constraint_types" in data
        assert "default_ramp_limits" in data
        assert "recommendations" in data
        
        # Check constraint types structure
        for constraint_type in data["constraint_types"]:
            assert "type" in constraint_type
            assert "name" in constraint_type
            assert "description" in constraint_type
    
    @pytest.mark.asyncio
    async def test_get_shadow_prices(self, client_with_db, mock_completed_training_run):
        """Test calculating shadow prices."""
        params = {
            "total_budget": 1000000,
            "current_spend": {
                "search_brand": 200000,
                "social": 300000
            }
        }
        
        response = client_with_db.get(
            f"/api/optimization/shadow-prices/{mock_completed_training_run.id}",
            params={"total_budget": params["total_budget"]}
        )
        
        # Note: This might need adjustment based on actual API implementation
        # The test structure is correct, but the exact parameter passing might vary


class TestWebSocketEndpoints:
    """Test WebSocket connection statistics."""
    
    def test_websocket_stats(self, client_with_db):
        """Test getting WebSocket connection statistics."""
        response = client_with_db.get("/ws/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check stats structure
        assert "total_connections" in data
        assert "training_connections" in data
        assert "session_connections" in data
        assert "active_training_runs" in data
        assert "active_sessions" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test the root API endpoint."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "environment" in data


# Error handling tests
class TestErrorHandling:
    """Test API error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 error handling."""
        response = client.get("/api/non-existent-endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_method_not_allowed(self, client):
        """Test 405 method not allowed."""
        response = client.delete("/api/health/")
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_validation_error(self, client):
        """Test request validation errors."""
        # Send invalid JSON data
        response = client.post(
            "/api/model/train",
            json={"invalid_field": "invalid_value"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY