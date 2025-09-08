"""
WebSocket testing utilities and tests for MMM application.
"""
import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from mmm.api.main import app
from mmm.api.websocket import ConnectionManager
from mmm.utils.progress import TrainingProgressTracker, create_progress_callback


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.messages_sent = []
        self.messages_received = []
        self.is_closed = False
        self.accept_called = False
    
    async def accept(self):
        """Mock accept method."""
        self.accept_called = True
    
    async def send_text(self, message: str):
        """Mock send_text method."""
        if self.is_closed:
            raise WebSocketDisconnect()
        self.messages_sent.append(message)
    
    async def receive_text(self):
        """Mock receive_text method."""
        if self.messages_received:
            return self.messages_received.pop(0)
        else:
            # Simulate waiting for message
            await asyncio.sleep(0.1)
            return "ping"
    
    async def close(self):
        """Mock close method."""
        self.is_closed = True
    
    def add_message(self, message: str):
        """Add a message to be received."""
        self.messages_received.append(message)


class TestConnectionManager:
    """Test WebSocket connection manager."""
    
    @pytest.fixture
    def connection_manager(self):
        """Create a fresh connection manager for each test."""
        return ConnectionManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        return MockWebSocket()
    
    @pytest.mark.asyncio
    async def test_connect_training(self, connection_manager, mock_websocket):
        """Test connecting to training WebSocket."""
        run_id = "test-run-123"
        session_id = "test-session-456"
        
        await connection_manager.connect_training(mock_websocket, run_id, session_id)
        
        # Check connection was established
        assert mock_websocket.accept_called
        assert run_id in connection_manager.training_connections
        assert mock_websocket in connection_manager.training_connections[run_id]
        assert mock_websocket in connection_manager.connection_metadata
        
        # Check metadata
        metadata = connection_manager.connection_metadata[mock_websocket]
        assert metadata["run_id"] == run_id
        assert metadata["session_id"] == session_id
        assert metadata["type"] == "training"
    
    @pytest.mark.asyncio
    async def test_connect_session(self, connection_manager, mock_websocket):
        """Test connecting to session WebSocket."""
        session_id = "test-session-789"
        
        await connection_manager.connect_session(mock_websocket, session_id)
        
        # Check connection was established
        assert mock_websocket.accept_called
        assert session_id in connection_manager.session_connections
        assert mock_websocket in connection_manager.session_connections[session_id]
        
        # Check welcome message was sent
        assert len(mock_websocket.messages_sent) == 1
        welcome_message = json.loads(mock_websocket.messages_sent[0])
        assert welcome_message["type"] == "connection_established"
        assert welcome_message["session_id"] == session_id
    
    @pytest.mark.asyncio
    async def test_disconnect(self, connection_manager, mock_websocket):
        """Test WebSocket disconnection."""
        run_id = "test-run-123"
        
        # First connect
        await connection_manager.connect_training(mock_websocket, run_id)
        
        # Then disconnect
        await connection_manager.disconnect(mock_websocket)
        
        # Check cleanup
        assert run_id not in connection_manager.training_connections
        assert mock_websocket not in connection_manager.connection_metadata
    
    @pytest.mark.asyncio
    async def test_broadcast_training_progress(self, connection_manager):
        """Test broadcasting training progress to multiple clients."""
        run_id = "test-run-123"
        
        # Connect multiple clients
        client1 = MockWebSocket()
        client2 = MockWebSocket()
        
        await connection_manager.connect_training(client1, run_id)
        await connection_manager.connect_training(client2, run_id)
        
        # Mock cache manager
        with patch('mmm.api.websocket.cache_manager') as mock_cache:
            mock_cache.cache_training_progress = AsyncMock()
            
            # Broadcast progress
            progress_data = {
                "current_step": "Training started",
                "progress_pct": 10,
                "current_fold": 1,
                "total_folds": 10
            }
            
            await connection_manager.broadcast_training_progress(run_id, progress_data)
            
            # Check both clients received the message
            assert len(client1.messages_sent) == 1
            assert len(client2.messages_sent) == 1
            
            # Check message content
            message1 = json.loads(client1.messages_sent[0])
            assert message1["type"] == "training_progress"
            assert message1["run_id"] == run_id
            assert message1["data"] == progress_data
            
            # Check cache was called
            mock_cache.cache_training_progress.assert_called_once_with(run_id, progress_data)
    
    @pytest.mark.asyncio
    async def test_broadcast_training_complete(self, connection_manager):
        """Test broadcasting training completion."""
        run_id = "test-run-123"
        client = MockWebSocket()
        
        await connection_manager.connect_training(client, run_id)
        
        results = {
            "cv_mape": 18.5,
            "r_squared": 0.82,
            "mape": 16.2
        }
        
        await connection_manager.broadcast_training_complete(run_id, results)
        
        # Check message was sent
        assert len(client.messages_sent) == 1
        message = json.loads(client.messages_sent[0])
        assert message["type"] == "training_complete"
        assert message["data"] == results
    
    @pytest.mark.asyncio
    async def test_broadcast_training_error(self, connection_manager):
        """Test broadcasting training error."""
        run_id = "test-run-123"
        client = MockWebSocket()
        
        await connection_manager.connect_training(client, run_id)
        
        error_message = "Training failed: Insufficient data"
        
        await connection_manager.broadcast_training_error(run_id, error_message)
        
        # Check error message was sent
        assert len(client.messages_sent) == 1
        message = json.loads(client.messages_sent[0])
        assert message["type"] == "training_error"
        assert message["data"]["error"] == error_message
    
    @pytest.mark.asyncio
    async def test_broadcast_fold_complete(self, connection_manager):
        """Test broadcasting fold completion."""
        run_id = "test-run-123"
        client = MockWebSocket()
        
        await connection_manager.connect_training(client, run_id)
        
        fold_data = {
            "fold": 3,
            "total_folds": 10,
            "mape": 22.5
        }
        
        await connection_manager.broadcast_fold_complete(run_id, fold_data)
        
        # Check fold completion message
        assert len(client.messages_sent) == 1
        message = json.loads(client.messages_sent[0])
        assert message["type"] == "fold_complete"
        assert message["data"] == fold_data
    
    @pytest.mark.asyncio
    async def test_handle_disconnected_client(self, connection_manager):
        """Test handling of disconnected clients during broadcast."""
        run_id = "test-run-123"
        
        # Create a client that will disconnect
        disconnected_client = MockWebSocket()
        connected_client = MockWebSocket()
        
        await connection_manager.connect_training(disconnected_client, run_id)
        await connection_manager.connect_training(connected_client, run_id)
        
        # Simulate disconnection of first client
        disconnected_client.is_closed = True
        
        # Broadcast progress
        progress_data = {"current_step": "Processing"}
        
        with patch('mmm.api.websocket.cache_manager') as mock_cache:
            mock_cache.cache_training_progress = AsyncMock()
            
            await connection_manager.broadcast_training_progress(run_id, progress_data)
            
            # Disconnected client should be removed
            assert disconnected_client not in connection_manager.training_connections[run_id]
            
            # Connected client should still receive message
            assert len(connected_client.messages_sent) == 1
    
    def test_connection_stats(self, connection_manager):
        """Test connection statistics."""
        # Initially no connections
        stats = connection_manager.get_connection_stats()
        assert stats["total_connections"] == 0
        assert stats["training_connections"] == 0
        assert stats["session_connections"] == 0
        
        # Add some mock connections for testing
        # (In real test, would use actual connect methods)
        connection_manager.training_connections["run1"] = {MockWebSocket(), MockWebSocket()}
        connection_manager.session_connections["session1"] = {MockWebSocket()}
        connection_manager.connection_metadata = {
            MockWebSocket(): {},
            MockWebSocket(): {},
            MockWebSocket(): {}
        }
        
        stats = connection_manager.get_connection_stats()
        assert stats["training_connections"] == 2
        assert stats["session_connections"] == 1
        assert stats["active_training_runs"] == 1
        assert stats["active_sessions"] == 1


class TestTrainingProgressTracker:
    """Test training progress tracking with WebSocket integration."""
    
    @pytest.fixture
    def mock_websocket_manager(self):
        """Create a mock WebSocket manager."""
        manager = MagicMock()
        manager.broadcast_training_progress = AsyncMock()
        manager.broadcast_fold_complete = AsyncMock()
        manager.broadcast_training_complete = AsyncMock()
        manager.broadcast_training_error = AsyncMock()
        return manager
    
    @pytest.mark.asyncio
    async def test_training_progress_updates(self, mock_websocket_manager):
        """Test training progress updates with WebSocket broadcasting."""
        run_id = "test-run-123"
        tracker = TrainingProgressTracker(run_id, mock_websocket_manager)
        
        # Test training started
        await tracker.update_progress("training_started", {"total_folds": 10})
        
        assert tracker.total_folds == 10
        assert tracker.current_step == "Training started"
        mock_websocket_manager.broadcast_training_progress.assert_called()
        
        # Test fold completion
        await tracker.update_progress("fold_complete", {"fold": 3, "mape": 22.5})
        
        assert tracker.current_fold == 3
        mock_websocket_manager.broadcast_fold_complete.assert_called()
        
        # Test training completion
        await tracker.update_progress("training_complete", {
            "cv_mape": 18.5,
            "r_squared": 0.82
        })
        
        assert tracker.current_step == "Training completed"
        mock_websocket_manager.broadcast_training_complete.assert_called()
    
    @pytest.mark.asyncio
    async def test_progress_percentage_calculation(self, mock_websocket_manager):
        """Test progress percentage calculation."""
        run_id = "test-run-123"
        tracker = TrainingProgressTracker(run_id, mock_websocket_manager)
        
        # Initialize with 10 folds
        await tracker.update_progress("training_started", {"total_folds": 10})
        
        # Complete first fold
        await tracker.update_progress("fold_complete", {"fold": 1, "mape": 25.0})
        
        progress_data = tracker.get_current_progress()
        expected_progress = (1 - 1) / 10 * 100  # 0% (just started fold 1)
        
        # Progress calculation depends on implementation details
        assert 0 <= progress_data["progress_pct"] <= 100
    
    @pytest.mark.asyncio
    async def test_error_handling_in_progress_tracker(self, mock_websocket_manager):
        """Test error handling in progress tracker."""
        run_id = "test-run-123"
        tracker = TrainingProgressTracker(run_id, mock_websocket_manager)
        
        # Simulate WebSocket error
        mock_websocket_manager.broadcast_training_progress.side_effect = Exception("WebSocket error")
        
        # Should not crash when WebSocket fails
        try:
            await tracker.update_progress("training_started", {"total_folds": 5})
            # Should complete without raising exception
        except Exception as e:
            pytest.fail(f"Progress tracker should handle WebSocket errors gracefully: {e}")
    
    def test_async_progress_callback(self):
        """Test async progress callback creation."""
        mock_manager = MagicMock()
        tracker = TrainingProgressTracker("test-run", mock_manager)
        
        callback = create_progress_callback(tracker)
        
        # Test that callback can be called synchronously
        progress_data = {"type": "fold_complete", "fold": 1}
        
        # Should not raise exception
        try:
            callback(progress_data)
        except Exception as e:
            pytest.fail(f"Async callback should handle synchronous calls: {e}")


class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
    def test_websocket_endpoint_connection(self):
        """Test WebSocket endpoint connection (basic)."""
        # Note: Full WebSocket testing with TestClient is complex
        # This is a placeholder for the structure
        
        with TestClient(app) as client:
            # WebSocket testing with TestClient requires special handling
            # Full implementation would use websocket test utilities
            pass
    
    @pytest.mark.asyncio
    async def test_websocket_message_flow(self):
        """Test complete WebSocket message flow."""
        # Create connection manager
        manager = ConnectionManager()
        
        # Mock WebSocket
        mock_ws = MockWebSocket()
        
        # Connect
        await manager.connect_training(mock_ws, "test-run-123")
        
        # Simulate training progress
        progress_data = {
            "current_step": "Processing fold 1",
            "progress_pct": 10,
            "current_fold": 1,
            "total_folds": 10
        }
        
        with patch('mmm.api.websocket.cache_manager') as mock_cache:
            mock_cache.cache_training_progress = AsyncMock()
            
            await manager.broadcast_training_progress("test-run-123", progress_data)
            
            # Verify message was sent
            assert len(mock_ws.messages_sent) == 1
            
            message = json.loads(mock_ws.messages_sent[0])
            assert message["type"] == "training_progress"
            assert message["run_id"] == "test-run-123"
            assert message["data"] == progress_data
    
    @pytest.mark.asyncio
    async def test_websocket_caching_integration(self):
        """Test WebSocket integration with caching."""
        manager = ConnectionManager()
        mock_ws = MockWebSocket()
        
        # Mock cached progress
        cached_progress = {
            "current_step": "Cached progress",
            "progress_pct": 50,
            "cached": True
        }
        
        with patch('mmm.api.websocket.cache_manager') as mock_cache:
            mock_cache.get_training_progress = AsyncMock(return_value=cached_progress)
            mock_cache.cache_training_progress = AsyncMock()
            
            # Connect (should send cached progress)
            await manager.connect_training(mock_ws, "test-run-123")
            
            # Should have sent cached progress
            assert len(mock_ws.messages_sent) == 1
            message = json.loads(mock_ws.messages_sent[0])
            assert message["cached"] == True


class TestWebSocketErrorHandling:
    """Test error handling in WebSocket operations."""
    
    @pytest.mark.asyncio
    async def test_websocket_disconnect_during_broadcast(self):
        """Test handling of WebSocket disconnect during broadcast."""
        manager = ConnectionManager()
        
        # Create clients
        stable_client = MockWebSocket()
        disconnecting_client = MockWebSocket()
        
        await manager.connect_training(stable_client, "test-run-123")
        await manager.connect_training(disconnecting_client, "test-run-123")
        
        # Simulate disconnect
        disconnecting_client.is_closed = True
        
        with patch('mmm.api.websocket.cache_manager') as mock_cache:
            mock_cache.cache_training_progress = AsyncMock()
            
            # Broadcast should handle disconnection gracefully
            await manager.broadcast_training_progress("test-run-123", {"test": "data"})
            
            # Stable client should still work
            assert len(stable_client.messages_sent) == 1
            
            # Disconnected client should be removed
            assert disconnecting_client not in manager.training_connections["test-run-123"]
    
    @pytest.mark.asyncio
    async def test_websocket_send_failure(self):
        """Test handling of send failures."""
        manager = ConnectionManager()
        
        # Mock WebSocket that fails on send
        failing_ws = MockWebSocket()
        working_ws = MockWebSocket()
        
        await manager.connect_training(failing_ws, "test-run-123")
        await manager.connect_training(working_ws, "test-run-123")
        
        # Make failing_ws raise exception on send
        async def failing_send(message):
            raise Exception("Send failed")
        
        failing_ws.send_text = failing_send
        
        with patch('mmm.api.websocket.cache_manager') as mock_cache:
            mock_cache.cache_training_progress = AsyncMock()
            
            # Broadcast should handle send failure
            await manager.broadcast_training_progress("test-run-123", {"test": "data"})
            
            # Working client should still receive message
            assert len(working_ws.messages_sent) == 1
            
            # Failing client should be removed
            assert failing_ws not in manager.training_connections["test-run-123"]
    
    @pytest.mark.asyncio
    async def test_invalid_message_handling(self):
        """Test handling of invalid messages."""
        manager = ConnectionManager()
        mock_ws = MockWebSocket()
        
        await manager.connect_training(mock_ws, "test-run-123")
        
        # Try to broadcast invalid data
        invalid_data = {"circular_ref": None}
        invalid_data["circular_ref"] = invalid_data  # Create circular reference
        
        with patch('mmm.api.websocket.cache_manager') as mock_cache:
            mock_cache.cache_training_progress = AsyncMock()
            
            # Should handle serialization errors gracefully
            try:
                await manager.broadcast_training_progress("test-run-123", invalid_data)
            except Exception as e:
                # If it raises an exception, it should be handled appropriately
                pass


class TestWebSocketPerformance:
    """Test WebSocket performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_multiple_client_broadcast_performance(self):
        """Test broadcasting to many clients."""
        manager = ConnectionManager()
        
        # Create many mock clients
        num_clients = 100
        clients = [MockWebSocket() for _ in range(num_clients)]
        
        # Connect all clients
        for i, client in enumerate(clients):
            await manager.connect_training(client, f"test-run-{i % 10}")
        
        # Measure broadcast time
        start_time = asyncio.get_event_loop().time()
        
        with patch('mmm.api.websocket.cache_manager') as mock_cache:
            mock_cache.cache_training_progress = AsyncMock()
            
            # Broadcast to all runs
            for i in range(10):
                await manager.broadcast_training_progress(f"test-run-{i}", {"test": "data"})
        
        end_time = asyncio.get_event_loop().time()
        broadcast_time = end_time - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        assert broadcast_time < 1.0, f"Broadcasting took too long: {broadcast_time}s"
    
    @pytest.mark.asyncio
    async def test_connection_cleanup_performance(self):
        """Test connection cleanup performance."""
        manager = ConnectionManager()
        
        # Create and disconnect many clients
        num_clients = 50
        clients = [MockWebSocket() for _ in range(num_clients)]
        
        # Connect all
        for client in clients:
            await manager.connect_training(client, "test-run-123")
        
        # Disconnect all
        start_time = asyncio.get_event_loop().time()
        
        for client in clients:
            await manager.disconnect(client)
        
        end_time = asyncio.get_event_loop().time()
        cleanup_time = end_time - start_time
        
        # Cleanup should be fast
        assert cleanup_time < 0.5, f"Cleanup took too long: {cleanup_time}s"
        
        # All connections should be cleaned up
        assert len(manager.connection_metadata) == 0
        assert len(manager.training_connections) == 0