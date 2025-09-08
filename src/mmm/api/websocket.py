"""
WebSocket connection manager for real-time updates.
"""
import json
import asyncio
from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import structlog

from mmm.utils.cache import cache_manager

logger = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        # Active connections by training run ID
        self.training_connections: Dict[str, Set[WebSocket]] = {}
        
        # Active connections by session ID (for general updates)
        self.session_connections: Dict[str, Set[WebSocket]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect_training(self, websocket: WebSocket, run_id: str, session_id: Optional[str] = None):
        """Connect WebSocket for training run updates."""
        await websocket.accept()
        
        # Add to training connections
        if run_id not in self.training_connections:
            self.training_connections[run_id] = set()
        self.training_connections[run_id].add(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "run_id": run_id,
            "session_id": session_id,
            "connected_at": datetime.utcnow(),
            "type": "training"
        }
        
        logger.info("WebSocket connected for training", run_id=run_id, session_id=session_id)
        
        # Send current progress if available
        await self._send_cached_progress(websocket, run_id)
    
    async def connect_session(self, websocket: WebSocket, session_id: str):
        """Connect WebSocket for general session updates."""
        await websocket.accept()
        
        # Add to session connections
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        self.session_connections[session_id].add(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "session_id": session_id,
            "connected_at": datetime.utcnow(),
            "type": "session"
        }
        
        logger.info("WebSocket connected for session", session_id=session_id)
        
        # Send welcome message
        await self._send_to_websocket(websocket, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket and clean up."""
        metadata = self.connection_metadata.get(websocket, {})
        
        try:
            # Remove from training connections
            run_id = metadata.get("run_id")
            if run_id and run_id in self.training_connections:
                self.training_connections[run_id].discard(websocket)
                if not self.training_connections[run_id]:
                    del self.training_connections[run_id]
            
            # Remove from session connections
            session_id = metadata.get("session_id")
            if session_id and session_id in self.session_connections:
                self.session_connections[session_id].discard(websocket)
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
            
            # Clean up metadata
            self.connection_metadata.pop(websocket, None)
            
            logger.info("WebSocket disconnected", metadata=metadata)
            
        except Exception as e:
            logger.error("Error during WebSocket disconnect", error=str(e))
    
    async def broadcast_training_progress(self, run_id: str, progress_data: Dict[str, Any]):
        """Broadcast training progress to all connected clients for a run."""
        if run_id not in self.training_connections:
            return
        
        message = {
            "type": "training_progress",
            "run_id": run_id,
            "data": progress_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache the progress
        await cache_manager.cache_training_progress(run_id, progress_data)
        
        # Send to all connected clients for this run
        disconnected = []
        for websocket in self.training_connections[run_id].copy():
            try:
                await self._send_to_websocket(websocket, message)
            except WebSocketDisconnect:
                disconnected.append(websocket)
            except Exception as e:
                logger.error("Error sending training progress", run_id=run_id, error=str(e))
                disconnected.append(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            await self.disconnect(websocket)
        
        logger.info("Training progress broadcast", run_id=run_id, clients=len(self.training_connections.get(run_id, [])))
    
    async def broadcast_training_complete(self, run_id: str, results: Dict[str, Any]):
        """Broadcast training completion."""
        message = {
            "type": "training_complete",
            "run_id": run_id,
            "data": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_run(run_id, message)
    
    async def broadcast_training_error(self, run_id: str, error_message: str):
        """Broadcast training error."""
        message = {
            "type": "training_error",
            "run_id": run_id,
            "data": {"error": error_message},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_run(run_id, message)
    
    async def broadcast_fold_complete(self, run_id: str, fold_data: Dict[str, Any]):
        """Broadcast CV fold completion."""
        message = {
            "type": "fold_complete",
            "run_id": run_id,
            "data": fold_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_run(run_id, message)
    
    async def send_optimization_progress(self, run_id: str, optimization_data: Dict[str, Any]):
        """Send optimization progress updates."""
        message = {
            "type": "optimization_progress",
            "run_id": run_id,
            "data": optimization_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_run(run_id, message)
    
    async def _broadcast_to_run(self, run_id: str, message: Dict[str, Any]):
        """Helper to broadcast message to all clients connected to a run."""
        if run_id not in self.training_connections:
            return
        
        disconnected = []
        for websocket in self.training_connections[run_id].copy():
            try:
                await self._send_to_websocket(websocket, message)
            except (WebSocketDisconnect, ConnectionError):
                disconnected.append(websocket)
            except Exception as e:
                logger.error("Error broadcasting to run", run_id=run_id, error=str(e))
                disconnected.append(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            await self.disconnect(websocket)
    
    async def _send_to_websocket(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to a specific WebSocket."""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error("Failed to send WebSocket message", error=str(e))
            raise
    
    async def _send_cached_progress(self, websocket: WebSocket, run_id: str):
        """Send cached progress data to newly connected client."""
        try:
            cached_progress = await cache_manager.get_training_progress(run_id)
            if cached_progress:
                message = {
                    "type": "training_progress",
                    "run_id": run_id,
                    "data": cached_progress,
                    "timestamp": datetime.utcnow().isoformat(),
                    "cached": True
                }
                await self._send_to_websocket(websocket, message)
        except Exception as e:
            logger.warning("Failed to send cached progress", run_id=run_id, error=str(e))
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        total_training_connections = sum(len(connections) for connections in self.training_connections.values())
        total_session_connections = sum(len(connections) for connections in self.session_connections.values())
        
        return {
            "total_connections": len(self.connection_metadata),
            "training_connections": total_training_connections,
            "session_connections": total_session_connections,
            "active_training_runs": len(self.training_connections),
            "active_sessions": len(self.session_connections),
            "connections_by_run": {run_id: len(connections) for run_id, connections in self.training_connections.items()},
            "connections_by_session": {session_id: len(connections) for session_id, connections in self.session_connections.items()}
        }


# Global connection manager instance
connection_manager = ConnectionManager()