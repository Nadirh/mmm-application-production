"""
WebSocket endpoints for real-time updates.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from typing import Optional
import structlog

from mmm.api.websocket import connection_manager

router = APIRouter()
logger = structlog.get_logger()


@router.websocket("/training/{run_id}")
async def websocket_training_endpoint(
    websocket: WebSocket, 
    run_id: str,
    session_id: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for training progress updates.
    
    Args:
        run_id: Training run ID to subscribe to
        session_id: Optional session ID for additional context
    """
    try:
        await connection_manager.connect_training(websocket, run_id, session_id)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client (ping/pong, etc.)
                message = await websocket.receive_text()
                
                # Handle client messages if needed
                if message == "ping":
                    await websocket.send_text("pong")
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("Error in training WebSocket", run_id=run_id, error=str(e))
                break
                
    except Exception as e:
        logger.error("Training WebSocket connection failed", run_id=run_id, error=str(e))
    finally:
        await connection_manager.disconnect(websocket)


@router.websocket("/session/{session_id}")
async def websocket_session_endpoint(
    websocket: WebSocket,
    session_id: str
):
    """
    WebSocket endpoint for general session updates.
    
    Args:
        session_id: Session ID to subscribe to
    """
    try:
        await connection_manager.connect_session(websocket, session_id)
        
        # Keep connection alive
        while True:
            try:
                message = await websocket.receive_text()
                
                # Handle client messages
                if message == "ping":
                    await websocket.send_text("pong")
                elif message == "get_stats":
                    stats = connection_manager.get_connection_stats()
                    await websocket.send_json({
                        "type": "connection_stats",
                        "data": stats
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("Error in session WebSocket", session_id=session_id, error=str(e))
                break
                
    except Exception as e:
        logger.error("Session WebSocket connection failed", session_id=session_id, error=str(e))
    finally:
        await connection_manager.disconnect(websocket)


@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    return connection_manager.get_connection_stats()