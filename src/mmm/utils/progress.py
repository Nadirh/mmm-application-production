"""
Progress tracking utilities for model training with WebSocket broadcasting.
"""
import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime, UTC
import structlog

logger = structlog.get_logger()


class TrainingProgressTracker:
    """Tracks training progress and broadcasts updates via WebSocket."""

    def __init__(self, run_id: str, websocket_manager=None, training_runs_dict=None):
        self.run_id = run_id
        self.websocket_manager = websocket_manager
        self.training_runs_dict = training_runs_dict
        self.start_time = datetime.now(UTC)
        self.current_fold = 0
        self.total_folds = 0
        self.current_step = ""
        self.progress_data = {}
        
    async def update_progress(self, progress_type: str, data: Dict[str, Any]):
        """Update progress and broadcast via WebSocket."""
        timestamp = datetime.now(UTC)
        
        # Update internal state
        if progress_type == "training_started":
            self.current_step = "Training started"
            self.total_folds = data.get("total_folds", 0)
            
        elif progress_type == "fold_started":
            self.current_fold = data.get("fold", 0)
            self.current_step = f"Processing fold {self.current_fold}/{self.total_folds}"
            
        elif progress_type == "fold_complete":
            fold_num = data.get("fold", 0)
            self.current_fold = fold_num
            mape = data.get("mape", 0)
            self.current_step = f"Fold {fold_num} complete (MAPE: {mape:.2f}%)"
            
        elif progress_type == "parameter_optimization":
            combination = data.get("combination", 0)
            total_combinations = data.get("total_combinations", 0)
            self.current_step = f"Testing parameter combination {combination}/{total_combinations}"
            
        elif progress_type == "bootstrap_started":
            self.current_step = "Calculating confidence intervals"

        elif progress_type == "cv_structure":
            # Pass through CV structure info directly
            self.current_step = "Displaying CV structure"
            # Don't return early - let it broadcast

        elif progress_type == "outer_fold_start":
            fold = data.get("fold", 0)
            total_folds = data.get("total_folds", 0)
            weeks = data.get("weeks", "")
            self.current_step = f"Nested CV - Outer Fold {fold}/{total_folds} (Weeks {weeks})"
            self.current_fold = fold
            self.total_folds = total_folds

        elif progress_type == "outer_fold_complete":
            fold = data.get("fold", 0)
            mape = data.get("mape", 0)
            self.current_step = f"Outer Fold {fold} complete (MAPE: {mape:.2f}%)"

        elif progress_type == "inner_fold_info":
            outer_fold = data.get("outer_fold", 0)
            inner_train = data.get("inner_train_days", 0)
            inner_test = data.get("inner_test_days", 0)
            self.current_step = f"Inner fold: {inner_train} train days, {inner_test} test days"

        elif progress_type == "training_complete":
            self.current_step = "Training completed"

        elif progress_type == "training_error":
            self.current_step = "Training failed"
        
        # Calculate progress percentage
        progress_pct = 0
        if self.total_folds > 0:
            fold_progress = (self.current_fold - 1) / self.total_folds * 100
            if progress_type == "parameter_optimization":
                # Add sub-progress within current fold
                combination = data.get("combination", 0)
                total_combinations = data.get("total_combinations", 1)
                sub_progress = (combination / total_combinations) * (100 / self.total_folds)
                progress_pct = fold_progress + sub_progress
            else:
                progress_pct = fold_progress
            
            if progress_type == "training_complete":
                progress_pct = 100
        
        # Create progress message
        progress_message = {
            "type": progress_type,
            "run_id": self.run_id,
            "timestamp": timestamp.isoformat(),
            "elapsed_time": (timestamp - self.start_time).total_seconds(),
            "current_step": self.current_step,
            "progress_pct": min(progress_pct, 100),
            "current_fold": self.current_fold,
            "total_folds": self.total_folds,
            **data
        }
        
        # Store progress data
        self.progress_data = progress_message

        # Update the global training_runs dict if available
        if self.training_runs_dict is not None and self.run_id in self.training_runs_dict:
            self.training_runs_dict[self.run_id]["progress"] = progress_message
            self.training_runs_dict[self.run_id]["last_update"] = timestamp

        # Broadcast via WebSocket if manager available
        if self.websocket_manager:
            try:
                if progress_type == "fold_complete":
                    await self.websocket_manager.broadcast_fold_complete(self.run_id, progress_message)
                elif progress_type == "training_complete":
                    await self.websocket_manager.broadcast_training_complete(self.run_id, progress_message)
                elif progress_type == "training_error":
                    error_msg = data.get("error", "Unknown error")
                    await self.websocket_manager.broadcast_training_error(self.run_id, error_msg)
                else:
                    await self.websocket_manager.broadcast_training_progress(self.run_id, progress_message)
                    
            except Exception as e:
                logger.error("Failed to broadcast progress", run_id=self.run_id, error=str(e))
        
        # Log progress
        logger.info("Training progress", **progress_message)
    
    def get_current_progress(self) -> Dict[str, Any]:
        """Get current progress state."""
        return self.progress_data.copy()


class AsyncProgressCallback:
    """Async-compatible progress callback for model training."""
    
    def __init__(self, tracker: TrainingProgressTracker):
        self.tracker = tracker
        self.loop = None
        
    def __call__(self, progress_data: Dict[str, Any]):
        """Synchronous callback that schedules async update."""
        try:
            # Get current event loop or create new one
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Schedule the async update
            if loop.is_running():
                # If loop is running, schedule as a task
                asyncio.create_task(self._async_update(progress_data))
            else:
                # If loop is not running, run until complete
                loop.run_until_complete(self._async_update(progress_data))
                
        except Exception as e:
            logger.error("Error in progress callback", error=str(e))
    
    async def _async_update(self, progress_data: Dict[str, Any]):
        """Async update method."""
        progress_type = progress_data.get("type", "progress")
        await self.tracker.update_progress(progress_type, progress_data)


def create_progress_tracker(run_id: str, websocket_manager=None, training_runs_dict=None) -> TrainingProgressTracker:
    """Factory function to create a progress tracker."""
    return TrainingProgressTracker(run_id, websocket_manager, training_runs_dict)


def create_progress_callback(tracker: TrainingProgressTracker) -> AsyncProgressCallback:
    """Factory function to create an async progress callback."""
    return AsyncProgressCallback(tracker)