"""
Model training and results endpoints.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
import uuid
import asyncio
from datetime import datetime, UTC
import structlog
import pandas as pd

from mmm.model.mmm_model import MMMModel
from mmm.data.processor import DataProcessor
from mmm.config.settings import settings
from mmm.api.routes.data import upload_sessions, get_upload_session_from_db
from mmm.database.connection import get_db
from mmm.api.websocket import connection_manager
from mmm.utils.progress import create_progress_tracker, create_progress_callback
from mmm.model.response_curves import create_response_curve_generator
from mmm.utils.cache import cache_manager

router = APIRouter()
logger = structlog.get_logger()

# In-memory storage for training runs
training_runs: Dict[str, Dict[str, Any]] = {}


# TrainingProgress class replaced by WebSocket-enabled progress system


async def train_model_background(upload_id: str, run_id: str, config: Dict[str, Any]):
    """Background task for model training."""
    from mmm.database.connection import db_manager
    
    try:
        logger.info("Starting model training", run_id=run_id, upload_id=upload_id)
        
        # Get upload data from database or cache
        session = None
        if upload_id in upload_sessions:
            session = upload_sessions[upload_id]
        else:
            # Try to retrieve from database
            async for db in db_manager.get_session():
                session = await get_upload_session_from_db(upload_id, db)
        
        if session is None:
            raise ValueError(f"Upload session {upload_id} not found")
            
        logger.info("Found upload session", upload_id=upload_id, filename=session.get("filename"))
        processed_df = session["processed_df"]
        channel_info = session["channel_info"]
        
        # Get parameter grids (with custom r values if provided)
        processor = DataProcessor()
        custom_r_values = config.get("custom_r_values", None)
        channel_grids = processor.get_parameter_grid(channel_info, custom_r_values)
        
        # Initialize model
        model = MMMModel(
            training_window_days=config.get("training_window_days", settings.model.training_window_days),
            test_window_days=config.get("test_window_days", settings.model.test_window_days),
            n_bootstrap=config.get("n_bootstrap", settings.model.n_bootstrap)
        )
        
        # Create progress tracker with WebSocket support and training_runs reference
        progress_tracker = create_progress_tracker(run_id, connection_manager, training_runs)
        progress_callback = create_progress_callback(progress_tracker)
        
        # Update status in database and memory
        training_runs[run_id]["status"] = "training"
        training_runs[run_id]["cancelled"] = False  # Initialize cancellation flag
        
        # Update database status
        async for db_session in db_manager.get_session():
            try:
                from sqlalchemy import update
                from mmm.database.models import TrainingRun
                await db_session.execute(
                    update(TrainingRun)
                    .where(TrainingRun.id == run_id)
                    .values(status="training", current_progress={"type": "training_started"})
                )
                await db_session.commit()
                logger.info("Updated training status in database", run_id=run_id, status="training")
            except Exception as e:
                logger.error("Failed to update training status in database", run_id=run_id, error=str(e))
        
        await progress_tracker.update_progress("training_started", {
            "total_folds": len(range(0, len(processed_df) - settings.model.training_window_days - settings.model.test_window_days + 1, settings.model.test_window_days))
        })
        
        # Check for cancellation before starting expensive training
        if training_runs[run_id].get("cancelled", False):
            logger.info("Training cancelled before model fitting", run_id=run_id)
            await progress_tracker.update_progress("cancelled", {"message": "Training cancelled by user"})
            return
        
        # Train model with periodic cancellation checks and timeout
        try:
            training_timeout = 1800  # 30 minutes timeout
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    model.fit,
                    processed_df,
                    channel_grids,
                    progress_callback,
                    lambda: training_runs[run_id].get("cancelled", False)
                ),
                timeout=training_timeout
            )
        except asyncio.TimeoutError:
            # Training timed out
            error_msg = f"Training timed out after {training_timeout/60:.1f} minutes"
            logger.error("Training timeout", run_id=run_id, timeout_minutes=training_timeout/60)
            await progress_tracker.update_progress("failed", {"error": error_msg})
            # Update database status to failed
            async for db_session in db_manager.get_session():
                try:
                    from sqlalchemy import update
                    from mmm.database.models import TrainingRun
                    await db_session.execute(
                        update(TrainingRun)
                        .where(TrainingRun.id == run_id)
                        .values(status="failed", error_message=error_msg, current_progress={"type": "failed", "error": error_msg})
                    )
                    await db_session.commit()
                except Exception as db_error:
                    logger.error("Failed to update timeout status in database", run_id=run_id, error=str(db_error))
            return
        except InterruptedError as e:
            # Training was cancelled
            logger.info("Training cancelled during model fitting", run_id=run_id, error=str(e))
            await progress_tracker.update_progress("cancelled", {"message": str(e)})
            return
        
        # Store results in memory and database
        completion_time = datetime.now(UTC)
        training_runs[run_id].update({
            "status": "completed",
            "results": results,
            "model": model,
            "completion_time": completion_time,
        })
        
        # Update database with completion
        async for db_session in db_manager.get_session():
            try:
                from sqlalchemy import update
                from mmm.database.models import TrainingRun
                
                model_results = {
                    "parameters": {
                        "alpha_baseline": results.parameters.alpha_baseline,
                        "alpha_trend": results.parameters.alpha_trend,
                        "channel_alphas": results.parameters.channel_alphas,
                        "channel_betas": results.parameters.channel_betas,
                        "channel_rs": results.parameters.channel_rs
                    },
                    "performance": {
                        "cv_mape": results.cv_mape,
                        "r_squared": results.r_squared,
                        "mape": results.mape,
                        "n_folds_averaged": results.n_folds_averaged  # Add fold averaging info
                    },
                    "diagnostics": results.diagnostics,
                    "confidence_intervals": results.confidence_intervals
                }
                
                await db_session.execute(
                    update(TrainingRun)
                    .where(TrainingRun.id == run_id)
                    .values(
                        status="completed",
                        completion_time=completion_time,
                        model_parameters=model_results["parameters"],
                        model_performance=model_results["performance"],
                        diagnostics=model_results["diagnostics"],
                        confidence_intervals=model_results["confidence_intervals"],
                        current_progress={"type": "completed"}
                    )
                )
                await db_session.commit()
                logger.info("Training completion saved to database", run_id=run_id)
                
                # Cache model results
                await cache_manager.cache_model_results(run_id, model_results)
                
            except Exception as e:
                logger.error("Failed to save training completion to database", run_id=run_id, error=str(e))
        
        # Broadcast training completion
        await progress_tracker.update_progress("training_complete", {
            "cv_mape": results.cv_mape,
            "r_squared": results.r_squared,
            "final_mape": results.mape
        })
        
        logger.info(
            "Model training completed",
            run_id=run_id,
            cv_mape=results.cv_mape,
            r_squared=results.r_squared
        )
        
    except Exception as e:
        logger.error("Model training failed", run_id=run_id, error=str(e))
        completion_time = datetime.now(UTC)
        
        # Update memory
        if run_id in training_runs:
            training_runs[run_id].update({
                "status": "failed",
                "error": str(e),
                "completion_time": completion_time,
            })
        
        # Update database with error
        try:
            async for db_session in db_manager.get_session():
                from sqlalchemy import update
                from mmm.database.models import TrainingRun
                await db_session.execute(
                    update(TrainingRun)
                    .where(TrainingRun.id == run_id)
                    .values(
                        status="failed",
                        completion_time=completion_time,
                        error_message=str(e),
                        current_progress={"type": "failed", "error": str(e)}
                    )
                )
                await db_session.commit()
                logger.info("Training failure saved to database", run_id=run_id)
        except Exception as db_error:
            logger.error("Failed to save training error to database", run_id=run_id, error=str(db_error))
        
        # Broadcast training error
        if 'progress_tracker' in locals():
            await progress_tracker.update_progress("training_error", {
                "error": str(e)
            })


@router.post("/train")
async def train_model(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Start model training for uploaded data.

    Args:
        request: JSON body containing upload_id and optional config

    Returns:
        run_id: Unique identifier for this training run
    """
    # Extract parameters from request body
    upload_id = request.get("upload_id")
    config = request.get("config")

    if not upload_id:
        raise HTTPException(status_code=400, detail="upload_id is required")

    # Check if upload session exists in cache or database
    session = None
    if upload_id in upload_sessions:
        session = upload_sessions[upload_id]
    else:
        session = await get_upload_session_from_db(upload_id, db)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    # Validate data requirements before starting training
    data_summary = session.get("data_summary")
    if not data_summary:
        raise HTTPException(status_code=400, detail="No data summary found in upload session")
    total_days = data_summary.total_days
    
    # Get training configuration with defaults
    config = config or {}
    training_window_days = config.get("training_window_days", settings.model.training_window_days)
    test_window_days = config.get("test_window_days", settings.model.test_window_days)
    
    # Calculate minimum required days for cross-validation
    min_required_days = max(
        settings.model.min_training_days,  # Absolute minimum from config
        training_window_days + test_window_days  # Minimum for at least one fold
    )
    
    if total_days < min_required_days:
        error_msg = (
            f"Insufficient data for training. Your dataset has {total_days} days, "
            f"but training requires at least {min_required_days} days. "
            f"This ensures proper cross-validation with a {training_window_days}-day training window "
            f"and {test_window_days}-day test window. Please upload a dataset with more historical data."
        )
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Calculate how many cross-validation folds will be created
    available_folds = (total_days - training_window_days - test_window_days) // test_window_days + 1
    if available_folds < 1:
        error_msg = (
            f"Cannot create cross-validation folds with current settings. "
            f"Dataset: {total_days} days, Training window: {training_window_days} days, "
            f"Test window: {test_window_days} days. "
            f"Try reducing the training or test window sizes, or upload more data."
        )
        raise HTTPException(status_code=400, detail=error_msg)
    
    logger.info(
        "Data validation passed for training",
        upload_id=upload_id,
        total_days=total_days,
        min_required_days=min_required_days,
        available_folds=available_folds
    )
    
    run_id = str(uuid.uuid4())
    
    # Create database training run entry with proper transaction handling
    try:
        from mmm.database.models import TrainingRun
        
        start_time = datetime.now(UTC)
        db_training_run = TrainingRun(
            id=run_id,
            upload_session_id=upload_id,
            start_time=start_time,
            status="queued",
            training_config=config,
            current_progress={"type": "queued"}
        )
        db.add(db_training_run)
        await db.commit()
        
        logger.info("Training run created in database", run_id=run_id, upload_id=upload_id)
        
        # Also store in memory cache for immediate access
        training_runs[run_id] = {
            "upload_id": upload_id,
            "run_id": run_id,
            "start_time": start_time,
            "status": "queued",
            "config": config,
            "progress": {"type": "queued"}
        }
        
        # Start background training
        background_tasks.add_task(train_model_background, upload_id, run_id, config)
        
    except Exception as e:
        logger.error("Failed to create training run", run_id=run_id, upload_id=upload_id, error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create training run: {str(e)}")
    
    logger.info("Model training queued", run_id=run_id, upload_id=upload_id)
    
    return {"run_id": run_id, "status": "queued"}


@router.get("/channels/{upload_id}")
async def get_channel_classifications(upload_id: str) -> Dict[str, Any]:
    """Get channel classifications and default half-lives for an uploaded file."""

    if upload_id not in upload_sessions:
        raise HTTPException(status_code=404, detail="Upload not found")

    upload_info = upload_sessions[upload_id]
    df = upload_info["processed_df"]

    # Extract only paid channel names (exclude non-spend variables)
    non_spend_columns = {'date', 'revenue', 'profit', 'days_since_start', 'day_of_week',
                        'is_weekend', 'month', 'quarter', 'week_of_month', 'season'}
    channel_columns = [col for col in df.columns if col not in non_spend_columns]

    # Classify channels and get default r values
    channel_info = {}

    # Get default center r values from settings
    default_r_centers = {
        "search_brand": 0.15,      # ~15% carries to next day
        "search_non_brand": 0.2,   # ~20% carries to next day
        "social": 0.3,             # ~30% carries to next day
        "display": 0.35,           # ~35% carries to next day
        "tv_video_youtube": 0.5,   # ~50% carries to next day
        "other": 0.3               # ~30% carries to next day (default)
    }

    for channel in channel_columns:
        channel_type = settings.classify_channel_type(channel)

        # With Bayesian optimization, we don't need to send r_grid values
        # The optimization explores the full parameter space automatically
        channel_info[channel] = {
            "type": channel_type
            # Removed default_r and r_grid - not needed with Bayesian optimization
        }

    return {
        "upload_id": upload_id,
        "channels": channel_info,
        "total_channels": len(channel_columns)
    }


@router.post("/train-with-custom-r-values")
async def train_model_with_custom_r_values(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Train model with custom r values for channels."""

    upload_id = request.get("upload_id")
    config = request.get("config", {})
    custom_r_center = request.get("custom_r_values", {})

    if not upload_id:
        raise HTTPException(status_code=400, detail="upload_id is required")

    if upload_id not in upload_sessions:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Store custom r values directly in config
    # The settings module will generate appropriate grids from these center values
    config["custom_r_values"] = custom_r_center

    # Continue with normal training process
    upload_info = upload_sessions[upload_id]
    df = upload_info["processed_df"]

    # Validate data sufficiency for training with specified window sizes
    total_days = len(df)
    training_window_days = config.get("training_window_days", 84)
    test_window_days = config.get("test_window_days", 14)
    min_required_days = training_window_days + test_window_days

    if total_days < min_required_days:
        error_msg = (
            f"Insufficient data for training. Your dataset has {total_days} days, "
            f"but training requires at least {min_required_days} days."
        )
        raise HTTPException(status_code=400, detail=error_msg)

    run_id = str(uuid.uuid4())

    # Create database training run entry
    try:
        from mmm.database.models import TrainingRun

        start_time = datetime.now(UTC)
        db_training_run = TrainingRun(
            id=run_id,
            upload_session_id=upload_id,
            start_time=start_time,
            status="queued",
            training_config=config,
            current_progress={"type": "queued"}
        )
        db.add(db_training_run)
        await db.commit()

        # Store in memory cache
        training_runs[run_id] = {
            "upload_id": upload_id,
            "run_id": run_id,
            "start_time": start_time,
            "status": "queued",
            "config": config,
            "progress": {"type": "queued"}
        }

        # Start background training
        background_tasks.add_task(train_model_background, upload_id, run_id, config)

    except Exception as e:
        logger.error("Failed to create training run", run_id=run_id, error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create training run: {str(e)}")

    return {"run_id": run_id, "status": "queued"}


@router.get("/training/progress/{run_id}")
async def get_training_progress(run_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get real-time training progress."""
    # Check memory cache first
    if run_id in training_runs:
        run = training_runs[run_id]
        return {
            "run_id": run_id,
            "status": run["status"],
            "start_time": run["start_time"].isoformat(),
            "last_update": run.get("last_update", run["start_time"]).isoformat(),
            "progress": run.get("progress", {}),
            "error": run.get("error")
        }
    
    # Check database with timeout protection
    try:
        from sqlalchemy import select
        from mmm.database.models import TrainingRun
        
        # Use a simpler query with timeout
        result = await db.execute(
            select(TrainingRun.id, TrainingRun.status, TrainingRun.start_time, 
                   TrainingRun.completion_time, TrainingRun.current_progress, 
                   TrainingRun.error_message)
            .where(TrainingRun.id == run_id)
        )
        db_run = result.first()
        
        if db_run is None:
            raise HTTPException(status_code=404, detail="Training run not found")
        
        return {
            "run_id": run_id,
            "status": db_run.status,
            "start_time": db_run.start_time.isoformat() if db_run.start_time else None,
            "last_update": (db_run.completion_time or db_run.start_time).isoformat() if db_run.start_time else None,
            "progress": db_run.current_progress or {},
            "error": db_run.error_message
        }
        
    except Exception as e:
        logger.error("Failed to retrieve training progress", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve training progress")


@router.get("/results/{run_id}")
async def get_model_results(run_id: str) -> Dict[str, Any]:
    """Get complete model results after training."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    run = training_runs[run_id]
    
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Training not completed. Status: {run['status']}")
    
    results = run["results"]
    
    return {
        "run_id": run_id,
        "training_info": {
            "start_time": run["start_time"].isoformat(),
            "completion_time": run["completion_time"].isoformat(),
            "config": run["config"]
        },
        "model_performance": {
            "cv_mape": results.cv_mape,
            "r_squared": results.r_squared,
            "mape": results.mape,
            "n_folds_averaged": results.n_folds_averaged
        },
        "parameters": {
            "alpha_baseline": results.parameters.alpha_baseline,
            "alpha_trend": results.parameters.alpha_trend,
            "channel_alphas": results.parameters.channel_alphas,
            "channel_betas": results.parameters.channel_betas,
            "channel_rs": results.parameters.channel_rs
        },
        "diagnostics": results.diagnostics,
        "confidence_intervals": results.confidence_intervals,
        "cv_structure_info": results.cv_structure_info,
        "fold_parameters": results.fold_parameters if hasattr(results, 'fold_parameters') else None
    }


@router.get("/results/{run_id}/channel-performance")
async def get_channel_performance(run_id: str) -> Dict[str, Any]:
    """Get channel performance metrics."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    run = training_runs[run_id]
    
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    results = run["results"]
    channel_attributions = results.diagnostics.get("channel_attributions", {})
    
    # Calculate performance metrics for each channel
    channel_performance = {}
    total_attribution = sum(channel_attributions.values())
    
    for channel, attribution in channel_attributions.items():
        alpha = results.parameters.channel_alphas.get(channel, 0)
        beta = results.parameters.channel_betas.get(channel, 0)
        r = results.parameters.channel_rs.get(channel, 0)
        
        confidence_interval = results.confidence_intervals.get(channel, (0, 0))
        
        channel_performance[channel] = {
            "attribution": attribution,
            "attribution_share": attribution / total_attribution * 100 if total_attribution > 0 else 0,
            "incremental_strength": alpha,
            "saturation_parameter": beta,
            "adstock_parameter": r,
            "confidence_interval": {
                "lower": confidence_interval[0],
                "upper": confidence_interval[1]
            }
        }
    
    return {
        "run_id": run_id,
        "channel_performance": channel_performance,
        "total_media_attribution": results.diagnostics.get("media_attribution_percentage", 0)
    }


@router.get("/response-curves/{run_id}/{channel}")
async def get_response_curve(
    run_id: str, 
    channel: str,
    num_points: Optional[int] = 100,
    max_spend: Optional[float] = None,
    current_spend: Optional[float] = None
) -> Dict[str, Any]:
    """Get response curve data for a specific channel with caching."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    run = training_runs[run_id]
    
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    # Get cached model results first
    cached_results = await cache_manager.get_model_results(run_id)
    if cached_results:
        model_parameters = cached_results["parameters"]
    else:
        # Fallback to run data
        results = run["results"]
        model_parameters = {
            "alpha_baseline": results.parameters.alpha_baseline,
            "alpha_trend": results.parameters.alpha_trend,
            "channel_alphas": results.parameters.channel_alphas,
            "channel_betas": results.parameters.channel_betas,
            "channel_rs": results.parameters.channel_rs
        }
    
    if channel not in model_parameters["channel_alphas"]:
        raise HTTPException(status_code=404, detail="Channel not found in model")
    
    # Create cached response curve generator with uncertainty metrics
    if cached_results:
        confidence_intervals = cached_results.get('confidence_intervals')
        cv_mape = cached_results.get('performance', {}).get('cv_mape')
        holdout_mape = cached_results.get('diagnostics', {}).get('holdout_validation', {}).get('holdout_mape')
    else:
        results = run["results"]
        confidence_intervals = results.confidence_intervals if hasattr(results, 'confidence_intervals') else None
        cv_mape = results.cv_mape if hasattr(results, 'cv_mape') else None
        holdout_mape = results.diagnostics.get('holdout_validation', {}).get('holdout_mape') if hasattr(results, 'diagnostics') else None

    curve_generator = create_response_curve_generator(
        run_id, model_parameters, confidence_intervals, cv_mape, holdout_mape
    )
    
    # Determine max spend if not provided
    if max_spend is None and current_spend:
        max_spend = current_spend * 3
    elif max_spend is None:
        # Use default
        upload_session = upload_sessions[run["upload_id"]]
        processed_df = upload_session["processed_df"]
        max_spend = processed_df[channel].mean() * 3 * 365  # Annualized
    
    # Calculate 28-day average daily spend for this channel
    upload_session = upload_sessions.get(run["upload_id"])
    avg_daily_spend_28_days = None

    if upload_session and "processed_df" in upload_session:
        processed_df = upload_session["processed_df"]
        if channel in processed_df.columns:
            # Get the last 28 days of data
            if len(processed_df) >= 28:
                recent_data = processed_df.tail(28)
            else:
                recent_data = processed_df  # Use all available data if less than 28 days

            avg_daily_spend_28_days = float(recent_data[channel].mean())

    # Get response curve (with caching)
    curve = await curve_generator.get_response_curve(
        channel=channel,
        max_spend=max_spend,
        num_points=num_points,
        current_spend=current_spend
    )

    return {
        "run_id": run_id,
        "channel": channel,
        "curve_data": curve,
        "avg_daily_spend_28_days": avg_daily_spend_28_days
    }


@router.get("/response-curves/{run_id}")
async def get_all_response_curves(
    run_id: str,
    num_points: Optional[int] = 100,
    current_spend: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Get response curves for all channels with caching."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    run = training_runs[run_id]
    
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    # Get cached model results first
    cached_results = await cache_manager.get_model_results(run_id)
    if cached_results:
        model_parameters = cached_results["parameters"]
    else:
        # Fallback to run data
        results = run["results"]
        model_parameters = {
            "alpha_baseline": results.parameters.alpha_baseline,
            "alpha_trend": results.parameters.alpha_trend,
            "channel_alphas": results.parameters.channel_alphas,
            "channel_betas": results.parameters.channel_betas,
            "channel_rs": results.parameters.channel_rs
        }
    
    # Create cached response curve generator with uncertainty metrics
    if cached_results:
        confidence_intervals = cached_results.get('confidence_intervals')
        cv_mape = cached_results.get('performance', {}).get('cv_mape')
        holdout_mape = cached_results.get('diagnostics', {}).get('holdout_validation', {}).get('holdout_mape')
    else:
        results = run["results"]
        confidence_intervals = results.confidence_intervals if hasattr(results, 'confidence_intervals') else None
        cv_mape = results.cv_mape if hasattr(results, 'cv_mape') else None
        holdout_mape = results.diagnostics.get('holdout_validation', {}).get('holdout_mape') if hasattr(results, 'diagnostics') else None

    curve_generator = create_response_curve_generator(
        run_id, model_parameters, confidence_intervals, cv_mape, holdout_mape
    )
    
    # Calculate 28-day average daily spend from uploaded data
    upload_session = upload_sessions.get(run["upload_id"])
    avg_daily_spend_28_days = {}

    if upload_session and "processed_df" in upload_session:
        processed_df = upload_session["processed_df"]

        # Get the last 28 days of data
        if len(processed_df) >= 28:
            recent_data = processed_df.tail(28)
        else:
            recent_data = processed_df  # Use all available data if less than 28 days

        # Calculate average daily spend for each channel and correlations
        # Only include actual spend channels, not feature/categorical columns
        excluded_columns = ["date", "profit", "is_holiday", "promo_flag", "site_outage", "days_since_start",
                           "day_of_week", "is_weekend", "month", "quarter", "year", "day_of_year"]
        channel_columns = [col for col in processed_df.columns if col not in excluded_columns]

        # Calculate correlations with profit
        spend_correlations = {}

        logger.info(f"Calculating correlations: channel_columns={channel_columns}, profit_column_exists={'profit' in processed_df.columns}, data_rows={len(processed_df)}")

        for channel in channel_columns:
            if channel in processed_df.columns:
                # Calculate 28-day average if channel exists in recent data
                if channel in recent_data.columns:
                    avg_daily_spend_28_days[channel] = float(recent_data[channel].mean())
                else:
                    avg_daily_spend_28_days[channel] = float(processed_df[channel].mean())

                # Calculate spend correlation with profit
                if "profit" in processed_df.columns and len(processed_df) > 10:
                    try:
                        # Debug data for correlation
                        channel_data = processed_df[channel]
                        profit_data = processed_df["profit"]

                        logger.info(f"Channel {channel} data: mean={channel_data.mean():.2f}, std={channel_data.std():.2f}, non_zero_count={(channel_data > 0).sum()}")
                        logger.info(f"Profit data: mean={profit_data.mean():.2f}, std={profit_data.std():.2f}")

                        # Check for any NaN values before correlation
                        channel_na_count = channel_data.isna().sum()
                        profit_na_count = profit_data.isna().sum()
                        logger.info(f"NaN values - channel: {channel_na_count}, profit: {profit_na_count}")
                        logger.info(f"Data types - channel: {channel_data.dtype}, profit: {profit_data.dtype}")

                        # Ensure both are numeric
                        channel_data_numeric = pd.to_numeric(channel_data, errors='coerce')
                        profit_data_numeric = pd.to_numeric(profit_data, errors='coerce')

                        spend_corr = channel_data_numeric.corr(profit_data_numeric)
                        logger.info(f"Spend correlation for {channel}: {spend_corr:.4f}")
                        spend_correlations[channel] = float(spend_corr) if not pd.isna(spend_corr) else 0.0

                    except Exception as e:
                        logger.error(f"Error calculating correlation for {channel}: {str(e)}")
                        spend_correlations[channel] = 0.0
                else:
                    logger.warning(f"Skipping correlation for {channel}: profit column missing or insufficient data")
            else:
                logger.warning(f"Channel {channel} not found in recent_data columns: {list(recent_data.columns)}")

        logger.info(f"Final correlations - spend: {spend_correlations}")

    # Get all response curves (with caching) - pass 28-day averages for max spend calculation
    curves = await curve_generator.get_all_response_curves(
        num_points=num_points,
        current_spend=avg_daily_spend_28_days
    )

    return {
        "run_id": run_id,
        "response_curves": curves,
        "avg_daily_spend_28_days": avg_daily_spend_28_days,
        "spend_correlations": spend_correlations
    }


@router.get("/response-curves/{run_id}/analysis")
async def get_response_curve_analysis(run_id: str) -> Dict[str, Any]:
    """Get response curve analysis and comparisons."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    run = training_runs[run_id]
    
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    # Get cached model results first
    cached_results = await cache_manager.get_model_results(run_id)
    if cached_results:
        model_parameters = cached_results["parameters"]
    else:
        # Fallback to run data
        results = run["results"]
        model_parameters = {
            "alpha_baseline": results.parameters.alpha_baseline,
            "alpha_trend": results.parameters.alpha_trend,
            "channel_alphas": results.parameters.channel_alphas,
            "channel_betas": results.parameters.channel_betas,
            "channel_rs": results.parameters.channel_rs
        }
    
    # Create cached response curve generator with uncertainty metrics
    if cached_results:
        confidence_intervals = cached_results.get('confidence_intervals')
        cv_mape = cached_results.get('performance', {}).get('cv_mape')
        holdout_mape = cached_results.get('diagnostics', {}).get('holdout_validation', {}).get('holdout_mape')
    else:
        results = run["results"]
        confidence_intervals = results.confidence_intervals if hasattr(results, 'confidence_intervals') else None
        cv_mape = results.cv_mape if hasattr(results, 'cv_mape') else None
        holdout_mape = results.diagnostics.get('holdout_validation', {}).get('holdout_mape') if hasattr(results, 'diagnostics') else None

    curve_generator = create_response_curve_generator(
        run_id, model_parameters, confidence_intervals, cv_mape, holdout_mape
    )
    
    # Get all curves
    curves = await curve_generator.get_all_response_curves()
    
    # Analyze curves
    from mmm.model.response_curves import compare_response_curves, calculate_portfolio_efficiency
    
    # Compare by peak efficiency
    efficiency_comparison = compare_response_curves(curves, "peak_efficiency")
    
    # Calculate portfolio metrics if we have current spend data
    upload_session = upload_sessions[run["upload_id"]]
    processed_df = upload_session["processed_df"]
    
    # Estimate current annual spend
    current_spend = {}
    for channel in model_parameters["channel_alphas"].keys():
        if channel in processed_df.columns:
            current_spend[channel] = processed_df[channel].sum()  # Sum over training period
    
    portfolio_efficiency = calculate_portfolio_efficiency(curves, current_spend)
    
    return {
        "run_id": run_id,
        "efficiency_comparison": efficiency_comparison,
        "portfolio_efficiency": portfolio_efficiency,
        "recommendations": {
            "top_efficiency_channels": efficiency_comparison[:3],
            "underperforming_channels": efficiency_comparison[-3:],
            "overall_efficiency_score": portfolio_efficiency["efficiency_score"]
        }
    }


@router.post("/training/cancel/{run_id}")
async def cancel_training(run_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, str]:
    """Cancel an active training run."""
    # Check if run exists in memory
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    run = training_runs[run_id]
    
    # Can only cancel running or queued training
    if run["status"] not in ["queued", "training"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel training with status: {run['status']}")
    
    # Set cancellation flag
    training_runs[run_id]["cancelled"] = True
    training_runs[run_id]["status"] = "cancelled"
    
    # Update database status
    try:
        from sqlalchemy import update
        from mmm.database.models import TrainingRun
        
        await db.execute(
            update(TrainingRun)
            .where(TrainingRun.id == run_id)
            .values(
                status="cancelled",
                completion_time=datetime.now(UTC),
                current_progress={"type": "cancelled", "message": "Training cancelled by user"}
            )
        )
        await db.commit()
        
        logger.info("Training run cancelled", run_id=run_id)
        
        # Notify via WebSocket if connected
        await connection_manager.broadcast_training_progress(run_id, {
            "type": "cancelled",
            "message": "Training cancelled by user",
            "status": "cancelled"
        })
        
    except Exception as e:
        logger.error("Failed to update cancelled training status in database", run_id=run_id, error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to cancel training: {str(e)}")
    
    return {"message": "Training run cancelled successfully", "run_id": run_id}


@router.delete("/training/{run_id}")
async def delete_training_run(run_id: str) -> Dict[str, str]:
    """Delete a training run and associated data."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    del training_runs[run_id]
    
    logger.info("Training run deleted", run_id=run_id)
    
    return {"status": "deleted", "run_id": run_id}


@router.post("/training/force-cancel/{run_id}")
async def force_cancel_training(run_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, str]:
    """Force cancel a stuck training run (admin endpoint for orphaned runs)."""
    try:
        # Update database directly without checking in-memory state
        from sqlalchemy import update
        from mmm.database.models import TrainingRun

        result = await db.execute(
            update(TrainingRun)
            .where(TrainingRun.id == run_id)
            .values(
                status="cancelled",
                completion_time=datetime.now(UTC),
                current_progress={"type": "cancelled", "message": "Training force-cancelled (system restart)"},
                error_message="Training was interrupted by system restart"
            )
        )

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Training run not found in database")

        await db.commit()

        # Clean up in-memory state if exists
        if run_id in training_runs:
            training_runs[run_id]["status"] = "cancelled"
            training_runs[run_id]["cancelled"] = True

        logger.info("Training run force-cancelled", run_id=run_id)

        # Notify via WebSocket if connected (ignore errors for orphaned runs)
        try:
            await connection_manager.broadcast_training_progress(run_id, {
                "type": "cancelled",
                "message": "Training force-cancelled (system restart)",
                "status": "cancelled"
            })
        except Exception as ws_error:
            # Ignore WebSocket errors for orphaned runs
            logger.debug("WebSocket notification failed (expected for orphaned runs)",
                        run_id=run_id, error=str(ws_error))

        return {"message": "Training run force-cancelled successfully", "run_id": run_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to force-cancel training", run_id=run_id, error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to force-cancel training: {str(e)}")


@router.get("/marginal-roi/{run_id}")
async def get_marginal_roi(run_id: str) -> Dict[str, Any]:
    """Get marginal ROI for all channels at current spend levels."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")

    run = training_runs[run_id]

    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")

    # Get cached model results first
    cached_results = await cache_manager.get_model_results(run_id)
    if cached_results:
        model_parameters = cached_results["parameters"]
    else:
        # Fallback to run data
        results = run["results"]
        model_parameters = {
            "alpha_baseline": results.parameters.alpha_baseline,
            "alpha_trend": results.parameters.alpha_trend,
            "channel_alphas": results.parameters.channel_alphas,
            "channel_betas": results.parameters.channel_betas,
            "channel_rs": results.parameters.channel_rs
        }

    # Get 28-day average daily spend from uploaded data
    upload_session = upload_sessions.get(run["upload_id"])

    if not upload_session or "processed_df" not in upload_session:
        raise HTTPException(status_code=400, detail="Upload data not found")

    processed_df = upload_session["processed_df"]

    # Get the last 28 days of data
    if len(processed_df) >= 28:
        recent_data = processed_df.tail(28)
    else:
        recent_data = processed_df  # Use all available data if less than 28 days

    # Calculate average daily spend for each channel
    excluded_columns = ["date", "profit", "is_holiday", "promo_flag", "site_outage", "days_since_start",
                       "day_of_week", "is_weekend", "month", "quarter", "year", "day_of_year"]
    channel_columns = [col for col in processed_df.columns if col not in excluded_columns]

    current_spend = {}
    baseline_spend_per_day = {}
    marginal_roi_by_channel = {}

    for channel in channel_columns:
        if channel in model_parameters["channel_alphas"]:
            # Get model parameters for this channel
            alpha = model_parameters["channel_alphas"][channel]
            beta = model_parameters["channel_betas"][channel]
            r = model_parameters["channel_rs"][channel]

            # Calculate 28-day average daily spend
            if channel in recent_data.columns:
                avg_daily_spend = float(recent_data[channel].mean())
            else:
                avg_daily_spend = float(processed_df[channel].mean())

            current_spend[channel] = avg_daily_spend
            baseline_spend_per_day[channel] = avg_daily_spend

            # Calculate marginal ROI at current spend level
            # Apply adstock transformation
            adstock_factor = 1 / (1 - r) if r < 1 else 1

            if avg_daily_spend > 0:
                # Calculate marginal ROI: first derivative of the response function
                # d/dx[alpha * (x/(1-r))^beta] = alpha * beta * (1/(1-r))^beta * x^(beta-1)
                # This gives the additional profit per additional dollar spent
                marginal_roi = alpha * beta * (adstock_factor ** beta) * (avg_daily_spend ** (beta - 1))
                marginal_roi_by_channel[channel] = float(max(0, marginal_roi))
            else:
                # For zero spend, calculate marginal ROI at the limit as x approaches 0
                # If beta >= 1, marginal ROI at x=0 is 0
                # If beta < 1, marginal ROI at x=0 approaches infinity, so we use the alpha * beta * adstock_factor
                if beta >= 1:
                    marginal_roi_by_channel[channel] = 0.0
                else:
                    # For diminishing returns (beta < 1), at x=0 the marginal return is very high
                    # We'll use a small epsilon to approximate the derivative near zero
                    epsilon = 0.01
                    marginal_roi = alpha * beta * (adstock_factor ** beta) * (epsilon ** (beta - 1))
                    marginal_roi_by_channel[channel] = float(max(0, marginal_roi))

    # Generate interpretation
    avg_roi = sum(marginal_roi_by_channel.values()) / len(marginal_roi_by_channel) if marginal_roi_by_channel else 0
    interpretation = f"Average marginal ROI across channels: ${avg_roi:.2f} per dollar spent"

    return {
        "run_id": run_id,
        "marginal_roi_by_channel": marginal_roi_by_channel,
        "current_spend": current_spend,
        "baseline_spend_per_day": baseline_spend_per_day,
        "interpretation": interpretation
    }