"""
Model training and results endpoints.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import uuid
import asyncio
from datetime import datetime
import structlog

from mmm.model.mmm_model import MMMModel
from mmm.data.processor import DataProcessor
from mmm.config.settings import settings
from mmm.api.routes.data import upload_sessions
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
    try:
        logger.info("Starting model training", run_id=run_id, upload_id=upload_id)
        
        # Get upload data
        if upload_id not in upload_sessions:
            raise ValueError(f"Upload session {upload_id} not found")
        
        session = upload_sessions[upload_id]
        processed_df = session["processed_df"]
        channel_info = session["channel_info"]
        
        # Get parameter grids
        processor = DataProcessor()
        channel_grids = processor.get_parameter_grid(channel_info)
        
        # Initialize model
        model = MMMModel(
            training_window_days=config.get("training_window_days", settings.model.training_window_days),
            test_window_days=config.get("test_window_days", settings.model.test_window_days),
            n_bootstrap=config.get("n_bootstrap", settings.model.n_bootstrap)
        )
        
        # Create progress tracker with WebSocket support
        progress_tracker = create_progress_tracker(run_id, connection_manager)
        progress_callback = create_progress_callback(progress_tracker)
        
        # Update status and broadcast training start
        training_runs[run_id]["status"] = "training"
        await progress_tracker.update_progress("training_started", {
            "total_folds": len(range(0, len(processed_df) - settings.model.training_window_days - settings.model.test_window_days + 1, settings.model.test_window_days))
        })
        
        # Train model
        results = model.fit(processed_df, channel_grids, progress_callback)
        
        # Store results
        training_runs[run_id].update({
            "status": "completed",
            "results": results,
            "model": model,
            "completion_time": datetime.utcnow(),
        })
        
        # Cache model results
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
                "mape": results.mape
            },
            "diagnostics": results.diagnostics,
            "confidence_intervals": results.confidence_intervals
        }
        await cache_manager.cache_model_results(run_id, model_results)
        
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
        training_runs[run_id].update({
            "status": "failed",
            "error": str(e),
            "completion_time": datetime.utcnow(),
        })
        
        # Broadcast training error
        if 'progress_tracker' in locals():
            await progress_tracker.update_progress("training_error", {
                "error": str(e)
            })


@router.post("/train")
async def train_model(
    upload_id: str,
    background_tasks: BackgroundTasks,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Start model training for uploaded data.
    
    Args:
        upload_id: ID of the uploaded data
        config: Optional training configuration parameters
        
    Returns:
        run_id: Unique identifier for this training run
    """
    if upload_id not in upload_sessions:
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    config = config or {}
    run_id = str(uuid.uuid4())
    
    # Initialize training run
    training_runs[run_id] = {
        "upload_id": upload_id,
        "run_id": run_id,
        "start_time": datetime.utcnow(),
        "status": "queued",
        "config": config,
        "progress": {"type": "queued"}
    }
    
    # Start background training
    background_tasks.add_task(train_model_background, upload_id, run_id, config)
    
    logger.info("Model training queued", run_id=run_id, upload_id=upload_id)
    
    return {"run_id": run_id, "status": "queued"}


@router.get("/training/progress/{run_id}")
async def get_training_progress(run_id: str) -> Dict[str, Any]:
    """Get real-time training progress."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    run = training_runs[run_id]
    
    return {
        "run_id": run_id,
        "status": run["status"],
        "start_time": run["start_time"].isoformat(),
        "last_update": run.get("last_update", run["start_time"]).isoformat(),
        "progress": run.get("progress", {}),
        "error": run.get("error")
    }


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
            "mape": results.mape
        },
        "parameters": {
            "alpha_baseline": results.parameters.alpha_baseline,
            "alpha_trend": results.parameters.alpha_trend,
            "channel_alphas": results.parameters.channel_alphas,
            "channel_betas": results.parameters.channel_betas,
            "channel_rs": results.parameters.channel_rs
        },
        "diagnostics": results.diagnostics,
        "confidence_intervals": results.confidence_intervals
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
    
    # Create cached response curve generator
    curve_generator = create_response_curve_generator(run_id, model_parameters)
    
    # Determine max spend if not provided
    if max_spend is None and current_spend:
        max_spend = current_spend * 3
    elif max_spend is None:
        # Use default
        upload_session = upload_sessions[run["upload_id"]]
        processed_df = upload_session["processed_df"]
        max_spend = processed_df[channel].mean() * 3 * 365  # Annualized
    
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
        "curve_data": curve
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
    
    # Create cached response curve generator
    curve_generator = create_response_curve_generator(run_id, model_parameters)
    
    # Get all response curves (with caching)
    curves = await curve_generator.get_all_response_curves(
        num_points=num_points,
        current_spend=current_spend or {}
    )
    
    return {
        "run_id": run_id,
        "response_curves": curves
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
    
    # Create cached response curve generator
    curve_generator = create_response_curve_generator(run_id, model_parameters)
    
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


@router.delete("/training/{run_id}")
async def delete_training_run(run_id: str) -> Dict[str, str]:
    """Delete a training run and associated data."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    del training_runs[run_id]
    
    logger.info("Training run deleted", run_id=run_id)
    
    return {"status": "deleted", "run_id": run_id}