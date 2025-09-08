"""
Data upload and validation endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import pandas as pd
import uuid
import os
from datetime import datetime
import structlog

from mmm.data.validator import DataValidator, ValidationError
from mmm.data.processor import DataProcessor
from mmm.config.settings import settings

router = APIRouter()
logger = structlog.get_logger()

# In-memory storage for demo purposes
# In production, use proper database or session management
upload_sessions: Dict[str, Dict[str, Any]] = {}


@router.post("/upload")
async def upload_data(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload and validate CSV data file.
    
    Returns:
        - upload_id: Unique identifier for this upload
        - data_summary: Summary of uploaded data
        - validation_errors: List of validation errors/warnings
    """
    logger.info("Data upload started", filename=file.filename)
    
    # Validate file
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Generate upload session ID
    upload_id = str(uuid.uuid4())
    
    try:
        # Read CSV file
        content = await file.read()
        if len(content) > settings.api.max_upload_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.api.max_upload_size // (1024*1024)}MB"
            )
        
        # Save file temporarily
        upload_path = os.path.join(settings.api.upload_dir, f"{upload_id}.csv")
        with open(upload_path, "wb") as f:
            f.write(content)
        
        # Parse CSV
        df = pd.read_csv(upload_path)
        
        # Validate data
        validator = DataValidator()
        data_summary, validation_errors = validator.validate_upload(df)
        
        # Process data if validation passes
        processor = DataProcessor()
        processed_df, channel_info = processor.process_data(df)
        
        # Store session data
        upload_sessions[upload_id] = {
            "filename": file.filename,
            "upload_time": datetime.utcnow(),
            "file_path": upload_path,
            "data_summary": data_summary,
            "validation_errors": validation_errors,
            "channel_info": channel_info,
            "processed_df": processed_df,
            "status": "validated"
        }
        
        logger.info(
            "Data upload completed",
            upload_id=upload_id,
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
            business_tier=data_summary.business_tier.value
        )
        
        return {
            "upload_id": upload_id,
            "data_summary": {
                "total_days": data_summary.total_days,
                "total_profit": data_summary.total_profit,
                "total_annual_spend": data_summary.total_annual_spend,
                "channel_count": data_summary.channel_count,
                "date_range": {
                    "start": data_summary.date_range[0].isoformat(),
                    "end": data_summary.date_range[1].isoformat()
                },
                "business_tier": data_summary.business_tier.value,
                "data_quality_score": data_summary.data_quality_score
            },
            "validation_errors": [
                {
                    "code": error.code.name,
                    "message": error.message,
                    "column": error.column,
                    "row": error.row,
                    "severity": error.severity
                }
                for error in validation_errors
            ],
            "channel_info": {
                name: {
                    "type": info.type.value,
                    "total_spend": info.total_spend,
                    "spend_share": info.spend_share,
                    "days_active": info.days_active
                }
                for name, info in channel_info.items()
            }
        }
        
    except Exception as e:
        logger.error("Data upload failed", upload_id=upload_id, error=str(e))
        
        # Clean up file if it exists
        upload_path = os.path.join(settings.api.upload_dir, f"{upload_id}.csv")
        if os.path.exists(upload_path):
            os.remove(upload_path)
            
        raise HTTPException(status_code=400, detail=f"Data upload failed: {str(e)}")


@router.get("/upload/{upload_id}/summary")
async def get_upload_summary(upload_id: str) -> Dict[str, Any]:
    """Get summary information for an uploaded dataset."""
    if upload_id not in upload_sessions:
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    session = upload_sessions[upload_id]
    data_summary = session["data_summary"]
    
    return {
        "upload_id": upload_id,
        "filename": session["filename"],
        "upload_time": session["upload_time"].isoformat(),
        "status": session["status"],
        "data_summary": {
            "total_days": data_summary.total_days,
            "total_profit": data_summary.total_profit,
            "total_annual_spend": data_summary.total_annual_spend,
            "channel_count": data_summary.channel_count,
            "business_tier": data_summary.business_tier.value,
            "data_quality_score": data_summary.data_quality_score
        }
    }


@router.get("/upload/{upload_id}/channels")
async def get_channel_info(upload_id: str) -> Dict[str, Any]:
    """Get detailed channel information for an uploaded dataset."""
    if upload_id not in upload_sessions:
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    session = upload_sessions[upload_id]
    channel_info = session["channel_info"]
    
    return {
        "upload_id": upload_id,
        "channels": {
            name: {
                "type": info.type.value,
                "total_spend": info.total_spend,
                "spend_share": info.spend_share,
                "days_active": info.days_active
            }
            for name, info in channel_info.items()
        }
    }


@router.get("/upload/{upload_id}/validation")
async def get_validation_results(upload_id: str) -> Dict[str, Any]:
    """Get validation results for an uploaded dataset."""
    if upload_id not in upload_sessions:
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    session = upload_sessions[upload_id]
    validation_errors = session["validation_errors"]
    
    return {
        "upload_id": upload_id,
        "validation_errors": [
            {
                "code": error.code.name,
                "message": error.message,
                "column": error.column,
                "row": error.row,
                "severity": error.severity
            }
            for error in validation_errors
        ]
    }


@router.delete("/upload/{upload_id}")
async def delete_upload(upload_id: str) -> Dict[str, str]:
    """Delete an upload session and associated files."""
    if upload_id not in upload_sessions:
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    session = upload_sessions[upload_id]
    
    # Delete file if it exists
    file_path = session.get("file_path")
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
    
    # Remove from sessions
    del upload_sessions[upload_id]
    
    logger.info("Upload session deleted", upload_id=upload_id)
    
    return {"status": "deleted", "upload_id": upload_id}