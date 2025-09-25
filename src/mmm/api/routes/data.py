"""
Data upload and validation endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import pandas as pd
import uuid
import os
from datetime import datetime, UTC
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from mmm.data.validator import DataValidator, ValidationError
from mmm.data.processor import DataProcessor
from mmm.config.settings import settings
from mmm.database.connection import get_db
from mmm.database.models import UploadSession

router = APIRouter()
logger = structlog.get_logger()

# Database storage for upload sessions
# upload_sessions in-memory cache for backward compatibility with model training
upload_sessions: Dict[str, Dict[str, Any]] = {}


async def get_upload_session_from_db(upload_id: str, db: AsyncSession, include_processed_data: bool = True) -> Dict[str, Any]:
    """Retrieve upload session from database and populate cache."""
    from sqlalchemy import select
    
    # Check cache first
    if upload_id in upload_sessions:
        return upload_sessions[upload_id]
    
    # Retrieve from database
    result = await db.execute(select(UploadSession).where(UploadSession.id == upload_id))
    db_session = result.scalar_one_or_none()
    
    if db_session is None:
        return None
    
    # Only process CSV data if needed (expensive operation)
    try:
        processed_df = None
        channel_info = {}
        
        if include_processed_data:
            # Load the CSV data and recreate processed_df
            processed_df = pd.read_csv(db_session.file_path)
            processor = DataProcessor()
            processed_df, channel_info_obj = processor.process_data(processed_df)
            
            # Convert channel info objects to dict format expected by training
            for name, info in channel_info_obj.items():
                channel_info[name] = info
        else:
            # Use stored channel info from database
            channel_info = db_session.channel_info or {}
            
        # Reconstruct data summary
        from mmm.data.validator import DataSummary, BusinessTier
        data_summary = DataSummary(
            total_days=db_session.total_days,
            total_profit=db_session.total_profit,
            total_annual_spend=db_session.total_annual_spend,
            channel_count=db_session.channel_count,
            date_range=(db_session.date_range_start, db_session.date_range_end),
            business_tier=BusinessTier(db_session.business_tier),
            data_quality_score=db_session.data_quality_score
        )
        
        # Create session data
        session_data = {
            "filename": db_session.filename,
            "upload_time": db_session.upload_time,
            "file_path": db_session.file_path,
            "data_summary": data_summary,
            "validation_errors": [
                type('ValidationError', (), {
                    'code': type('Code', (), {'name': err['code']})(),
                    'message': err['message'],
                    'column': err.get('column'),
                    'row': err.get('row'),
                    'severity': err['severity']
                })() for err in (db_session.validation_errors or [])
            ],
            "channel_info": channel_info,
            "status": db_session.status
        }
        
        # Only add processed_df if it was loaded
        if processed_df is not None:
            session_data["processed_df"] = processed_df
            # Populate cache only when we have full data
            upload_sessions[upload_id] = session_data
        
        return session_data
        
    except Exception as e:
        logger.error("Failed to reconstruct upload session", upload_id=upload_id, error=str(e))
        return None


@router.post("/upload")
async def upload_data(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
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

        # Calculate profit-channel correlations
        profit_channel_correlations = {}
        if "profit" in processed_df.columns:
            for channel_name in channel_info.keys():
                if channel_name in processed_df.columns:
                    correlation = processed_df["profit"].corr(processed_df[channel_name])
                    profit_channel_correlations[channel_name] = float(correlation) if not pd.isna(correlation) else 0.0

        # Store session data
        upload_sessions[upload_id] = {
            "filename": file.filename,
            "upload_time": datetime.now(UTC),
            "file_path": upload_path,
            "data_summary": data_summary,
            "validation_errors": validation_errors,
            "channel_info": channel_info,
            "processed_df": processed_df,
            "profit_channel_correlations": profit_channel_correlations,
            "status": "validated"
        }
        
        # Save to database with default client_id for now (Chunk 1)
        db_upload_session = UploadSession(
            id=upload_id,
            client_id="default",  # Default client for backward compatibility
            organization_id="default",  # Default organization for backward compatibility
            filename=file.filename,
            file_path=upload_path,
            total_days=data_summary.total_days,
            total_profit=data_summary.total_profit,
            total_annual_spend=data_summary.total_annual_spend,
            channel_count=data_summary.channel_count,
            date_range_start=data_summary.date_range[0],
            date_range_end=data_summary.date_range[1],
            business_tier=data_summary.business_tier.value,
            data_quality_score=data_summary.data_quality_score,
            validation_errors=[{
                "code": error.code.name,
                "message": error.message,
                "column": error.column,
                "row": error.row,
                "severity": error.severity
            } for error in validation_errors],
            channel_info={
                name: {
                    "type": info.type.value,
                    "total_spend": info.total_spend,
                    "spend_share": info.spend_share,
                    "days_active": info.days_active
                }
                for name, info in channel_info.items()
            },
            status="validated"
        )
        db.add(db_upload_session)
        await db.commit()
        await db.refresh(db_upload_session)
        
        logger.info(
            "Data upload completed",
            upload_id=upload_id,
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
            business_tier=data_summary.business_tier.value
        )
        
        return {
            "status": "success",
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
            },
            "profit_channel_correlations": profit_channel_correlations
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 413 file too large) as-is
        raise
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