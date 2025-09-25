"""
SQLAlchemy database models for MMM application.
"""
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import relationship
import uuid

from mmm.database.connection import Base


class UploadSession(Base):
    """Model for data upload sessions."""
    __tablename__ = "upload_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Multi-tenant fields (nullable for backward compatibility)
    client_id = Column(String, nullable=True, default="default", index=True)
    organization_id = Column(String, nullable=True, default="default", index=True)

    filename = Column(String, nullable=False)
    upload_time = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(UTC))
    file_path = Column(String, nullable=False)

    # Data summary
    total_days = Column(Integer)
    total_profit = Column(Float)
    total_annual_spend = Column(Float)
    channel_count = Column(Integer)
    date_range_start = Column(TIMESTAMP(timezone=True))
    date_range_end = Column(TIMESTAMP(timezone=True))
    business_tier = Column(String)
    data_quality_score = Column(Float)
    
    # Validation results
    validation_errors = Column(JSON)  # Store validation errors as JSON
    channel_info = Column(JSON)       # Store channel information as JSON
    
    # Status
    status = Column(String, default="validated")
    
    # Relationships
    training_runs = relationship("TrainingRun", back_populates="upload_session")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "upload_id": self.id,
            "filename": self.filename,
            "upload_time": self.upload_time.isoformat() if self.upload_time else None,
            "status": self.status,
            "data_summary": {
                "total_days": self.total_days,
                "total_profit": self.total_profit,
                "total_annual_spend": self.total_annual_spend,
                "channel_count": self.channel_count,
                "date_range": {
                    "start": self.date_range_start.isoformat() if self.date_range_start else None,
                    "end": self.date_range_end.isoformat() if self.date_range_end else None
                },
                "business_tier": self.business_tier,
                "data_quality_score": self.data_quality_score
            },
            "validation_errors": self.validation_errors or [],
            "channel_info": self.channel_info or {}
        }


class TrainingRun(Base):
    """Model for model training runs."""
    __tablename__ = "training_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    upload_session_id = Column(String, ForeignKey("upload_sessions.id"), nullable=False)

    # Multi-tenant fields (nullable for backward compatibility)
    client_id = Column(String, nullable=True, default="default", index=True)
    organization_id = Column(String, nullable=True, default="default", index=True)

    # Training metadata
    start_time = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(UTC))
    completion_time = Column(TIMESTAMP(timezone=True))
    status = Column(String, default="queued")  # queued, training, completed, failed
    
    # Training configuration
    training_config = Column(JSON)
    
    # Model results (stored as JSON)
    model_parameters = Column(JSON)
    model_performance = Column(JSON)
    diagnostics = Column(JSON)
    confidence_intervals = Column(JSON)
    
    # Progress tracking
    current_progress = Column(JSON)
    error_message = Column(Text)
    
    # Relationships
    upload_session = relationship("UploadSession", back_populates="training_runs")
    optimization_runs = relationship("OptimizationRun", back_populates="training_run")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "run_id": self.id,
            "upload_id": self.upload_session_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "completion_time": self.completion_time.isoformat() if self.completion_time else None,
            "status": self.status,
            "config": self.training_config or {},
            "model_parameters": self.model_parameters,
            "model_performance": self.model_performance,
            "diagnostics": self.diagnostics,
            "confidence_intervals": self.confidence_intervals,
            "progress": self.current_progress or {},
            "error": self.error_message
        }


class OptimizationRun(Base):
    """Model for budget optimization runs."""
    __tablename__ = "optimization_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    training_run_id = Column(String, ForeignKey("training_runs.id"), nullable=False)

    # Multi-tenant fields (nullable for backward compatibility)
    client_id = Column(String, nullable=True, default="default", index=True)
    organization_id = Column(String, nullable=True, default="default", index=True)

    # Optimization metadata
    created_time = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(UTC))
    
    # Optimization inputs
    total_budget = Column(Float, nullable=False)
    current_spend = Column(JSON, nullable=False)  # Channel spend allocation
    constraints = Column(JSON)  # Business constraints
    optimization_window_days = Column(Integer, default=365)
    
    # Optimization results
    optimal_spend = Column(JSON)
    optimal_profit = Column(Float)
    current_profit = Column(Float)
    profit_uplift = Column(Float)
    shadow_prices = Column(JSON)
    constraints_binding = Column(JSON)
    scenario_analysis = Column(JSON)
    
    # Relationships
    training_run = relationship("TrainingRun", back_populates="optimization_runs")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "optimization_id": self.id,
            "run_id": self.training_run_id,
            "created_time": self.created_time.isoformat() if self.created_time else None,
            "inputs": {
                "total_budget": self.total_budget,
                "current_spend": self.current_spend or {},
                "constraints": self.constraints or [],
                "optimization_window_days": self.optimization_window_days
            },
            "results": {
                "optimal_spend": self.optimal_spend or {},
                "optimal_profit": self.optimal_profit,
                "current_profit": self.current_profit,
                "profit_uplift": self.profit_uplift,
                "shadow_prices": self.shadow_prices or {},
                "constraints_binding": self.constraints_binding or [],
                "scenario_analysis": self.scenario_analysis or {}
            }
        }


class DataPoint(Base):
    """Model for storing processed daily data points."""
    __tablename__ = "data_points"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    upload_session_id = Column(String, ForeignKey("upload_sessions.id"), nullable=False)
    
    # Time series data
    date = Column(TIMESTAMP(timezone=True), nullable=False)
    profit = Column(Float, nullable=False)
    
    # Channel spend data (stored as JSON for flexibility)
    channel_spend = Column(JSON, nullable=False)
    
    # Control variables
    is_holiday = Column(Boolean, default=False)
    promo_flag = Column(Boolean, default=False)
    site_outage = Column(Boolean, default=False)
    
    # Derived features
    days_since_start = Column(Integer)
    day_of_week = Column(Integer)
    is_weekend = Column(Boolean, default=False)
    month = Column(Integer)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat() if self.date else None,
            "profit": self.profit,
            "channel_spend": self.channel_spend or {},
            "is_holiday": self.is_holiday,
            "promo_flag": self.promo_flag,
            "site_outage": self.site_outage,
            "days_since_start": self.days_since_start,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "month": self.month
        }


class CachedResponseCurve(Base):
    """Model for caching response curves."""
    __tablename__ = "cached_response_curves"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    training_run_id = Column(String, ForeignKey("training_runs.id"), nullable=False)
    channel_name = Column(String, nullable=False)
    
    # Cache metadata
    created_time = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(UTC))
    cache_key = Column(String, nullable=False, unique=True)
    
    # Response curve data
    spend_range = Column(JSON, nullable=False)
    profit_values = Column(JSON, nullable=False)
    marginal_efficiency = Column(JSON)
    
    # Parameters used for generation
    max_spend = Column(Float)
    resolution = Column(Integer)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel": self.channel_name,
            "response_curve": {
                "spend_range": self.spend_range or [],
                "profit_values": self.profit_values or [],
                "marginal_efficiency": self.marginal_efficiency or []
            },
            "metadata": {
                "max_spend": self.max_spend,
                "resolution": self.resolution,
                "created_time": self.created_time.isoformat() if self.created_time else None
            }
        }