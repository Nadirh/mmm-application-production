"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, UTC
from enum import Enum


class BusinessTierEnum(str, Enum):
    ENTERPRISE = "enterprise"
    MID_MARKET = "mid_market"
    SMALL_BUSINESS = "small_business"
    PROTOTYPE = "prototype"


class ChannelTypeEnum(str, Enum):
    SEARCH_BRAND = "search_brand"
    SEARCH_NON_BRAND = "search_non_brand"
    SOCIAL = "social"
    TV_VIDEO = "tv_video"
    DISPLAY = "display"
    UNKNOWN = "unknown"


class ValidationErrorSchema(BaseModel):
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    column: Optional[str] = Field(None, description="Column name if applicable")
    row: Optional[int] = Field(None, description="Row number if applicable")
    severity: str = Field(..., description="Error severity: error or warning")


class DataSummarySchema(BaseModel):
    total_days: int = Field(..., description="Total number of days in dataset")
    total_profit: float = Field(..., description="Total profit across all days")
    total_annual_spend: float = Field(..., description="Total annual marketing spend")
    channel_count: int = Field(..., description="Number of marketing channels")
    date_range: Dict[str, str] = Field(..., description="Date range with start and end")
    business_tier: BusinessTierEnum = Field(..., description="Business tier classification")
    data_quality_score: float = Field(..., description="Data quality score (0-100)")


class ChannelInfoSchema(BaseModel):
    type: ChannelTypeEnum = Field(..., description="Channel type classification")
    total_spend: float = Field(..., description="Total spend for this channel")
    spend_share: float = Field(..., description="Share of total spend (0-1)")
    days_active: int = Field(..., description="Number of days with non-zero spend")


class UploadResponseSchema(BaseModel):
    upload_id: str = Field(..., description="Unique upload identifier")
    data_summary: DataSummarySchema
    validation_errors: List[ValidationErrorSchema]
    channel_info: Dict[str, ChannelInfoSchema]


class TrainingConfigSchema(BaseModel):
    training_window_days: Optional[int] = Field(126, description="Training window size in days")
    test_window_days: Optional[int] = Field(14, description="Test window size in days")
    n_bootstrap: Optional[int] = Field(1000, description="Number of bootstrap samples")
    
    @validator('training_window_days')
    def validate_training_window(cls, v):
        if v < 90:
            raise ValueError('Training window must be at least 90 days')
        return v
    
    @validator('test_window_days')
    def validate_test_window(cls, v):
        if v < 7:
            raise ValueError('Test window must be at least 7 days')
        return v


class TrainingRequestSchema(BaseModel):
    upload_id: str = Field(..., description="Upload ID for the dataset")
    config: Optional[TrainingConfigSchema] = None


class TrainingProgressSchema(BaseModel):
    type: str = Field(..., description="Progress event type")
    fold: Optional[int] = Field(None, description="Current CV fold")
    total_folds: Optional[int] = Field(None, description="Total CV folds")
    mape: Optional[float] = Field(None, description="Current MAPE")
    error: Optional[str] = Field(None, description="Error message if failed")


class TrainingStatusSchema(BaseModel):
    run_id: str = Field(..., description="Training run identifier")
    status: str = Field(..., description="Training status")
    start_time: str = Field(..., description="Training start time")
    last_update: str = Field(..., description="Last update time")
    progress: TrainingProgressSchema
    error: Optional[str] = Field(None, description="Error message if failed")


class ModelParametersSchema(BaseModel):
    alpha_baseline: float = Field(..., description="Baseline intercept parameter")
    alpha_trend: float = Field(..., description="Trend parameter")
    channel_alphas: Dict[str, float] = Field(..., description="Channel incremental strength")
    channel_betas: Dict[str, float] = Field(..., description="Channel saturation parameters")
    channel_rs: Dict[str, float] = Field(..., description="Channel adstock parameters")


class ModelPerformanceSchema(BaseModel):
    cv_mape: float = Field(..., description="Cross-validation MAPE")
    r_squared: float = Field(..., description="R-squared value")
    mape: float = Field(..., description="Full dataset MAPE")


class ModelResultsSchema(BaseModel):
    run_id: str = Field(..., description="Training run identifier")
    training_info: Dict[str, Any] = Field(..., description="Training metadata")
    model_performance: ModelPerformanceSchema
    parameters: ModelParametersSchema
    diagnostics: Dict[str, Any] = Field(..., description="Model diagnostics")
    confidence_intervals: Dict[str, List[float]] = Field(..., description="Parameter confidence intervals")


class ChannelPerformanceSchema(BaseModel):
    attribution: float = Field(..., description="Total attribution for channel")
    attribution_share: float = Field(..., description="Attribution share percentage")
    incremental_strength: float = Field(..., description="Alpha parameter")
    saturation_parameter: float = Field(..., description="Beta parameter")
    adstock_parameter: float = Field(..., description="R parameter")
    confidence_interval: Dict[str, float] = Field(..., description="Confidence interval bounds")


class ResponseCurveSchema(BaseModel):
    spend_range: List[float] = Field(..., description="Spend values")
    profit_values: List[float] = Field(..., description="Corresponding profit values")
    marginal_efficiency: Optional[List[float]] = Field(None, description="Marginal efficiency curve")


class ConstraintSchema(BaseModel):
    channel: str = Field(..., description="Channel name")
    type: str = Field(..., description="Constraint type: floor, cap, lock, ramp")
    value: float = Field(..., description="Constraint value")
    description: Optional[str] = Field("", description="Constraint description")
    
    @validator('type')
    def validate_constraint_type(cls, v):
        allowed_types = ['floor', 'cap', 'lock', 'ramp']
        if v not in allowed_types:
            raise ValueError(f'Constraint type must be one of: {allowed_types}')
        return v


class OptimizationRequestSchema(BaseModel):
    run_id: str = Field(..., description="Training run ID")
    total_budget: float = Field(..., description="Total budget for optimization")
    current_spend: Dict[str, float] = Field(..., description="Current spend by channel")
    constraints: Optional[List[ConstraintSchema]] = Field([], description="Business constraints")
    optimization_window_days: Optional[int] = Field(365, description="Optimization time horizon")
    
    @validator('total_budget')
    def validate_total_budget(cls, v):
        if v <= 0:
            raise ValueError('Total budget must be positive')
        return v
    
    @validator('current_spend')
    def validate_current_spend(cls, v):
        if any(spend < 0 for spend in v.values()):
            raise ValueError('All spend values must be non-negative')
        return v


class OptimizationResultSchema(BaseModel):
    optimal_spend: Dict[str, float] = Field(..., description="Optimal spend allocation")
    optimal_profit: float = Field(..., description="Projected profit at optimal allocation")
    current_profit: float = Field(..., description="Current profit level")
    profit_uplift: float = Field(..., description="Absolute profit uplift")
    profit_uplift_pct: float = Field(..., description="Percentage profit uplift")
    shadow_prices: Dict[str, float] = Field(..., description="Shadow prices for constraints")
    constraints_binding: List[str] = Field(..., description="List of binding constraints")


class OptimizationResponseSchema(BaseModel):
    run_id: str = Field(..., description="Training run ID")
    optimization_results: OptimizationResultSchema
    response_curves: Dict[str, ResponseCurveSchema] = Field(..., description="Response curves by channel")
    scenario_analysis: Dict[str, Any] = Field(..., description="Scenario analysis results")


class ScenarioConfigSchema(BaseModel):
    current_spend: Dict[str, float] = Field(..., description="Spend allocation for scenario")
    total_budget: float = Field(..., description="Total budget for scenario")
    optimization_window_days: Optional[int] = Field(365, description="Time horizon")


class ScenarioResultSchema(BaseModel):
    spend: Dict[str, float] = Field(..., description="Spend allocation")
    total_budget: float = Field(..., description="Total budget")
    projected_profit: float = Field(..., description="Projected profit")


class HealthCheckSchema(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Timestamp")
    environment: str = Field(..., description="Environment name")
    version: str = Field(..., description="Application version")


class DetailedHealthCheckSchema(HealthCheckSchema):
    system: Dict[str, Any] = Field(..., description="System information")
    configuration: Dict[str, Any] = Field(..., description="Configuration summary")


class ErrorResponseSchema(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    traceback: Optional[str] = Field(None, description="Stack trace (dev only)")


# WebSocket message schemas
class WebSocketMessageSchema(BaseModel):
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class TrainingProgressMessageSchema(WebSocketMessageSchema):
    type: str = Field("training_progress", const=True)


class TrainingCompleteMessageSchema(WebSocketMessageSchema):
    type: str = Field("training_complete", const=True)


class TrainingErrorMessageSchema(WebSocketMessageSchema):
    type: str = Field("training_error", const=True)