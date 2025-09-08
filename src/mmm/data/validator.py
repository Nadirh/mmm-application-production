"""
Data validation module for MMM application.
Handles CSV upload validation, data quality scoring, and business tier classification.
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class ValidationErrorCode(Enum):
    ERROR_001 = "Negative spend detected"
    ERROR_002 = "Missing date gaps"
    ERROR_003 = "Day-over-day spend jump >300%"
    ERROR_004 = "Negative profit detected"
    WARNING_001 = "Zero spend across all channels"


class BusinessTier(Enum):
    ENTERPRISE = "enterprise"
    MID_MARKET = "mid_market"
    SMALL_BUSINESS = "small_business"
    PROTOTYPE = "prototype"


@dataclass
class ValidationError:
    code: ValidationErrorCode
    message: str
    column: Optional[str] = None
    row: Optional[int] = None
    severity: str = "error"  # "error" or "warning"


@dataclass
class DataSummary:
    total_days: int
    total_profit: float
    total_annual_spend: float
    channel_count: int
    date_range: Tuple[datetime, datetime]
    business_tier: BusinessTier
    data_quality_score: float


class DataValidator:
    """Validates uploaded CSV data according to MMM requirements."""
    
    def __init__(self):
        self.required_columns = ["date", "profit"]
        self.business_tier_thresholds = {
            BusinessTier.ENTERPRISE: {"days": 365, "annual_spend": 2000000, "min_channel_spend": 25000},
            BusinessTier.MID_MARKET: {"days": 280, "annual_spend": 500000, "min_channel_spend": 15000},
            BusinessTier.SMALL_BUSINESS: {"days": 182, "annual_spend": 200000, "min_channel_spend": 8000},
            BusinessTier.PROTOTYPE: {"days": 182, "annual_spend": 50000, "min_channel_spend": 0}
        }
    
    def validate_upload(self, df: pd.DataFrame) -> Tuple[DataSummary, List[ValidationError]]:
        """
        Validates uploaded dataframe and returns summary and errors.
        
        Args:
            df: Uploaded pandas DataFrame
            
        Returns:
            Tuple of (DataSummary, List[ValidationError])
        """
        errors = []
        
        # Basic structure validation
        errors.extend(self._validate_structure(df))
        
        # Data quality validation
        errors.extend(self._validate_data_quality(df))
        
        # Generate summary
        summary = self._generate_summary(df)
        
        return summary, errors
    
    def _validate_structure(self, df: pd.DataFrame) -> List[ValidationError]:
        """Validates basic DataFrame structure."""
        errors = []
        
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            errors.append(ValidationError(
                code=ValidationErrorCode.ERROR_002,
                message=f"Missing required columns: {missing_cols}"
            ))
        
        return errors
    
    def _validate_data_quality(self, df: pd.DataFrame) -> List[ValidationError]:
        """Validates data quality according to business rules."""
        errors = []
        
        if "profit" in df.columns:
            # Check for negative profit
            negative_profit_rows = df[df["profit"] < 0].index.tolist()
            if negative_profit_rows:
                errors.append(ValidationError(
                    code=ValidationErrorCode.ERROR_004,
                    message=f"Negative profit detected in {len(negative_profit_rows)} rows",
                    column="profit"
                ))
        
        # Check spend columns for negative values and jumps
        spend_columns = [col for col in df.columns if col not in ["date", "profit", "is_holiday", "promo_flag", "site_outage"]]
        
        for col in spend_columns:
            if col in df.columns:
                # Check for negative spend
                negative_spend_rows = df[df[col] < 0].index.tolist()
                if negative_spend_rows:
                    errors.append(ValidationError(
                        code=ValidationErrorCode.ERROR_001,
                        message=f"Negative spend detected in column {col}",
                        column=col
                    ))
                
                # Check for spend jumps >300%
                df_sorted = df.sort_values("date")
                daily_changes = df_sorted[col].pct_change()
                large_jumps = daily_changes[daily_changes > 3.0].index.tolist()
                if large_jumps:
                    errors.append(ValidationError(
                        code=ValidationErrorCode.ERROR_003,
                        message=f"Day-over-day spend jump >300% detected in {col}",
                        column=col
                    ))
        
        return errors
    
    def _generate_summary(self, df: pd.DataFrame) -> DataSummary:
        """Generates data summary including business tier classification."""
        spend_columns = [col for col in df.columns if col not in ["date", "profit", "is_holiday", "promo_flag", "site_outage"]]
        
        total_days = len(df)
        total_profit = df["profit"].sum() if "profit" in df.columns else 0
        total_annual_spend = df[spend_columns].sum().sum() if spend_columns else 0
        channel_count = len(spend_columns)
        
        if "date" in df.columns:
            date_col = pd.to_datetime(df["date"])
            date_range = (date_col.min(), date_col.max())
        else:
            date_range = (datetime.now(), datetime.now())
        
        business_tier = self._classify_business_tier(total_days, total_annual_spend, df, spend_columns)
        data_quality_score = self._calculate_quality_score(df)
        
        return DataSummary(
            total_days=total_days,
            total_profit=total_profit,
            total_annual_spend=total_annual_spend,
            channel_count=channel_count,
            date_range=date_range,
            business_tier=business_tier,
            data_quality_score=data_quality_score
        )
    
    def _classify_business_tier(self, total_days: int, total_annual_spend: float, 
                               df: pd.DataFrame, spend_columns: List[str]) -> BusinessTier:
        """Classifies business tier based on data characteristics."""
        # Check each tier from highest to lowest
        for tier, thresholds in self.business_tier_thresholds.items():
            if tier == BusinessTier.PROTOTYPE:
                continue  # Handle prototype last
                
            meets_days = total_days >= thresholds["days"]
            meets_spend = total_annual_spend >= thresholds["annual_spend"]
            
            # Check channel spend requirements
            meets_channel_spend = True
            if spend_columns and thresholds["min_channel_spend"] > 0:
                for col in spend_columns:
                    if col in df.columns and df[col].sum() < thresholds["min_channel_spend"]:
                        meets_channel_spend = False
                        break
            
            if meets_days and meets_spend and meets_channel_spend:
                return tier
        
        return BusinessTier.PROTOTYPE
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculates data quality score (0-100)."""
        score = 100.0
        
        # Deduct for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score -= missing_pct * 50
        
        # Deduct for negative values in spend columns
        spend_columns = [col for col in df.columns if col not in ["date", "profit", "is_holiday", "promo_flag", "site_outage"]]
        for col in spend_columns:
            if col in df.columns:
                negative_pct = (df[col] < 0).sum() / len(df)
                score -= negative_pct * 30
        
        return max(0.0, score)