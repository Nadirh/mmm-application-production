"""
Data validation module for MMM application.
Handles CSV upload validation, data quality scoring, and business tier classification.
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, UTC


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
        
        # Convert numeric columns to proper data types first
        df = self._convert_numeric_columns(df)
        
        if "profit" in df.columns:
            # Check for negative profit (only if numeric)
            try:
                profit_numeric = pd.to_numeric(df["profit"], errors='coerce')
                if not profit_numeric.isna().all():
                    negative_profit_rows = df[profit_numeric < 0].index.tolist()
                    if negative_profit_rows:
                        errors.append(ValidationError(
                            code=ValidationErrorCode.ERROR_004,
                            message=f"Negative profit detected in {len(negative_profit_rows)} rows",
                            column="profit"
                        ))
            except Exception as e:
                errors.append(ValidationError(
                    code=ValidationErrorCode.WARNING_001,
                    message=f"Could not validate profit column: non-numeric data detected",
                    column="profit",
                    severity="warning"
                ))
        
        # Check spend columns for negative values and jumps
        spend_columns = [col for col in df.columns if col not in ["date", "profit", "is_holiday", "promo_flag", "site_outage"]]
        
        # Check for zero spend across all channels
        if spend_columns:
            all_zero_spend = all(df[col].sum() == 0 for col in spend_columns if col in df.columns)
            if all_zero_spend:
                errors.append(ValidationError(
                    code=ValidationErrorCode.WARNING_001,
                    message="Zero spend across all channels",
                    severity="warning"
                ))
        
        for col in spend_columns:
            if col in df.columns:
                try:
                    # Check for negative spend (convert to numeric first)
                    col_numeric = pd.to_numeric(df[col], errors='coerce')
                    if not col_numeric.isna().all():
                        negative_spend_rows = df[col_numeric < 0].index.tolist()
                        if negative_spend_rows:
                            errors.append(ValidationError(
                                code=ValidationErrorCode.ERROR_001,
                                message=f"Negative spend detected in column {col}",
                                column=col
                            ))
                except Exception:
                    # Skip validation for non-numeric columns
                    continue
                
                # Check for spend jumps >300% (only if date column exists)
                if "date" in df.columns:
                    try:
                        col_numeric = pd.to_numeric(df[col], errors='coerce')
                        if not col_numeric.isna().all():
                            df_temp = df.copy()
                            df_temp[col] = col_numeric
                            df_sorted = df_temp.sort_values("date")
                            daily_changes = df_sorted[col].pct_change()
                            large_jumps = daily_changes[daily_changes > 3.0].index.tolist()
                            if large_jumps:
                                errors.append(ValidationError(
                                    code=ValidationErrorCode.ERROR_003,
                                    message=f"Day-over-day spend jump >300% detected in {col}",
                                    column=col
                                ))
                    except Exception:
                        # Skip jump validation for non-numeric columns
                        continue
        
        return errors
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns that should be numeric to proper data types."""
        df_copy = df.copy()
        
        # Convert profit column if it exists
        if "profit" in df_copy.columns:
            df_copy["profit"] = pd.to_numeric(df_copy["profit"], errors='coerce')
        
        # Convert spend columns (exclude date and categorical columns)
        exclude_cols = ["date", "profit", "is_holiday", "promo_flag", "site_outage"]
        spend_columns = [col for col in df_copy.columns if col not in exclude_cols]
        
        for col in spend_columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        return df_copy
    
    def _generate_summary(self, df: pd.DataFrame) -> DataSummary:
        """Generates data summary including business tier classification."""
        spend_columns = [col for col in df.columns if col not in ["date", "profit", "is_holiday", "promo_flag", "site_outage"]]
        
        total_days = int(len(df))
        total_profit = float(df["profit"].sum()) if "profit" in df.columns else 0.0
        total_annual_spend = float(df[spend_columns].sum().sum()) if spend_columns else 0.0
        channel_count = int(len(spend_columns))
        
        if "date" in df.columns:
            try:
                date_col = pd.to_datetime(df["date"], errors='coerce')
                # Localize timezone-naive timestamps to UTC
                if date_col.dt.tz is None:
                    date_col = date_col.dt.tz_localize(UTC)
                # Convert to timezone-aware datetime objects
                date_min = date_col.min().to_pydatetime()
                date_max = date_col.max().to_pydatetime()
                date_range = (date_min, date_max)
            except Exception as e:
                # Fallback for malformed dates
                print(f"Date parsing error: {e}")  # Debug print
                date_range = (datetime.now(UTC), datetime.now(UTC))
        else:
            date_range = (datetime.now(UTC), datetime.now(UTC))
        
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
        # Handle empty dataframe
        if df.empty or len(df.columns) == 0:
            return 0.0
        
        score = 100.0
        
        # Deduct for missing values
        missing_pct = float(df.isnull().sum().sum()) / (len(df) * len(df.columns))
        score -= missing_pct * 50
        
        # Deduct for negative values in spend columns
        spend_columns = [col for col in df.columns if col not in ["date", "profit", "is_holiday", "promo_flag", "site_outage"]]
        for col in spend_columns:
            if col in df.columns:
                negative_pct = float((df[col] < 0).sum()) / len(df)
                score -= negative_pct * 30
        
        return float(max(0.0, score))