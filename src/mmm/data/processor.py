"""
Data processing module for MMM application.
Handles data preprocessing, channel classification, and feature engineering.
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass


class ChannelType(Enum):
    SEARCH_BRAND = "search_brand"
    SEARCH_NON_BRAND = "search_non_brand"
    SOCIAL = "social"
    TV_VIDEO = "tv_video"
    DISPLAY = "display"
    UNKNOWN = "unknown"


@dataclass
class ChannelInfo:
    name: str
    type: ChannelType
    total_spend: float
    spend_share: float
    days_active: int


class DataProcessor:
    """Processes and prepares data for MMM training."""
    
    def __init__(self):
        self.channel_keywords = {
            ChannelType.SEARCH_BRAND: ["brand", "branded", "trademark"],
            ChannelType.SEARCH_NON_BRAND: ["search", "google", "bing", "sem", "ppc"],
            ChannelType.SOCIAL: ["facebook", "instagram", "twitter", "linkedin", "tiktok", "social"],
            ChannelType.TV_VIDEO: ["tv", "video", "youtube", "connected_tv", "ctv", "ott"],
            ChannelType.DISPLAY: ["display", "banner", "programmatic", "native"]
        }
    
    def process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, ChannelInfo]]:
        """
        Processes raw data for MMM training.
        
        Args:
            df: Raw DataFrame with date, profit, and channel columns
            
        Returns:
            Tuple of (processed_df, channel_info_dict)
        """
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Ensure date column is datetime
        processed_df["date"] = pd.to_datetime(processed_df["date"])
        
        # Sort by date
        processed_df = processed_df.sort_values("date").reset_index(drop=True)
        
        # Add time features
        processed_df = self._add_time_features(processed_df)
        
        # Get channel information
        channel_info = self._analyze_channels(processed_df)
        
        # Fill missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Validate final data
        processed_df = self._validate_processed_data(processed_df)
        
        return processed_df, channel_info
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds time-based features for modeling."""
        df = df.copy()
        
        # Add days since start (for trend modeling)
        min_date = df["date"].min()
        df["days_since_start"] = (df["date"] - min_date).dt.days
        
        # Add day of week
        df["day_of_week"] = df["date"].dt.dayofweek
        
        # Add is_weekend flag
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        
        # Add month
        df["month"] = df["date"].dt.month
        
        return df
    
    def _analyze_channels(self, df: pd.DataFrame) -> Dict[str, ChannelInfo]:
        """Analyzes channel characteristics and classifies channel types."""
        channel_info = {}
        
        # Get spend columns (excluding date, profit, and control variables, only numeric columns)
        exclude_cols = ["date", "profit", "is_holiday", "promo_flag", "site_outage", 
                       "days_since_start", "day_of_week", "is_weekend", "month"]
        spend_columns = [col for col in df.columns 
                        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        # Skip processing if no spend columns found
        if not spend_columns:
            return {}
        
        total_spend = df[spend_columns].sum().sum()
        
        for channel in spend_columns:
            channel_spend = float(df[channel].sum())
            channel_type = self._classify_channel_type(channel)
            days_active = int((df[channel] > 0).sum())
            spend_share = float(channel_spend / total_spend) if total_spend > 0 else 0.0
            
            channel_info[channel] = ChannelInfo(
                name=channel,
                type=channel_type,
                total_spend=channel_spend,
                spend_share=spend_share,
                days_active=days_active
            )
        
        return channel_info
    
    def _classify_channel_type(self, channel_name: str) -> ChannelType:
        """Classifies channel type based on name keywords."""
        channel_lower = channel_name.lower()
        
        for channel_type, keywords in self.channel_keywords.items():
            if any(keyword in channel_lower for keyword in keywords):
                return channel_type
        
        return ChannelType.UNKNOWN
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values in the dataset."""
        df = df.copy()
        
        # For spend columns, fill missing values with 0
        exclude_cols = ["date", "profit", "days_since_start", "day_of_week", "is_weekend", "month"]
        spend_columns = [col for col in df.columns if col not in exclude_cols]
        
        for col in spend_columns:
            df[col] = df[col].fillna(0)
        
        # For profit, forward fill then backward fill
        df["profit"] = df["profit"].ffill().bfill()
        
        # For optional control variables, fill with 0
        for col in ["is_holiday", "promo_flag", "site_outage"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        return df
    
    def _validate_processed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and cleaning of processed data."""
        df = df.copy()
        
        # Ensure no negative values in spend columns (numeric only)
        exclude_cols = ["date", "profit", "days_since_start", "day_of_week", "is_weekend", "month"]
        spend_columns = [col for col in df.columns 
                        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        for col in spend_columns:
            df[col] = df[col].clip(lower=0)
        
        # Ensure profit is non-negative
        df["profit"] = df["profit"].clip(lower=0)
        
        # Remove any rows with all-zero spend and zero profit
        spend_sum = df[spend_columns].sum(axis=1)
        valid_rows = (spend_sum > 0) | (df["profit"] > 0)
        df = df[valid_rows].reset_index(drop=True)
        
        return df
    
    def get_parameter_grid(self, channel_info: Dict[str, ChannelInfo]) -> Dict[str, Dict[str, List[float]]]:
        """
        Returns parameter grid for optimization based on channel types.
        
        Args:
            channel_info: Dictionary of channel information
            
        Returns:
            Dictionary mapping channel names to their parameter grids
        """
        # Parameter grids by channel type (beta: saturation, r: adstock)
        # 3x3 grid: all channels use same beta and r values for fast training
        type_grids = {
            ChannelType.SEARCH_BRAND: {
                "beta": [0.7, 0.8, 0.9],
                "r": [0.1, 0.2, 0.3]
            },
            ChannelType.SEARCH_NON_BRAND: {
                "beta": [0.7, 0.8, 0.9],
                "r": [0.1, 0.2, 0.3]
            },
            ChannelType.SOCIAL: {
                "beta": [0.7, 0.8, 0.9],
                "r": [0.1, 0.2, 0.3]
            },
            ChannelType.TV_VIDEO: {
                "beta": [0.7, 0.8, 0.9],
                "r": [0.1, 0.2, 0.3]
            },
            ChannelType.DISPLAY: {
                "beta": [0.7, 0.8, 0.9],
                "r": [0.1, 0.2, 0.3]
            },
            ChannelType.UNKNOWN: {
                "beta": [0.7, 0.8, 0.9],
                "r": [0.1, 0.2, 0.3]
            }
        }
        
        channel_grids = {}
        for channel_name, info in channel_info.items():
            channel_grids[channel_name] = type_grids[info.type]
        
        return channel_grids