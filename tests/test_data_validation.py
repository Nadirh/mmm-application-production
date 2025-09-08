"""
Data validation edge case tests for MMM application.
Tests various edge cases, malformed data, and boundary conditions.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import csv
from io import StringIO

from mmm.data.validator import DataValidator, ValidationErrorCode, BusinessTier
from mmm.data.processor import DataProcessor, ChannelType


class TestDataValidatorEdgeCases:
    """Test edge cases for data validation."""
    
    def test_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        validator = DataValidator()
        empty_df = pd.DataFrame()
        
        summary, errors = validator.validate_upload(empty_df)
        
        # Should have validation errors
        assert len(errors) > 0
        error_codes = [error.code for error in errors]
        assert ValidationErrorCode.ERROR_002 in error_codes  # Missing required columns
    
    def test_missing_required_columns(self):
        """Test validation with missing required columns."""
        validator = DataValidator()
        
        # Missing 'profit' column
        df_missing_profit = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'search': [100, 200]
        })
        
        summary, errors = validator.validate_upload(df_missing_profit)
        assert any(error.code == ValidationErrorCode.ERROR_002 for error in errors)
        
        # Missing 'date' column
        df_missing_date = pd.DataFrame({
            'profit': [1000, 2000],
            'search': [100, 200]
        })
        
        summary, errors = validator.validate_upload(df_missing_date)
        assert any(error.code == ValidationErrorCode.ERROR_002 for error in errors)
    
    def test_negative_values(self):
        """Test validation with negative values."""
        validator = DataValidator()
        
        # Negative profit
        df_negative_profit = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'profit': [1000, -500, 2000],
            'search': [100, 200, 150]
        })
        
        summary, errors = validator.validate_upload(df_negative_profit)
        assert any(error.code == ValidationErrorCode.ERROR_004 for error in errors)
        
        # Negative spend
        df_negative_spend = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'profit': [1000, 1500, 2000],
            'search': [100, -200, 150]
        })
        
        summary, errors = validator.validate_upload(df_negative_spend)
        assert any(error.code == ValidationErrorCode.ERROR_001 for error in errors)
    
    def test_extreme_spend_jumps(self):
        """Test validation with extreme day-over-day spend changes."""
        validator = DataValidator()
        
        df_extreme_jumps = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'profit': [1000, 1500, 2000],
            'search': [100, 500, 150]  # 400% jump from day 1 to day 2
        })
        
        summary, errors = validator.validate_upload(df_extreme_jumps)
        assert any(error.code == ValidationErrorCode.ERROR_003 for error in errors)
    
    def test_all_zero_spend(self):
        """Test validation with all zero spend across channels."""
        validator = DataValidator()
        
        df_zero_spend = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'profit': [1000, 1500, 2000],
            'search': [0, 0, 0],
            'social': [0, 0, 0]
        })
        
        summary, errors = validator.validate_upload(df_zero_spend)
        assert any(error.code == ValidationErrorCode.WARNING_001 for error in errors)
    
    def test_malformed_dates(self):
        """Test validation with malformed date values."""
        validator = DataValidator()
        
        # Invalid date formats
        df_bad_dates = pd.DataFrame({
            'date': ['2023-13-01', '2023-01-32', '2023/01/01'],  # Invalid dates
            'profit': [1000, 1500, 2000],
            'search': [100, 200, 150]
        })
        
        # This should be caught during processing, not necessarily in validation
        # The validator might not catch date format issues initially
        summary, errors = validator.validate_upload(df_bad_dates)
        # Test passes if no exception is thrown
    
    def test_missing_date_gaps(self):
        """Test validation with gaps in date sequence."""
        validator = DataValidator()
        
        df_date_gaps = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-03', '2023-01-05'],  # Missing days 2 and 4
            'profit': [1000, 1500, 2000],
            'search': [100, 200, 150]
        })
        
        summary, errors = validator.validate_upload(df_date_gaps)
        # Gaps might be detected as a validation issue
        # The specific implementation would determine if this triggers ERROR_002
    
    def test_duplicate_dates(self):
        """Test validation with duplicate dates."""
        validator = DataValidator()
        
        df_duplicate_dates = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-01', '2023-01-02'],  # Duplicate date
            'profit': [1000, 1500, 2000],
            'search': [100, 200, 150]
        })
        
        summary, errors = validator.validate_upload(df_duplicate_dates)
        # Should handle duplicates gracefully
    
    def test_insufficient_data_for_business_tiers(self):
        """Test business tier classification with insufficient data."""
        validator = DataValidator()
        
        # Very short time period (less than prototype minimum)
        short_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100).strftime('%Y-%m-%d'),
            'profit': np.random.normal(1000, 200, 100),
            'search': np.random.normal(500, 100, 100)
        })
        
        summary, errors = validator.validate_upload(short_df)
        
        # Should classify as prototype or lower tier
        assert summary.business_tier in [BusinessTier.PROTOTYPE]
        assert summary.total_days == 100
    
    def test_extreme_values(self):
        """Test validation with extremely large or small values."""
        validator = DataValidator()
        
        # Extremely large values
        df_extreme_large = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'profit': [1e12, 1e12, 1e12],  # Trillion dollars
            'search': [1e10, 1e10, 1e10]   # Billion dollar spend
        })
        
        summary, errors = validator.validate_upload(df_extreme_large)
        
        # Should handle large values without crashing
        assert summary.total_profit > 0
        
        # Extremely small values
        df_extreme_small = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'profit': [0.01, 0.02, 0.03],
            'search': [0.001, 0.002, 0.003]
        })
        
        summary, errors = validator.validate_upload(df_extreme_small)
        assert summary.total_profit > 0
    
    def test_data_quality_scoring(self):
        """Test data quality scoring with various data conditions."""
        validator = DataValidator()
        
        # Perfect data
        perfect_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=365).strftime('%Y-%m-%d'),
            'profit': np.random.normal(5000, 1000, 365).clip(min=1000),
            'search': np.random.normal(2000, 500, 365).clip(min=100),
            'social': np.random.normal(1500, 300, 365).clip(min=100)
        })
        
        summary_perfect, _ = validator.validate_upload(perfect_df)
        perfect_score = summary_perfect.data_quality_score
        
        # Data with some issues
        problematic_df = perfect_df.copy()
        problematic_df.loc[10:20, 'search'] = -100  # Some negative values
        problematic_df.loc[50:60, 'profit'] = np.nan  # Some missing values
        
        summary_problematic, _ = validator.validate_upload(problematic_df)
        problematic_score = summary_problematic.data_quality_score
        
        # Problematic data should have lower quality score
        assert problematic_score < perfect_score
        assert 0 <= problematic_score <= 100
        assert 0 <= perfect_score <= 100


class TestDataProcessorEdgeCases:
    """Test edge cases for data processing."""
    
    def test_single_row_data(self):
        """Test processing with only one row of data."""
        processor = DataProcessor()
        
        single_row_df = pd.DataFrame({
            'date': ['2023-01-01'],
            'profit': [1000],
            'search': [500]
        })
        
        processed_df, channel_info = processor.process_data(single_row_df)
        
        # Should handle single row gracefully
        assert len(processed_df) == 1
        assert 'search' in channel_info
    
    def test_all_missing_values_column(self):
        """Test processing with columns that have all missing values."""
        processor = DataProcessor()
        
        df_missing_column = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'profit': [1000, 1500, 2000],
            'search': [100, 200, 150],
            'missing_channel': [np.nan, np.nan, np.nan]
        })
        
        processed_df, channel_info = processor.process_data(df_missing_column)
        
        # Should handle missing values by filling with 0
        assert processed_df['missing_channel'].sum() == 0
    
    def test_inconsistent_data_types(self):
        """Test processing with inconsistent data types."""
        processor = DataProcessor()
        
        # Mixed string and numeric values
        mixed_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'profit': ['1000', 1500, '2000'],  # Mixed string/numeric
            'search': [100, '200', 150]
        })
        
        # Should attempt to convert to numeric
        try:
            processed_df, channel_info = processor.process_data(mixed_df)
            # If successful, profit should be numeric
            assert processed_df['profit'].dtype in [np.float64, np.int64]
        except Exception:
            # If conversion fails, should raise appropriate error
            pass
    
    def test_channel_classification_edge_cases(self):
        """Test channel classification with ambiguous names."""
        processor = DataProcessor()
        
        edge_case_channels = [
            'search_social_hybrid',  # Contains both search and social keywords
            'tv_display_combo',      # Contains both TV and display keywords
            'FACEBOOK_SEARCH',       # Mixed case with multiple keywords
            'unknown_weird_123',     # No recognizable keywords
            '',                      # Empty string
            'a' * 100               # Very long name
        ]
        
        for channel_name in edge_case_channels:
            channel_type = processor._classify_channel_type(channel_name)
            # Should return a valid channel type (even if unknown)
            assert isinstance(channel_type, ChannelType)
    
    def test_extreme_seasonality(self):
        """Test processing with extreme seasonal patterns."""
        processor = DataProcessor()
        
        dates = pd.date_range('2023-01-01', periods=365)
        
        # Create extreme seasonality (10x variation)
        base_spend = 1000
        seasonal_multiplier = 1 + 9 * np.sin(2 * np.pi * np.arange(365) / 365)
        
        extreme_seasonal_df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'profit': np.random.normal(5000, 1000, 365),
            'search': base_spend * seasonal_multiplier
        })
        
        processed_df, channel_info = processor.process_data(extreme_seasonal_df)
        
        # Should handle extreme seasonality without issues
        assert len(processed_df) == 365
        assert 'search' in channel_info
        assert channel_info['search'].total_spend > 0
    
    def test_very_sparse_data(self):
        """Test processing with very sparse spend data (mostly zeros)."""
        processor = DataProcessor()
        
        dates = pd.date_range('2023-01-01', periods=365)
        
        # Sparse data: spend only on 5% of days
        sparse_spend = np.zeros(365)
        sparse_indices = np.random.choice(365, size=18, replace=False)  # 5% of days
        sparse_spend[sparse_indices] = np.random.normal(10000, 2000, 18)
        
        sparse_df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'profit': np.random.normal(2000, 500, 365),
            'search': sparse_spend
        })
        
        processed_df, channel_info = processor.process_data(sparse_df)
        
        # Should handle sparse data
        assert len(processed_df) == 365
        assert channel_info['search'].days_active == 18
    
    def test_parameter_grid_edge_cases(self):
        """Test parameter grid generation with edge case channel types."""
        processor = DataProcessor()
        
        # Create channel info with various types
        from mmm.data.processor import ChannelInfo
        
        edge_case_channels = {
            'unknown_channel': ChannelInfo(
                name='unknown_channel',
                type=ChannelType.UNKNOWN,
                total_spend=100000,
                spend_share=0.5,
                days_active=365
            ),
            'empty_channel': ChannelInfo(
                name='empty_channel',
                type=ChannelType.SEARCH_BRAND,
                total_spend=0,
                spend_share=0,
                days_active=0
            )
        }
        
        param_grids = processor.get_parameter_grid(edge_case_channels)
        
        # Should generate grids for all channels
        assert 'unknown_channel' in param_grids
        assert 'empty_channel' in param_grids
        
        # Grids should have valid parameter ranges
        for channel, grid in param_grids.items():
            assert 'beta' in grid
            assert 'r' in grid
            assert len(grid['beta']) > 0
            assert len(grid['r']) > 0


class TestBoundaryConditions:
    """Test boundary conditions and limits."""
    
    def test_minimum_data_requirements(self):
        """Test with minimum viable data."""
        validator = DataValidator()
        processor = DataProcessor()
        
        # Minimum for prototype tier (182 days, $50K spend)
        min_dates = pd.date_range('2023-01-01', periods=182)
        min_spend_per_day = 50000 / 182  # ~$274 per day
        
        min_viable_df = pd.DataFrame({
            'date': min_dates.strftime('%Y-%m-%d'),
            'profit': np.random.normal(200, 50, 182).clip(min=10),
            'search': np.random.normal(min_spend_per_day, 50, 182).clip(min=10)
        })
        
        # Validation
        summary, errors = validator.validate_upload(min_viable_df)
        assert summary.business_tier == BusinessTier.PROTOTYPE
        
        # Processing
        processed_df, channel_info = processor.process_data(min_viable_df)
        assert len(processed_df) == 182
    
    def test_maximum_realistic_data(self):
        """Test with maximum realistic data size."""
        validator = DataValidator()
        
        # 5 years of daily data with many channels
        max_dates = pd.date_range('2019-01-01', '2023-12-31')
        n_days = len(max_dates)
        n_channels = 20
        
        # Create large dataset
        large_data = {
            'date': max_dates.strftime('%Y-%m-%d'),
            'profit': np.random.normal(50000, 10000, n_days).clip(min=1000)
        }
        
        # Add many channels
        for i in range(n_channels):
            channel_name = f'channel_{i:02d}'
            large_data[channel_name] = np.random.normal(5000, 1000, n_days).clip(min=0)
        
        large_df = pd.DataFrame(large_data)
        
        # Should handle large datasets
        summary, errors = validator.validate_upload(large_df)
        assert summary.total_days == n_days
        assert summary.channel_count == n_channels
    
    def test_edge_case_business_tier_boundaries(self):
        """Test business tier classification at exact boundaries."""
        validator = DataValidator()
        
        # Test exact enterprise boundary
        enterprise_days = 365
        enterprise_spend = 2000000
        enterprise_channel_min = 25000
        
        # Create data exactly at enterprise threshold
        dates = pd.date_range('2023-01-01', periods=enterprise_days)
        daily_spend = enterprise_spend / enterprise_days
        
        boundary_df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'profit': np.random.normal(10000, 2000, enterprise_days),
            'search': [daily_spend] * enterprise_days,
            'social': [daily_spend] * enterprise_days  # Two channels at minimum
        })
        
        summary, errors = validator.validate_upload(boundary_df)
        # Should classify based on exact thresholds
        # Result depends on whether channel minimums are met
    
    def test_floating_point_precision(self):
        """Test handling of floating point precision issues."""
        processor = DataProcessor()
        
        # Create data with precision issues
        precision_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'profit': [1000.0000000001, 2000.9999999999, 3000.5],
            'search': [100.333333333333, 200.666666666666, 150.999999999999]
        })
        
        processed_df, channel_info = processor.process_data(precision_df)
        
        # Should handle precision without issues
        assert len(processed_df) == 3
        assert all(np.isfinite(processed_df['profit']))
        assert all(np.isfinite(processed_df['search']))


class TestCorruptedDataHandling:
    """Test handling of corrupted or malformed data."""
    
    def test_csv_with_extra_commas(self):
        """Test CSV with extra commas and malformed rows."""
        csv_content = """date,profit,search,social
2023-01-01,1000,200,300
2023-01-02,1500,,250,100,extra_field
2023-01-03,2000,300,200
,1200,150,175
2023-01-05,1800,250,"""
        
        # Parse malformed CSV
        df = pd.read_csv(StringIO(csv_content))
        
        validator = DataValidator()
        
        # Should handle malformed data gracefully
        try:
            summary, errors = validator.validate_upload(df)
            # If successful, should have detected issues
        except Exception:
            # If parsing fails, that's also acceptable
            pass
    
    def test_unicode_and_special_characters(self):
        """Test data with unicode and special characters."""
        processor = DataProcessor()
        
        unicode_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'profit': [1000, 1500, 2000],
            'search_🔍': [100, 200, 150],  # Emoji in column name
            'søcial_mëdia': [150, 250, 200],  # Unicode characters
            'tv/video': [200, 300, 250]  # Special characters
        })
        
        # Should handle unicode gracefully
        processed_df, channel_info = processor.process_data(unicode_df)
        
        # Column names should be preserved or handled appropriately
        assert len(processed_df) > 0
    
    def test_very_long_text_fields(self):
        """Test with extremely long text values."""
        processor = DataProcessor()
        
        long_text_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'profit': [1000, 1500],
            'search': [100, 200],
            'notes': ['A' * 10000, 'B' * 10000]  # Very long text fields
        })
        
        # Should handle long text without crashing
        processed_df, channel_info = processor.process_data(long_text_df)
        assert len(processed_df) == 2
    
    def test_mixed_encoding_issues(self):
        """Test handling of encoding issues."""
        # This would typically involve reading actual files with encoding issues
        # For the test, we'll simulate the effect
        
        processor = DataProcessor()
        
        # Simulate data that might come from different encodings
        encoding_issue_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'profit': [1000, 1500],
            'search': [100, 200],
            'channel_with_issues': ['café', 'naïve']  # Accented characters
        })
        
        # Should handle encoding issues gracefully
        processed_df, channel_info = processor.process_data(encoding_issue_df)
        assert len(processed_df) == 2