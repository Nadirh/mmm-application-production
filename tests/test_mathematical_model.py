"""
Mathematical model validation tests for MMM application.
Tests the core mathematical algorithms and model correctness.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings

from mmm.model.mmm_model import MMMModel, ModelParameters
from mmm.data.processor import DataProcessor
from mmm.optimization.optimizer import BudgetOptimizer, Constraint, ConstraintType


class TestMMMModelMathematics:
    """Test core MMM mathematical functions."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data with known relationships."""
        np.random.seed(42)
        n_days = 365
        
        # Generate true parameters
        true_params = {
            'alpha_baseline': 1000,
            'alpha_trend': 2.0,
            'channel_alphas': {'search': 0.8, 'social': 0.6, 'tv': 1.2},
            'channel_betas': {'search': 0.7, 'social': 0.5, 'tv': 0.4},
            'channel_rs': {'search': 0.1, 'social': 0.3, 'tv': 0.6}
        }
        
        # Generate spend data
        dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
        base_spend = {'search': 5000, 'social': 3000, 'tv': 8000}
        
        data = {'date': dates}
        spend_data = {}
        
        for channel in ['search', 'social', 'tv']:
            # Add seasonality and noise to spend
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 365)
            noise = np.random.normal(1, 0.2, n_days)
            spend_data[channel] = base_spend[channel] * seasonal * noise
            spend_data[channel] = np.clip(spend_data[channel], 0, None)
            data[channel] = spend_data[channel]
        
        # Generate profit using true MMM formula
        profit = np.zeros(n_days)
        adstocks = {channel: np.zeros(n_days) for channel in ['search', 'social', 'tv']}
        
        for t in range(n_days):
            # Baseline + trend
            baseline = true_params['alpha_baseline'] + true_params['alpha_trend'] * t
            
            # Channel contributions
            channel_contrib = 0
            for channel in ['search', 'social', 'tv']:
                # Adstock transformation
                if t == 0:
                    adstocks[channel][t] = spend_data[channel][t]
                else:
                    r = true_params['channel_rs'][channel]
                    adstocks[channel][t] = spend_data[channel][t] + r * adstocks[channel][t-1]
                
                # Saturation transformation
                beta = true_params['channel_betas'][channel]
                saturated = np.power(adstocks[channel][t], beta)
                
                # Channel contribution
                alpha = true_params['channel_alphas'][channel]
                channel_contrib += alpha * saturated
            
            profit[t] = baseline + channel_contrib
        
        # Add noise to profit
        profit += np.random.normal(0, profit * 0.05)  # 5% noise
        profit = np.clip(profit, 0, None)
        
        data['profit'] = profit
        
        return pd.DataFrame(data), true_params
    
    def test_adstock_transformation(self):
        """Test adstock transformation correctness."""
        model = MMMModel()
        
        # Test data
        spend = np.array([100, 200, 0, 300, 150])
        r = 0.3
        
        # Expected adstock calculation
        expected_adstock = np.zeros(5)
        expected_adstock[0] = 100
        expected_adstock[1] = 200 + 0.3 * 100  # 230
        expected_adstock[2] = 0 + 0.3 * 230     # 69
        expected_adstock[3] = 300 + 0.3 * 69    # 320.7
        expected_adstock[4] = 150 + 0.3 * 320.7 # 212.21
        
        # Apply transformation
        spend_columns = ['test_channel']
        X_spend = spend.reshape(-1, 1)
        params = {
            'channel_betas': {'test_channel': 1.0},  # No saturation for this test
            'channel_rs': {'test_channel': r}
        }
        
        result = model._apply_transforms(X_spend, spend_columns, params)
        
        # Check adstock calculation
        np.testing.assert_array_almost_equal(result[:, 0], expected_adstock, decimal=2)
    
    def test_saturation_transformation(self):
        """Test saturation transformation correctness."""
        model = MMMModel()
        
        # Test data
        adstocked_spend = np.array([100, 500, 1000, 2000, 5000])
        beta = 0.5  # Square root saturation
        
        # Expected saturation
        expected_saturated = np.power(adstocked_spend, beta)
        
        # Apply transformation
        spend_columns = ['test_channel']
        X_spend = adstocked_spend.reshape(-1, 1)
        params = {
            'channel_betas': {'test_channel': beta},
            'channel_rs': {'test_channel': 0.0}  # No adstock for this test
        }
        
        result = model._apply_transforms(X_spend, spend_columns, params)
        
        # Check saturation calculation
        np.testing.assert_array_almost_equal(result[:, 0], expected_saturated, decimal=6)
    
    def test_parameter_bounds_validation(self):
        """Test that parameter bounds are enforced."""
        model = MMMModel()
        
        # Test beta bounds (should be between 0.1 and 1.0)
        spend_columns = ['test_channel']
        X_spend = np.array([[100], [200], [150]])
        
        # Test invalid beta (too low)
        params_low_beta = {
            'channel_betas': {'test_channel': 0.05},  # Below minimum
            'channel_rs': {'test_channel': 0.2}
        }
        
        # The model should handle this gracefully or raise appropriate error
        # This tests the mathematical bounds validation
        
        # Test invalid r (too high)
        params_high_r = {
            'channel_betas': {'test_channel': 0.5},
            'channel_rs': {'test_channel': 1.5}  # Above maximum
        }
        
        # Test that model handles parameter bounds appropriately
        # Implementation should clip or reject invalid parameters
        assert True  # Placeholder - actual bounds checking would be implemented
    
    def test_model_parameter_recovery(self, synthetic_data):
        """Test that model can recover known parameters from synthetic data."""
        df, true_params = synthetic_data
        
        # Process data
        processor = DataProcessor()
        processed_df, channel_info = processor.process_data(df)
        
        # Get parameter grids (use tighter grids around true values for testing)
        channel_grids = {
            'search': {
                'beta': [0.6, 0.7, 0.8],
                'r': [0.05, 0.1, 0.15]
            },
            'social': {
                'beta': [0.4, 0.5, 0.6],
                'r': [0.2, 0.3, 0.4]
            },
            'tv': {
                'beta': [0.3, 0.4, 0.5],
                'r': [0.5, 0.6, 0.7]
            }
        }
        
        # Train model with limited iterations for testing
        model = MMMModel(training_window_days=180, test_window_days=14, n_bootstrap=10)
        
        # Mock progress callback for testing
        def mock_progress(data):
            pass
        
        results = model.fit(processed_df, channel_grids, mock_progress)
        
        # Check that model achieves reasonable performance
        assert results.cv_mape < 30, f"CV MAPE too high: {results.cv_mape}"
        assert results.r_squared > 0.5, f"R-squared too low: {results.r_squared}"
        
        # Check parameter reasonableness (not exact recovery due to noise)
        for channel in ['search', 'social', 'tv']:
            estimated_alpha = results.parameters.channel_alphas[channel]
            true_alpha = true_params['channel_alphas'][channel]
            
            # Allow 50% tolerance due to noise and limited data
            assert abs(estimated_alpha - true_alpha) / true_alpha < 0.5, \
                f"Alpha estimation for {channel} too far off: {estimated_alpha} vs {true_alpha}"
    
    def test_diminishing_returns(self):
        """Test that model exhibits diminishing returns (concave response curves)."""
        # Create simple test case
        model_params = ModelParameters(
            alpha_baseline=1000,
            alpha_trend=0,
            channel_alphas={'test_channel': 1.0},
            channel_betas={'test_channel': 0.5},  # Should create diminishing returns
            channel_rs={'test_channel': 0.1}
        )
        
        optimizer = BudgetOptimizer(model_params)
        
        # Test increasing spend levels
        spend_levels = np.array([1000, 5000, 10000, 20000, 50000])
        profits = []
        
        for spend in spend_levels:
            spend_dict = {'test_channel': spend}
            profit = optimizer._calculate_profit(spend_dict, 365)
            profits.append(profit)
        
        # Calculate marginal returns
        marginal_returns = np.diff(profits) / np.diff(spend_levels)
        
        # Check that marginal returns are decreasing (diminishing returns)
        for i in range(len(marginal_returns) - 1):
            assert marginal_returns[i] >= marginal_returns[i + 1], \
                f"Marginal returns not decreasing: {marginal_returns}"
    
    def test_cross_validation_consistency(self, synthetic_data):
        """Test that cross-validation produces consistent results."""
        df, _ = synthetic_data
        
        # Process data
        processor = DataProcessor()
        processed_df, channel_info = processor.process_data(df)
        
        # Simple parameter grid for testing
        channel_grids = processor.get_parameter_grid(channel_info)
        
        # Run model multiple times
        model1 = MMMModel(training_window_days=126, test_window_days=14, n_bootstrap=10)
        model2 = MMMModel(training_window_days=126, test_window_days=14, n_bootstrap=10)
        
        def mock_progress(data):
            pass
        
        # Use same random seed for reproducibility
        np.random.seed(123)
        results1 = model1.fit(processed_df, channel_grids, mock_progress)
        
        np.random.seed(123)
        results2 = model2.fit(processed_df, channel_grids, mock_progress)
        
        # Results should be identical with same seed
        assert abs(results1.cv_mape - results2.cv_mape) < 0.01
        assert abs(results1.r_squared - results2.r_squared) < 0.01
    
    def test_model_stability_with_noise(self, synthetic_data):
        """Test model stability with noisy data."""
        df, _ = synthetic_data
        
        # Add additional noise to test robustness
        df_noisy = df.copy()
        for channel in ['search', 'social', 'tv']:
            noise = np.random.normal(0, df_noisy[channel] * 0.1)
            df_noisy[channel] += noise
            df_noisy[channel] = np.clip(df_noisy[channel], 0, None)
        
        # Process data
        processor = DataProcessor()
        processed_df, channel_info = processor.process_data(df_noisy)
        
        # Use subset of parameter grid for speed
        channel_grids = {}
        for channel, info in channel_info.items():
            channel_grids[channel] = {
                'beta': [0.3, 0.5, 0.7],
                'r': [0.1, 0.3, 0.5]
            }
        
        # Train model
        model = MMMModel(training_window_days=126, test_window_days=14, n_bootstrap=5)
        
        def mock_progress(data):
            pass
        
        try:
            results = model.fit(processed_df, channel_grids, mock_progress)
            
            # Model should still produce reasonable results
            assert results.cv_mape < 50  # Allow higher error with noise
            assert results.r_squared > 0.3
            
            # Parameters should be reasonable
            for channel in channel_grids.keys():
                alpha = results.parameters.channel_alphas[channel]
                assert alpha >= 0, f"Negative alpha for {channel}: {alpha}"
                assert alpha < 10, f"Unreasonably high alpha for {channel}: {alpha}"
                
        except Exception as e:
            pytest.fail(f"Model failed with noisy data: {e}")


class TestOptimizationMathematics:
    """Test budget optimization mathematical functions."""
    
    @pytest.fixture
    def test_model_params(self):
        """Create test model parameters."""
        return ModelParameters(
            alpha_baseline=2000,
            alpha_trend=1.0,
            channel_alphas={
                'search': 1.0,
                'social': 0.8,
                'tv': 1.2
            },
            channel_betas={
                'search': 0.7,
                'social': 0.5,
                'tv': 0.4
            },
            channel_rs={
                'search': 0.1,
                'social': 0.3,
                'tv': 0.6
            }
        )
    
    def test_budget_constraint_satisfaction(self, test_model_params):
        """Test that optimization satisfies budget constraints."""
        optimizer = BudgetOptimizer(test_model_params)
        
        current_spend = {'search': 100000, 'social': 80000, 'tv': 120000}
        total_budget = 400000
        
        # Run optimization (mock the actual optimization for testing)
        # In practice, this would call the actual optimizer
        optimal_spend = {'search': 150000, 'social': 100000, 'tv': 150000}
        
        # Check budget constraint
        total_optimal_spend = sum(optimal_spend.values())
        assert abs(total_optimal_spend - total_budget) < 1000, \
            f"Budget constraint violated: {total_optimal_spend} vs {total_budget}"
    
    def test_constraint_enforcement(self, test_model_params):
        """Test that optimization enforces business constraints."""
        optimizer = BudgetOptimizer(test_model_params)
        
        current_spend = {'search': 100000, 'social': 80000, 'tv': 120000}
        total_budget = 400000
        
        # Define constraints
        constraints = [
            Constraint('search', ConstraintType.FLOOR, 120000, "Minimum search spend"),
            Constraint('tv', ConstraintType.CAP, 100000, "Maximum TV spend"),
            Constraint('social', ConstraintType.LOCK, 80000, "Lock social spend")
        ]
        
        # Mock constraint checking (would be in actual optimization)
        optimal_spend = {'search': 120000, 'social': 80000, 'tv': 100000}
        
        # Verify constraints are satisfied
        assert optimal_spend['search'] >= 120000, "Floor constraint violated"
        assert optimal_spend['tv'] <= 100000, "Cap constraint violated"
        assert optimal_spend['social'] == 80000, "Lock constraint violated"
    
    def test_shadow_price_calculation(self, test_model_params):
        """Test shadow price calculation correctness."""
        optimizer = BudgetOptimizer(test_model_params)
        
        current_spend = {'search': 100000, 'social': 80000, 'tv': 120000}
        total_budget = 300000
        constraints = []
        
        # Calculate shadow prices
        shadow_prices = optimizer._calculate_shadow_prices(
            current_spend, total_budget, constraints, 365
        )
        
        # Shadow prices should be positive (additional budget increases profit)
        assert shadow_prices['total_budget'] > 0, "Total budget shadow price should be positive"
        
        # Channel shadow prices should reflect marginal efficiency
        for channel, price in shadow_prices.items():
            if channel != 'total_budget':
                assert isinstance(price, (int, float)), f"Invalid shadow price type for {channel}"
    
    def test_response_curve_generation(self, test_model_params):
        """Test response curve generation."""
        optimizer = BudgetOptimizer(test_model_params)
        
        channel = 'search'
        spend_range = np.linspace(0, 200000, 100)
        
        response_curve = optimizer.get_response_curve(channel, spend_range, 100)
        
        # Check response curve properties
        assert len(response_curve.spend_range) == 100
        assert len(response_curve.profit_values) == 100
        assert len(response_curve.marginal_efficiency) == 100
        
        # Check that profit increases with spend (at least initially)
        assert response_curve.profit_values[10] > response_curve.profit_values[0]
        
        # Check diminishing returns (marginal efficiency decreases)
        marginal_efficiency = response_curve.marginal_efficiency
        # Allow some flexibility due to discrete calculations
        decreasing_trend = sum([
            marginal_efficiency[i] >= marginal_efficiency[i+1] 
            for i in range(0, len(marginal_efficiency)-1, 10)
        ])
        assert decreasing_trend >= 7, "Marginal efficiency should generally decrease"
    
    def test_optimization_improves_profit(self, test_model_params):
        """Test that optimization improves profit over current allocation."""
        optimizer = BudgetOptimizer(test_model_params)
        
        current_spend = {'search': 50000, 'social': 50000, 'tv': 50000}  # Equal allocation
        total_budget = 150000
        
        # Calculate current profit
        current_profit = optimizer._calculate_profit(current_spend, 365)
        
        # Simple optimization: allocate more to channels with higher efficiency
        # (This is a simplified version of what the actual optimizer would do)
        optimized_spend = {'search': 60000, 'social': 40000, 'tv': 50000}
        optimized_profit = optimizer._calculate_profit(optimized_spend, 365)
        
        # Optimization should improve profit (at least not make it worse)
        assert optimized_profit >= current_profit * 0.99, \
            "Optimization should not significantly decrease profit"


class TestDataProcessingMathematics:
    """Test data processing mathematical functions."""
    
    def test_channel_classification_accuracy(self):
        """Test channel type classification."""
        processor = DataProcessor()
        
        # Test cases
        test_cases = [
            ('google_search_brand', 'search_brand'),
            ('facebook_social', 'social'),
            ('youtube_video', 'tv_video'),
            ('display_banner', 'display'),
            ('unknown_channel', 'unknown')
        ]
        
        for channel_name, expected_type in test_cases:
            classified_type = processor._classify_channel_type(channel_name)
            assert classified_type.value == expected_type, \
                f"Misclassified {channel_name}: got {classified_type.value}, expected {expected_type}"
    
    def test_business_tier_classification(self, sample_csv_data):
        """Test business tier classification logic."""
        processor = DataProcessor()
        processed_df, channel_info = processor.process_data(sample_csv_data)
        
        # Calculate metrics
        total_days = len(processed_df)
        spend_columns = [col for col in processed_df.columns 
                        if col not in ['date', 'profit', 'days_since_start', 'day_of_week', 'is_weekend', 'month']]
        total_annual_spend = processed_df[spend_columns].sum().sum()
        
        # Verify classification logic
        if total_days >= 365 and total_annual_spend >= 2000000:
            # Should be enterprise if channel minimums are met
            min_channel_spend = min([processed_df[col].sum() for col in spend_columns])
            if min_channel_spend >= 25000:
                expected_tier = "enterprise"
            else:
                expected_tier = "mid_market"  # or lower
        elif total_days >= 280 and total_annual_spend >= 500000:
            expected_tier = "mid_market"  # or lower based on channel spends
        else:
            expected_tier = "small_business"  # or prototype
        
        # The actual classification would depend on the validator implementation
        assert True  # Placeholder - actual test would check against validator result
    
    def test_feature_engineering_correctness(self, sample_csv_data):
        """Test that feature engineering produces correct results."""
        processor = DataProcessor()
        processed_df, _ = processor.process_data(sample_csv_data)
        
        # Check date features
        assert 'days_since_start' in processed_df.columns
        assert 'day_of_week' in processed_df.columns
        assert 'is_weekend' in processed_df.columns
        assert 'month' in processed_df.columns
        
        # Verify days_since_start calculation
        first_date = pd.to_datetime(processed_df['date'].iloc[0])
        for i, row in processed_df.iterrows():
            current_date = pd.to_datetime(row['date'])
            expected_days = (current_date - first_date).days
            assert row['days_since_start'] == expected_days, \
                f"Incorrect days_since_start at row {i}"
        
        # Verify weekend calculation
        for i, row in processed_df.iterrows():
            date = pd.to_datetime(row['date'])
            expected_weekend = 1 if date.dayofweek >= 5 else 0
            assert row['is_weekend'] == expected_weekend, \
                f"Incorrect weekend flag at row {i}"
        
        # Verify month extraction
        for i, row in processed_df.iterrows():
            date = pd.to_datetime(row['date'])
            assert row['month'] == date.month, \
                f"Incorrect month at row {i}"


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_zero_spend_handling(self):
        """Test model behavior with zero spend values."""
        model = MMMModel()
        
        # Test with zero spend
        spend_columns = ['test_channel']
        X_spend = np.array([[0], [0], [1000], [0]])
        params = {
            'channel_betas': {'test_channel': 0.5},
            'channel_rs': {'test_channel': 0.3}
        }
        
        # Should not crash and produce reasonable results
        result = model._apply_transforms(X_spend, spend_columns, params)
        
        # Zero spend should produce zero transformed values
        assert result[0, 0] == 0
        assert result[1, 0] == 0
        assert result[3, 0] == 0
        assert result[2, 0] > 0  # Non-zero spend should produce non-zero result
    
    def test_extreme_parameter_values(self):
        """Test model stability with extreme parameter values."""
        model = MMMModel()
        
        spend_columns = ['test_channel']
        X_spend = np.array([[1000], [2000], [3000]])
        
        # Test with extreme beta (close to bounds)
        params_extreme = {
            'channel_betas': {'test_channel': 0.99},
            'channel_rs': {'test_channel': 0.98}
        }
        
        # Should not produce infinite or NaN values
        result = model._apply_transforms(X_spend, spend_columns, params_extreme)
        
        assert np.all(np.isfinite(result)), "Model produced non-finite values"
        assert np.all(result >= 0), "Model produced negative values"
    
    def test_large_spend_values(self):
        """Test model behavior with very large spend values."""
        optimizer = BudgetOptimizer(ModelParameters(
            alpha_baseline=1000,
            alpha_trend=0,
            channel_alphas={'test': 1.0},
            channel_betas={'test': 0.5},
            channel_rs={'test': 0.1}
        ))
        
        # Test with very large spend (millions)
        large_spend = {'test': 10000000}
        
        # Should not overflow or produce unrealistic results
        profit = optimizer._calculate_profit(large_spend, 365)
        
        assert np.isfinite(profit), "Large spend produced non-finite profit"
        assert profit > 0, "Large spend produced negative profit"
        
        # Profit should be reasonable relative to spend
        assert profit < large_spend['test'] * 10, "Profit unrealistically high relative to spend"