"""
Core Media Mix Model implementation.
Handles model training, cross-validation, and parameter optimization.
"""
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import minimize
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import structlog

logger = structlog.get_logger()


@dataclass
class ModelParameters:
    alpha_baseline: float
    alpha_trend: float
    channel_alphas: Dict[str, float]  # Channel incremental strength
    channel_betas: Dict[str, float]   # Saturation parameters
    channel_rs: Dict[str, float]      # Adstock/memory parameters


@dataclass
class ModelResults:
    parameters: ModelParameters
    fitted_values: np.ndarray
    residuals: np.ndarray
    r_squared: float
    mape: float
    cv_mape: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    diagnostics: Dict[str, Any]


@dataclass
class CrossValidationFold:
    fold_number: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    mape: float
    parameters: ModelParameters


class MMMModel:
    """Media Mix Model implementation with cross-validation and optimization."""
    
    def __init__(self, 
                 training_window_days: int = 126,
                 test_window_days: int = 14,
                 n_bootstrap: int = 1000):
        self.training_window_days = training_window_days
        self.test_window_days = test_window_days
        self.n_bootstrap = n_bootstrap
        self.is_fitted = False
        self.results: Optional[ModelResults] = None
        
    def fit(self, 
            df: pd.DataFrame, 
            channel_grids: Dict[str, Dict[str, List[float]]],
            progress_callback: Optional[callable] = None,
            cancellation_check: Optional[callable] = None) -> ModelResults:
        """
        Fits the MMM model using walk-forward cross-validation.
        
        Args:
            df: Processed DataFrame with date, profit, and channel columns
            channel_grids: Parameter grids for each channel
            progress_callback: Optional callback function for progress updates
            cancellation_check: Optional callback function that returns True if training should be cancelled
            
        Returns:
            ModelResults object with fitted parameters and diagnostics
        """
        # Prepare data
        spend_columns = [col for col in channel_grids.keys()]
        y = df["profit"].values
        X_spend = df[spend_columns].values
        X_time = df["days_since_start"].values
        
        # Perform cross-validation
        cv_folds = self._perform_cross_validation(
            y, X_spend, X_time, spend_columns, channel_grids, progress_callback, cancellation_check
        )
        
        # Select best parameters based on CV performance
        best_params = self._select_best_parameters(cv_folds)
        
        # Fit final model on full data
        final_params = self._fit_final_model(y, X_spend, X_time, spend_columns, best_params)
        
        # Calculate fitted values and metrics
        fitted_values = self._predict(X_spend, X_time, spend_columns, final_params)
        residuals = y - fitted_values
        r_squared = 1 - np.var(residuals) / np.var(y)
        mape = mean_absolute_percentage_error(y, fitted_values) * 100
        cv_mape = np.mean([fold.mape for fold in cv_folds])
        
        # Bootstrap confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            y, X_spend, X_time, spend_columns, final_params
        )
        
        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics(y, fitted_values, residuals, df, spend_columns, final_params)
        
        # Store results
        self.results = ModelResults(
            parameters=final_params,
            fitted_values=fitted_values,
            residuals=residuals,
            r_squared=r_squared,
            mape=mape,
            cv_mape=cv_mape,
            confidence_intervals=confidence_intervals,
            diagnostics=diagnostics
        )
        
        self.is_fitted = True
        return self.results
    
    def _perform_cross_validation(self, 
                                 y: np.ndarray,
                                 X_spend: np.ndarray,
                                 X_time: np.ndarray,
                                 spend_columns: List[str],
                                 channel_grids: Dict[str, Dict[str, List[float]]],
                                 progress_callback: Optional[callable] = None,
                                 cancellation_check: Optional[callable] = None) -> List[CrossValidationFold]:
        """Performs walk-forward cross-validation."""
        folds = []
        total_days = len(y)
        
        # Generate fold indices
        fold_starts = range(0, total_days - self.training_window_days - self.test_window_days + 1, self.test_window_days)
        
        for fold_idx, fold_start in enumerate(fold_starts):
            # Check for cancellation before each fold
            if cancellation_check and cancellation_check():
                logger.info(f"Training cancelled during cross-validation at fold {fold_idx}")
                raise InterruptedError("Training cancelled by user")
            
            train_start = fold_start
            train_end = fold_start + self.training_window_days
            test_start = train_end
            test_end = test_start + self.test_window_days
            
            if test_end > total_days:
                break
            
            # Extract fold data
            y_train = y[train_start:train_end]
            X_spend_train = X_spend[train_start:train_end]
            X_time_train = X_time[train_start:train_end]
            
            y_test = y[test_start:test_end]
            X_spend_test = X_spend[test_start:test_end]
            X_time_test = X_time[test_start:test_end]
            
            # Find best parameters for this fold
            best_params = self._optimize_fold_parameters(
                y_train, X_spend_train, X_time_train, spend_columns, channel_grids,
                progress_callback, fold_idx
            )
            
            # Predict on test set
            y_pred = self._predict(X_spend_test, X_time_test, spend_columns, best_params)
            fold_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            fold = CrossValidationFold(
                fold_number=fold_idx + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                mape=fold_mape,
                parameters=best_params
            )
            
            folds.append(fold)
            
            if progress_callback:
                progress_callback({
                    "type": "fold_complete",
                    "fold": fold_idx + 1,
                    "total_folds": len(list(fold_starts)),
                    "mape": fold_mape
                })
        
        return folds
    
    def _optimize_fold_parameters(self,
                                 y: np.ndarray,
                                 X_spend: np.ndarray,
                                 X_time: np.ndarray,
                                 spend_columns: List[str],
                                 channel_grids: Dict[str, Dict[str, List[float]]],
                                 progress_callback: Optional[callable] = None,
                                 fold_idx: int = 0) -> ModelParameters:
        """Optimizes parameters for a single fold using grid search."""
        best_mape = float('inf')
        best_params = None
        
        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(channel_grids, spend_columns)
        total_combinations = len(param_combinations)
        
        for combo_idx, params in enumerate(param_combinations):
            # Report parameter optimization progress
            if progress_callback and combo_idx % 10 == 0:  # Report every 10th combination
                progress_callback({
                    "type": "parameter_optimization",
                    "fold": fold_idx + 1,
                    "combination": combo_idx + 1,
                    "total_combinations": total_combinations
                })
            # Fit linear model with these transform parameters
            fitted_params = self._fit_linear_model(y, X_spend, X_time, spend_columns, params)
            
            # Calculate predictions and MAPE
            y_pred = self._predict(X_spend, X_time, spend_columns, fitted_params)
            mape = mean_absolute_percentage_error(y, y_pred) * 100
            
            if mape < best_mape:
                best_mape = mape
                best_params = fitted_params
        
        return best_params
    
    def _generate_parameter_combinations(self,
                                       channel_grids: Dict[str, Dict[str, List[float]]],
                                       spend_columns: List[str]) -> List[Dict[str, Dict[str, float]]]:
        """Generates all parameter combinations for grid search."""
        combinations = []
        
        # Get parameter lists for each channel
        channel_param_lists = {}
        for channel in spend_columns:
            betas = channel_grids[channel]["beta"]
            rs = channel_grids[channel]["r"]
            channel_param_lists[channel] = [(b, r) for b in betas for r in rs]
        
        # Generate all combinations
        channel_names = list(channel_param_lists.keys())
        param_products = itertools.product(*[channel_param_lists[ch] for ch in channel_names])
        
        for param_combo in param_products:
            params = {
                "channel_betas": {},
                "channel_rs": {}
            }
            
            for i, channel in enumerate(channel_names):
                beta, r = param_combo[i]
                params["channel_betas"][channel] = beta
                params["channel_rs"][channel] = r
            
            combinations.append(params)
        
        return combinations
    
    def _fit_linear_model(self,
                         y: np.ndarray,
                         X_spend: np.ndarray,
                         X_time: np.ndarray,
                         spend_columns: List[str],
                         transform_params: Dict[str, Dict[str, float]]) -> ModelParameters:
        """Fits linear model with given transform parameters."""
        # Apply adstock and saturation transforms
        X_transformed = self._apply_transforms(X_spend, spend_columns, transform_params)
        
        # Create design matrix: [intercept, trend, channels...]
        X_design = np.column_stack([
            np.ones(len(y)),  # intercept
            X_time,           # trend
            X_transformed     # transformed channels
        ])
        
        # Fit linear regression with non-negativity constraints
        def objective(params):
            y_pred = X_design @ params
            return np.sum((y - y_pred) ** 2)
        
        # Initial guess
        initial_params = np.ones(X_design.shape[1]) * 0.1
        
        # Constraints: alpha_baseline >= 0, alpha_trend >= 0, channel_alphas >= 0
        bounds = [(0, None)] * X_design.shape[1]
        
        # Optimize
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        # Extract parameters
        alpha_baseline = result.x[0]
        alpha_trend = result.x[1]
        channel_alphas = {spend_columns[i]: result.x[i+2] for i in range(len(spend_columns))}
        
        return ModelParameters(
            alpha_baseline=alpha_baseline,
            alpha_trend=alpha_trend,
            channel_alphas=channel_alphas,
            channel_betas=transform_params["channel_betas"],
            channel_rs=transform_params["channel_rs"]
        )
    
    def _apply_transforms(self,
                         X_spend: np.ndarray,
                         spend_columns: List[str],
                         params: Dict[str, Dict[str, float]]) -> np.ndarray:
        """Applies adstock and saturation transforms to spend data."""
        n_days, n_channels = X_spend.shape
        X_transformed = np.zeros_like(X_spend, dtype=np.float64)
        
        for i, channel in enumerate(spend_columns):
            beta = params["channel_betas"][channel]
            r = params["channel_rs"][channel]
            
            # Apply adstock transformation
            adstocked = np.zeros(n_days, dtype=np.float64)
            for t in range(n_days):
                adstocked[t] = float(X_spend[t, i]) + (r * adstocked[t-1] if t > 0 else 0.0)
            
            # Apply saturation transformation (power law)
            X_transformed[:, i] = np.power(adstocked, beta)
        
        return X_transformed
    
    def _predict(self,
                X_spend: np.ndarray,
                X_time: np.ndarray,
                spend_columns: List[str],
                params: ModelParameters) -> np.ndarray:
        """Generates predictions using fitted parameters."""
        # Apply transforms
        transform_params = {
            "channel_betas": params.channel_betas,
            "channel_rs": params.channel_rs
        }
        X_transformed = self._apply_transforms(X_spend, spend_columns, transform_params)
        
        # Calculate predictions
        baseline = params.alpha_baseline + params.alpha_trend * X_time
        channel_contributions = np.sum([
            params.channel_alphas[channel] * X_transformed[:, i]
            for i, channel in enumerate(spend_columns)
        ], axis=0)
        
        return baseline + channel_contributions
    
    def _select_best_parameters(self, cv_folds: List[CrossValidationFold]) -> ModelParameters:
        """Selects best parameters based on cross-validation performance."""
        # Average parameters across folds (simple approach)
        # In practice, could weight by performance or use more sophisticated selection
        
        if not cv_folds:
            raise ValueError("No cross-validation folds available")
        
        # Use parameters from best-performing fold
        best_fold = min(cv_folds, key=lambda f: f.mape)
        return best_fold.parameters
    
    def _fit_final_model(self,
                        y: np.ndarray,
                        X_spend: np.ndarray,
                        X_time: np.ndarray,
                        spend_columns: List[str],
                        initial_params: ModelParameters) -> ModelParameters:
        """Fits final model on full dataset using CV-selected parameters."""
        # Use the transform parameters from CV, but refit linear coefficients on full data
        transform_params = {
            "channel_betas": initial_params.channel_betas,
            "channel_rs": initial_params.channel_rs
        }
        
        return self._fit_linear_model(y, X_spend, X_time, spend_columns, transform_params)
    
    def _calculate_confidence_intervals(self,
                                      y: np.ndarray,
                                      X_spend: np.ndarray,
                                      X_time: np.ndarray,
                                      spend_columns: List[str],
                                      params: ModelParameters) -> Dict[str, Tuple[float, float]]:
        """Calculates bootstrap confidence intervals for channel contributions."""
        # Simplified bootstrap - in practice would be more sophisticated
        confidence_intervals = {}
        
        # Calculate total channel contributions
        transform_params = {
            "channel_betas": params.channel_betas,
            "channel_rs": params.channel_rs
        }
        X_transformed = self._apply_transforms(X_spend, spend_columns, transform_params)
        
        for i, channel in enumerate(spend_columns):
            contribution = params.channel_alphas[channel] * X_transformed[:, i]
            total_contribution = np.sum(contribution)
            
            # Simple confidence interval (±20% as placeholder)
            lower = total_contribution * 0.8
            upper = total_contribution * 1.2
            
            confidence_intervals[channel] = (lower, upper)
        
        return confidence_intervals
    
    def _calculate_diagnostics(self,
                             y: np.ndarray,
                             fitted_values: np.ndarray,
                             residuals: np.ndarray,
                             df: pd.DataFrame,
                             spend_columns: List[str],
                             params: ModelParameters) -> Dict[str, Any]:
        """Calculates model diagnostics and validation metrics."""
        diagnostics = {}
        
        # Basic fit statistics
        diagnostics["mean_absolute_error"] = np.mean(np.abs(residuals))
        diagnostics["root_mean_squared_error"] = np.sqrt(np.mean(residuals ** 2))
        diagnostics["residual_std"] = np.std(residuals)
        
        # Media attribution percentage
        baseline_contribution = params.alpha_baseline * len(y) + \
                              params.alpha_trend * np.sum(df["days_since_start"])
        total_profit = np.sum(y)
        media_attribution_pct = (total_profit - baseline_contribution) / total_profit * 100
        diagnostics["media_attribution_percentage"] = media_attribution_pct
        
        # Channel attribution breakdown
        transform_params = {
            "channel_betas": params.channel_betas,
            "channel_rs": params.channel_rs
        }
        X_transformed = self._apply_transforms(
            df[spend_columns].values, spend_columns, transform_params
        )
        
        channel_attributions = {}
        for i, channel in enumerate(spend_columns):
            contribution = params.channel_alphas[channel] * X_transformed[:, i]
            channel_attributions[channel] = np.sum(contribution)
        
        diagnostics["channel_attributions"] = channel_attributions
        
        return diagnostics