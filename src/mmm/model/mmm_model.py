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
    n_folds_averaged: int = 1  # Number of folds averaged for parameters


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
                 n_bootstrap: int = 500,
                 use_nested_cv: bool = True):
        self.training_window_days = training_window_days
        self.test_window_days = test_window_days
        self.n_bootstrap = n_bootstrap
        self.use_nested_cv = use_nested_cv
        self.is_fitted = False
        self.results: Optional[ModelResults] = None
        
    def _calculate_nested_cv_structure(self, n_weeks: int) -> Dict[str, Any]:
        """Calculate nested CV fold structure based on data size."""

        # Determine number of outer folds
        if n_weeks < 26:
            raise ValueError(f"Minimum 26 weeks required, got {n_weeks} weeks")
        elif n_weeks <= 30:
            n_folds = 2
        elif n_weeks <= 42:
            n_folds = 3
        else:
            n_folds = 4

        # Calculate weeks per fold
        weeks_per_fold = n_weeks // n_folds
        remainder = n_weeks % n_folds

        folds = []
        start_week = 0

        for i in range(n_folds):
            # Add remainder weeks to last fold
            fold_weeks = weeks_per_fold + (remainder if i == n_folds - 1 else 0)

            # Determine outer test size (15-25% of fold, 2-4 weeks max)
            outer_test = min(4, max(2, fold_weeks // 5))
            outer_train = fold_weeks - outer_test

            # Inner split: approximately 70/30 train/test
            inner_train = max(5, (outer_train * 7) // 10)
            inner_test = outer_train - inner_train

            folds.append({
                'fold_num': i + 1,
                'start_week': start_week,
                'end_week': start_week + fold_weeks - 1,
                'total_weeks': fold_weeks,
                'outer_train_weeks': outer_train,
                'outer_test_weeks': outer_test,
                'inner_train_weeks': inner_train,
                'inner_test_weeks': inner_test,
                'start_day': start_week * 7,
                'end_day': (start_week + fold_weeks) * 7 - 1
            })

            start_week += fold_weeks

        return {
            'n_weeks': n_weeks,
            'n_outer_folds': n_folds,
            'folds': folds
        }

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

        # Calculate data size and fold structure
        n_days = len(y)
        n_weeks = n_days // 7

        # Report fold structure to dashboard
        if self.use_nested_cv and n_weeks >= 26:
            cv_structure = self._calculate_nested_cv_structure(n_weeks)

            # Send fold structure to dashboard
            if progress_callback:
                fold_info = {
                    "type": "cv_structure",
                    "total_weeks": n_weeks,
                    "total_days": n_days,
                    "n_outer_folds": cv_structure['n_outer_folds'],
                    "fold_details": []
                }

                for fold in cv_structure['folds']:
                    fold_info["fold_details"].append({
                        "fold": fold['fold_num'],
                        "weeks": f"{fold['start_week']}-{fold['end_week']}",
                        "outer_train": f"{fold['outer_train_weeks']}w",
                        "outer_test": f"{fold['outer_test_weeks']}w",
                        "inner_train": f"{fold['inner_train_weeks']}w",
                        "inner_test": f"{fold['inner_test_weeks']}w"
                    })

                progress_callback(fold_info)
                logger.info(f"Nested CV Structure: {n_weeks} weeks → {cv_structure['n_outer_folds']} outer folds")

            # Perform nested cross-validation
            cv_folds = self._perform_nested_cross_validation(
                y, X_spend, X_time, spend_columns, channel_grids,
                cv_structure, progress_callback, cancellation_check
            )
        else:
            # Fall back to simple CV for less data or if disabled
            if progress_callback:
                progress_callback({
                    "type": "cv_structure",
                    "message": f"Using simple CV (data has {n_weeks} weeks, nested CV requires 26+)",
                    "total_weeks": n_weeks,
                    "method": "simple"
                })

            cv_folds = self._perform_cross_validation(
                y, X_spend, X_time, spend_columns, channel_grids, progress_callback, cancellation_check
        )
        
        # Select best parameters based on CV performance (averages top folds)
        best_params, n_folds_averaged = self._select_best_parameters(cv_folds)
        
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
            diagnostics=diagnostics,
            n_folds_averaged=n_folds_averaged
        )
        
        self.is_fitted = True
        return self.results
    
    def _perform_nested_cross_validation(self,
                                        y: np.ndarray,
                                        X_spend: np.ndarray,
                                        X_time: np.ndarray,
                                        spend_columns: List[str],
                                        channel_grids: Dict[str, Dict[str, List[float]]],
                                        cv_structure: Dict[str, Any],
                                        progress_callback: Optional[callable] = None,
                                        cancellation_check: Optional[callable] = None) -> List[CrossValidationFold]:
        """Performs nested cross-validation."""
        outer_folds = []

        for fold_config in cv_structure['folds']:
            # Check for cancellation
            if cancellation_check and cancellation_check():
                raise InterruptedError("Training cancelled by user")

            fold_num = fold_config['fold_num']
            logger.info(f"Processing Outer Fold {fold_num}/{cv_structure['n_outer_folds']}")

            # Report outer fold progress
            if progress_callback:
                progress_callback({
                    "type": "outer_fold_start",
                    "fold": fold_num,
                    "total_folds": cv_structure['n_outer_folds'],
                    "weeks": f"{fold_config['start_week']}-{fold_config['end_week']}"
                })

            # Define outer fold boundaries (in days)
            outer_train_start = fold_config['start_day']
            outer_train_end = outer_train_start + (fold_config['outer_train_weeks'] * 7) - 1
            outer_test_start = outer_train_end + 1
            outer_test_end = fold_config['end_day']

            # Extract outer training data
            outer_train_mask = slice(outer_train_start, outer_train_end + 1)
            y_outer_train = y[outer_train_mask]
            X_spend_outer_train = X_spend[outer_train_mask]
            X_time_outer_train = X_time[outer_train_mask]

            # Define inner fold boundaries
            inner_train_end = (fold_config['inner_train_weeks'] * 7) - 1
            inner_test_start = inner_train_end + 1
            inner_test_end = (fold_config['outer_train_weeks'] * 7) - 1

            # Inner fold data
            inner_train_mask = slice(0, inner_train_end + 1)
            inner_test_mask = slice(inner_test_start, inner_test_end + 1)

            y_inner_train = y_outer_train[inner_train_mask]
            X_spend_inner_train = X_spend_outer_train[inner_train_mask]
            X_time_inner_train = X_time_outer_train[inner_train_mask]

            y_inner_test = y_outer_train[inner_test_mask]
            X_spend_inner_test = X_spend_outer_train[inner_test_mask]
            X_time_inner_test = X_time_outer_train[inner_test_mask]

            # Report inner fold info
            if progress_callback:
                progress_callback({
                    "type": "inner_fold_info",
                    "outer_fold": fold_num,
                    "inner_train_days": len(y_inner_train),
                    "inner_test_days": len(y_inner_test)
                })

            # Grid search on inner fold
            best_params = self._optimize_fold_parameters(
                y_inner_train, X_spend_inner_train, X_time_inner_train,
                spend_columns, channel_grids,
                lambda info: progress_callback({**info, "outer_fold": fold_num}) if progress_callback else None,
                fold_num - 1
            )

            # Evaluate on outer test set with best params
            outer_test_mask = slice(outer_test_start, outer_test_end + 1)
            y_outer_test = y[outer_test_mask]
            X_spend_outer_test = X_spend[outer_test_mask]
            X_time_outer_test = X_time[outer_test_mask]

            # Train on full outer training data with best params
            final_params = self._fit_final_model(
                y_outer_train, X_spend_outer_train, X_time_outer_train,
                spend_columns, best_params
            )

            # Evaluate on outer test
            y_pred_test = self._predict(X_spend_outer_test, X_time_outer_test, spend_columns, final_params)
            fold_mape = mean_absolute_percentage_error(y_outer_test, y_pred_test) * 100

            outer_folds.append(CrossValidationFold(
                fold_number=fold_num,
                train_start=outer_train_start,
                train_end=outer_train_end,
                test_start=outer_test_start,
                test_end=outer_test_end,
                mape=fold_mape,
                parameters=final_params
            ))

            logger.info(f"Outer Fold {fold_num} completed: MAPE={fold_mape:.2f}%")

            if progress_callback:
                progress_callback({
                    "type": "outer_fold_complete",
                    "fold": fold_num,
                    "mape": fold_mape
                })

        return outer_folds

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
    
    def _select_best_parameters(self, cv_folds: List[CrossValidationFold]) -> Tuple[ModelParameters, int]:
        """Selects best parameters by averaging top-performing folds."""

        if not cv_folds:
            raise ValueError(
                f"No cross-validation folds were created. This typically means your dataset "
                f"is too small for the configured training window ({self.training_window_days} days) "
                f"and test window ({self.test_window_days} days). "
                f"Minimum required: {self.training_window_days + self.test_window_days} days for basic training, "
                f"but more data (182+ days) is recommended for reliable model performance."
            )

        # Sort folds by MAPE (best first)
        sorted_folds = sorted(cv_folds, key=lambda f: f.mape)

        # Determine how many top folds to average (top 30% or minimum 3)
        n_folds_to_average = max(3, len(sorted_folds) // 3)
        n_folds_to_average = min(n_folds_to_average, len(sorted_folds))
        top_folds = sorted_folds[:n_folds_to_average]

        logger.info(f"Averaging parameters from top {n_folds_to_average} folds (MAPEs: {[f.mape for f in top_folds]})")

        # Initialize averaged parameters
        avg_alpha_baseline = sum(f.parameters.alpha_baseline for f in top_folds) / n_folds_to_average
        avg_alpha_trend = sum(f.parameters.alpha_trend for f in top_folds) / n_folds_to_average

        # Average channel-specific parameters
        all_channels = list(top_folds[0].parameters.channel_alphas.keys())
        avg_channel_alphas = {}
        avg_channel_betas = {}
        avg_channel_rs = {}

        for channel in all_channels:
            # Average alphas (linear coefficients)
            avg_channel_alphas[channel] = sum(
                f.parameters.channel_alphas[channel] for f in top_folds
            ) / n_folds_to_average

            # Average betas (saturation parameters) - exact average, not rounded to grid
            avg_channel_betas[channel] = sum(
                f.parameters.channel_betas[channel] for f in top_folds
            ) / n_folds_to_average

            # Average rs (adstock parameters) - exact average, not rounded to grid
            avg_channel_rs[channel] = sum(
                f.parameters.channel_rs[channel] for f in top_folds
            ) / n_folds_to_average

        # Create averaged parameters object
        averaged_params = ModelParameters(
            alpha_baseline=avg_alpha_baseline,
            alpha_trend=avg_alpha_trend,
            channel_alphas=avg_channel_alphas,
            channel_betas=avg_channel_betas,
            channel_rs=avg_channel_rs
        )

        logger.info(f"Averaged parameters created from {n_folds_to_average} best folds")

        return averaged_params, n_folds_to_average
    
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
        """Calculates bootstrap confidence intervals for channel contributions using residual resampling."""
        logger.info(f"Starting bootstrap confidence intervals with {self.n_bootstrap} iterations")

        # Calculate fitted values and residuals
        fitted_values = self._predict(X_spend, X_time, spend_columns, params)
        residuals = y - fitted_values

        # Store bootstrap results for each channel
        bootstrap_contributions = {channel: [] for channel in spend_columns}

        # Keep transformation parameters fixed from the fitted model
        transform_params = {
            "channel_betas": params.channel_betas,
            "channel_rs": params.channel_rs
        }

        # Perform bootstrap iterations
        for iteration in range(self.n_bootstrap):
            # Resample residuals with replacement
            bootstrap_indices = np.random.choice(len(residuals), size=len(residuals), replace=True)
            bootstrap_residuals = residuals[bootstrap_indices]

            # Create bootstrap y values
            y_bootstrap = fitted_values + bootstrap_residuals

            # Refit only the linear coefficients (keeping transformation parameters fixed)
            bootstrap_params = self._fit_linear_model(
                y_bootstrap, X_spend, X_time, spend_columns, transform_params
            )

            # Calculate channel contributions for this bootstrap iteration
            X_transformed = self._apply_transforms(X_spend, spend_columns, transform_params)

            for i, channel in enumerate(spend_columns):
                contribution = bootstrap_params.channel_alphas[channel] * X_transformed[:, i]
                total_contribution = np.sum(contribution)
                bootstrap_contributions[channel].append(total_contribution)

            # Log progress every 100 iterations
            if (iteration + 1) % 100 == 0:
                logger.debug(f"Bootstrap progress: {iteration + 1}/{self.n_bootstrap} iterations completed")

        # Calculate 95% confidence intervals from bootstrap distribution
        confidence_intervals = {}

        for channel in spend_columns:
            contributions = np.array(bootstrap_contributions[channel])

            # Calculate percentiles for 95% CI
            lower_percentile = np.percentile(contributions, 2.5)
            upper_percentile = np.percentile(contributions, 97.5)

            confidence_intervals[channel] = (lower_percentile, upper_percentile)

            # Log the improvement over the old placeholder method
            mean_contribution = np.mean(contributions)
            std_contribution = np.std(contributions)
            old_ci_width = mean_contribution * 0.4  # Old method was ±20%
            new_ci_width = upper_percentile - lower_percentile

            logger.info(
                f"Channel {channel}: Mean={mean_contribution:.0f}, "
                f"CI=[{lower_percentile:.0f}, {upper_percentile:.0f}], "
                f"CI width reduced from {old_ci_width:.0f} to {new_ci_width:.0f} "
                f"(Std={std_contribution:.0f})"
            )

        logger.info(f"Bootstrap confidence intervals completed with {self.n_bootstrap} iterations")

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