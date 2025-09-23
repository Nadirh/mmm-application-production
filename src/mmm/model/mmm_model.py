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
import time
import asyncio
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
    cv_structure_info: Optional[Dict[str, Any]] = None  # CV structure details


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

        # Calculate weeks per fold (ensure integers)
        weeks_per_fold = int(n_weeks // n_folds)
        remainder = int(n_weeks % n_folds)

        folds = []
        start_week = 0

        for i in range(n_folds):
            # Add remainder weeks to last fold
            fold_weeks = int(weeks_per_fold + (remainder if i == n_folds - 1 else 0))

            # Determine outer test size (15-25% of fold, 2-4 weeks max)
            outer_test = int(min(4, max(2, fold_weeks // 5)))
            outer_train = int(fold_weeks - outer_test)

            # Inner split: approximately 70/30 train/test
            inner_train = int(max(5, (outer_train * 7) // 10))
            inner_test = int(outer_train - inner_train)

            folds.append({
                'fold_num': i + 1,
                'start_week': int(start_week),
                'end_week': int(start_week + fold_weeks - 1),
                'total_weeks': int(fold_weeks),
                'outer_train_weeks': int(outer_train),
                'outer_test_weeks': int(outer_test),
                'inner_train_weeks': int(inner_train),
                'inner_test_weeks': int(inner_test),
                'start_day': int(start_week * 7),
                'end_day': int((start_week + fold_weeks) * 7 - 1)
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

        # Reserve last 4 weeks (28 days) for final holdout validation
        # This is NEVER used during CV or parameter selection
        n_total_days = len(df)
        n_holdout_days = int(min(28, n_total_days // 10))  # 4 weeks or 10% of data, whichever is smaller

        if n_total_days < 56:  # Less than 8 weeks total
            logger.warning(f"Only {n_total_days} days available. Need at least 56 days for holdout validation.")
            n_holdout_days = 0  # Skip holdout if insufficient data

        # Split data into CV portion and holdout
        if n_holdout_days > 0:
            n_cv_days = int(n_total_days - n_holdout_days)

            # CV data (used for cross-validation and parameter selection)
            df_cv = df.iloc[:n_cv_days].copy()
            y = df_cv["profit"].values
            X_spend = df_cv[spend_columns].values
            X_time = df_cv["days_since_start"].values

            # Holdout data (used ONLY for final validation)
            df_holdout = df.iloc[n_cv_days:].copy()
            y_holdout = df_holdout["profit"].values
            X_spend_holdout = df_holdout[spend_columns].values
            X_time_holdout = df_holdout["days_since_start"].values

            logger.info(f"Data split: {n_cv_days} days for CV, {n_holdout_days} days for final holdout")
        else:
            # Use all data for CV if insufficient for holdout
            df_cv = df  # Use the entire dataframe
            y = df["profit"].values
            X_spend = df[spend_columns].values
            X_time = df["days_since_start"].values
            y_holdout = None
            X_spend_holdout = None
            X_time_holdout = None
            n_holdout_days = 0

        # Calculate data size and fold structure (using CV data only)
        n_days = len(y)
        n_weeks = n_days // 7
        cv_structure_info = None  # Will store CV structure for results

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
                    "holdout_days": n_holdout_days if n_holdout_days > 0 else 0,
                    "cv_days": n_days,
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
                cv_structure_info = fold_info  # Store for results
                logger.info(f"Nested CV Structure: {n_weeks} weeks → {cv_structure['n_outer_folds']} outer folds")

            # Perform nested cross-validation
            cv_folds = self._perform_nested_cross_validation(
                y, X_spend, X_time, spend_columns, channel_grids,
                cv_structure, progress_callback, cancellation_check
            )
        else:
            # Fall back to simple CV for less data or if disabled
            if progress_callback:
                simple_cv_info = {
                    "type": "cv_structure",
                    "message": f"Using simple CV (data has {n_weeks} weeks, nested CV requires 26+)",
                    "total_weeks": n_weeks,
                    "total_days": n_days,  # Add total_days like nested CV
                    "holdout_days": n_holdout_days if n_holdout_days > 0 else 0,
                    "cv_days": n_days,
                    "method": "simple"
                }
                progress_callback(simple_cv_info)
                cv_structure_info = simple_cv_info  # Store for results

            cv_folds = self._perform_cross_validation(
                y, X_spend, X_time, spend_columns, channel_grids, progress_callback, cancellation_check
        )
        
        # Select best parameters based on CV performance (averages top folds)
        best_params, n_folds_averaged = self._select_best_parameters(cv_folds)

        # Report final parameter selection to dashboard
        if progress_callback:
            # Get MAPEs for all folds for reporting
            fold_mapes = [fold.mape for fold in cv_folds]
            sorted_folds = sorted(cv_folds, key=lambda f: f.mape)
            top_folds = sorted_folds[:n_folds_averaged]

            progress_callback({
                "type": "parameter_selection_complete",
                "all_fold_mapes": fold_mapes,
                "folds_averaged": n_folds_averaged,
                "top_fold_numbers": [f.fold_number for f in top_folds],
                "top_fold_mapes": [f.mape for f in top_folds],
                "final_parameters": {
                    "alpha_baseline": best_params.alpha_baseline,
                    "alpha_trend": best_params.alpha_trend,
                    "channel_alphas": best_params.channel_alphas,
                    "channel_betas": best_params.channel_betas,
                    "channel_rs": best_params.channel_rs
                }
            })

        # Fit final model on CV data (not including holdout)
        final_params = self._fit_final_model(y, X_spend, X_time, spend_columns, best_params)

        # Calculate fitted values and metrics on CV data
        fitted_values = self._predict(X_spend, X_time, spend_columns, final_params)
        residuals = y - fitted_values
        r_squared = 1 - np.var(residuals) / np.var(y)
        mape = mean_absolute_percentage_error(y, fitted_values) * 100
        cv_mape = np.mean([fold.mape for fold in cv_folds])

        # Perform final holdout validation if we have holdout data
        holdout_mape = None
        if y_holdout is not None:
            logger.info(f"Performing final holdout validation on {n_holdout_days} days")

            # Predict on holdout set using final model parameters
            y_pred_holdout = self._predict(X_spend_holdout, X_time_holdout, spend_columns, final_params)
            holdout_mape = mean_absolute_percentage_error(y_holdout, y_pred_holdout) * 100

            logger.info(f"Holdout validation MAPE: {holdout_mape:.2f}% (CV MAPE: {cv_mape:.2f}%)")

            # Report holdout results to dashboard
            if progress_callback:
                progress_callback({
                    "type": "holdout_validation_complete",
                    "holdout_days": n_holdout_days,
                    "holdout_mape": holdout_mape,
                    "cv_mape": cv_mape,
                    "mape_difference": holdout_mape - cv_mape,
                    "is_overfit": bool(holdout_mape > cv_mape * 1.2)  # Flag if holdout is 20% worse than CV
                })

        # Bootstrap confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            y, X_spend, X_time, spend_columns, final_params
        )

        # Calculate diagnostics (use CV data for diagnostics)
        diagnostics = self._calculate_diagnostics(y, fitted_values, residuals, df_cv if n_holdout_days > 0 else df, spend_columns, final_params)

        # Add holdout information to diagnostics
        if holdout_mape is not None:
            diagnostics["holdout_validation"] = {
                "holdout_mape": holdout_mape,
                "cv_mape": cv_mape,
                "holdout_days": n_holdout_days,
                "overfit_warning": bool(holdout_mape > cv_mape * 1.2)
            }
        
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
            n_folds_averaged=n_folds_averaged,
            cv_structure_info=cv_structure_info
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

            # Define outer fold boundaries (in days) - ensure integers
            outer_train_start = int(fold_config['start_day'])
            outer_train_end = int(outer_train_start + (fold_config['outer_train_weeks'] * 7) - 1)
            outer_test_start = int(outer_train_end + 1)
            outer_test_end = int(fold_config['end_day'])

            # Extract outer training data
            outer_train_mask = slice(outer_train_start, outer_train_end + 1)
            y_outer_train = y[outer_train_mask]
            X_spend_outer_train = X_spend[outer_train_mask]
            X_time_outer_train = X_time[outer_train_mask]

            # Define inner fold boundaries - ensure integers
            inner_train_end = int((fold_config['inner_train_weeks'] * 7) - 1)
            inner_test_start = int(inner_train_end + 1)
            inner_test_end = int((fold_config['outer_train_weeks'] * 7) - 1)

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
                fold_num - 1,
                cancellation_check
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
                # Send detailed fold results including all parameters
                progress_callback({
                    "type": "outer_fold_complete",
                    "fold": fold_num,
                    "mape": fold_mape,
                    "parameters": {
                        "alpha_baseline": final_params.alpha_baseline,
                        "alpha_trend": final_params.alpha_trend,
                        "channel_alphas": final_params.channel_alphas,
                        "channel_betas": final_params.channel_betas,
                        "channel_rs": final_params.channel_rs
                    }
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
            
            train_start = int(fold_start)
            train_end = int(fold_start + self.training_window_days)
            test_start = int(train_end)
            test_end = int(test_start + self.test_window_days)
            
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
                progress_callback, fold_idx, cancellation_check
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
                    "mape": fold_mape,
                    "parameters": {
                        "alpha_baseline": best_params.alpha_baseline,
                        "alpha_trend": best_params.alpha_trend,
                        "channel_alphas": best_params.channel_alphas,
                        "channel_betas": best_params.channel_betas,
                        "channel_rs": best_params.channel_rs
                    }
                })
        
        return folds
    
    def _optimize_fold_parameters(self,
                                 y: np.ndarray,
                                 X_spend: np.ndarray,
                                 X_time: np.ndarray,
                                 spend_columns: List[str],
                                 channel_grids: Dict[str, Dict[str, List[float]]],
                                 progress_callback: Optional[callable] = None,
                                 fold_idx: int = 0,
                                 cancellation_check: Optional[callable] = None) -> ModelParameters:
        """Optimizes parameters for a single fold using Bayesian optimization for all channel counts."""
        # Always use Bayesian optimization
        return self._optimize_fold_parameters_bayesian(
            y, X_spend, X_time, spend_columns, channel_grids,
            progress_callback, fold_idx, cancellation_check
        )
    
    def _generate_parameter_combinations(self,
                                       channel_grids: Dict[str, Dict[str, List[float]]],
                                       spend_columns: List[str]) -> List[Dict[str, Dict[str, float]]]:
        """Generates all parameter combinations for grid search."""
        # Calculate total combinations first
        total_combos = 1
        for channel in spend_columns:
            betas = channel_grids[channel]["beta"]
            rs = channel_grids[channel]["r"]
            total_combos *= len(betas) * len(rs)

        # If too many combinations, sample randomly
        MAX_COMBINATIONS = 500000  # Allow up to 500k combinations for 4 channels (390,625)
        if total_combos > MAX_COMBINATIONS:
            logger.warning(f"Total combinations ({total_combos}) exceeds maximum ({MAX_COMBINATIONS}). Random sampling will be used.")
            return self._generate_sampled_combinations(channel_grids, spend_columns, 10000)  # Sample 10k if over limit

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

    def _generate_sampled_combinations(self,
                                       channel_grids: Dict[str, Dict[str, List[float]]],
                                       spend_columns: List[str],
                                       n_samples: int) -> List[Dict[str, Dict[str, float]]]:
        """Generates a random sample of parameter combinations for very large grids."""
        import random
        random.seed(42)  # For reproducibility

        combinations = []
        for _ in range(n_samples):
            params = {
                "channel_betas": {},
                "channel_rs": {}
            }
            for channel in spend_columns:
                params["channel_betas"][channel] = random.choice(channel_grids[channel]["beta"])
                params["channel_rs"][channel] = random.choice(channel_grids[channel]["r"])
            combinations.append(params)

        return combinations

    def _optimize_fold_parameters_bayesian(self,
                                          y: np.ndarray,
                                          X_spend: np.ndarray,
                                          X_time: np.ndarray,
                                          spend_columns: List[str],
                                          channel_grids: Dict[str, Dict[str, List[float]]],
                                          progress_callback: Optional[callable] = None,
                                          fold_idx: int = 0,
                                          cancellation_check: Optional[callable] = None) -> ModelParameters:
        """Optimizes parameters for a single fold using Bayesian optimization."""
        try:
            import optuna
            # Suppress Optuna logging except for errors
            optuna.logging.set_verbosity(optuna.logging.ERROR)
        except ImportError:
            raise ValueError("Optuna not installed. Please install with: pip install optuna")

        # Calculate number of trials based on channel count (doubled for better optimization)
        n_channels = len(spend_columns)
        if n_channels <= 3:
            n_trials = min(2000, 200 * n_channels)  # 200-600 trials for 1-3 channels
        else:
            n_trials = min(2000, 200 * n_channels)  # 800+ trials for 4+ channels

        # Calculate Sobol/TPE split (20% Sobol, 80% TPE)
        n_sobol_trials = int(n_trials * 0.2)
        n_tpe_trials = n_trials - n_sobol_trials

        logger.info(f"Starting hybrid optimization for fold {fold_idx + 1}: "
                   f"{n_sobol_trials} Sobol + {n_tpe_trials} TPE trials for {n_channels} channels")

        # Track best result
        best_mape = float('inf')
        best_params = None
        last_progress_time = time.time()

        # Create Optuna study with hybrid sampler approach
        # We'll run two separate optimization phases: Sobol then TPE
        from optuna.samplers import QMCSampler

        # Keep track of all trials across both phases
        all_trials_count = 0

        # Define objective function
        def objective(trial):
            nonlocal best_mape, best_params, last_progress_time, all_trials_count

            # Check for cancellation
            if cancellation_check and cancellation_check():
                logger.info(f"Training cancelled during optimization at trial {all_trials_count}")
                raise optuna.exceptions.OptunaError("Training cancelled by user")

            # Sample parameters for each channel
            params = {
                "channel_betas": {},
                "channel_rs": {}
            }

            for channel in spend_columns:
                # Use full range for both beta and r parameters
                params["channel_betas"][channel] = trial.suggest_float(
                    f'{channel}_beta', 0.01, 0.99
                )
                params["channel_rs"][channel] = trial.suggest_float(
                    f'{channel}_r', 0.01, 0.99
                )

            try:
                # Fit linear model with these transform parameters
                fitted_params = self._fit_linear_model(y, X_spend, X_time, spend_columns, params)

                # Calculate predictions and MAPE
                y_pred = self._predict(X_spend, X_time, spend_columns, fitted_params)
                mape = mean_absolute_percentage_error(y, y_pred) * 100

                # Track best result
                if mape < best_mape:
                    best_mape = mape
                    best_params = fitted_params
                    logger.info(f"Fold {fold_idx + 1}, Trial {trial.number}: New best MAPE = {mape:.2f}%")

                # Send progress update every 10 seconds
                current_time = time.time()
                if progress_callback and current_time - last_progress_time >= 10:
                    # Indicate whether in Sobol or TPE phase
                    optimization_phase = "sobol" if all_trials_count <= n_sobol_trials else "tpe"
                    progress_callback({
                        "type": "bayesian_optimization",
                        "fold": fold_idx + 1,
                        "trial": all_trials_count,
                        "total_trials": n_trials,
                        "phase": optimization_phase,
                        "sobol_trials": n_sobol_trials,
                        "tpe_trials": n_tpe_trials,
                        "best_mape": best_mape if best_mape != float('inf') else None,
                        "current_mape": mape,
                        "current_params": params
                    })
                    last_progress_time = current_time

                # Increment total counter
                all_trials_count += 1

                # Log progress every 10 trials
                if all_trials_count % 10 == 0:
                    optimization_phase = "Sobol" if all_trials_count <= n_sobol_trials else "TPE"
                    logger.info(
                        f"Fold {fold_idx + 1}, {optimization_phase} Trial {all_trials_count}/{n_trials}: "
                        f"Current MAPE = {mape:.2f}%, Best = {best_mape:.2f}%"
                    )

                return mape

            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {str(e)}")
                return float('inf')

        try:
            # Phase 1: Run Sobol sampling
            sobol_sampler = QMCSampler(
                qmc_type="sobol",
                scramble=True,
                seed=42
            )

            sobol_study = optuna.create_study(
                direction="minimize",
                sampler=sobol_sampler
            )

            logger.info(f"Starting Sobol phase with {n_sobol_trials} trials")
            sobol_study.optimize(objective, n_trials=n_sobol_trials)

            # Phase 2: Run TPE optimization
            if n_tpe_trials > 0:
                tpe_sampler = optuna.samplers.TPESampler(
                    seed=42,
                    n_startup_trials=10,  # Use a small number of startup trials
                    n_ei_candidates=50,
                    gamma=lambda x: int(x * 0.25),  # Ensure gamma returns integer
                    multivariate=True,
                    constant_liar=True
                )

                tpe_study = optuna.create_study(
                    direction="minimize",
                    sampler=tpe_sampler
                )

                logger.info(f"Starting TPE phase with {n_tpe_trials} additional trials")
                tpe_study.optimize(objective, n_trials=n_tpe_trials)

                # Use TPE study which has the best results from both phases
                study = tpe_study
            else:
                study = sobol_study

            # Send final progress update
            if progress_callback:
                progress_callback({
                    "type": "bayesian_optimization_complete",
                    "fold": fold_idx + 1,
                    "trials_completed": len(study.trials),
                    "sobol_trials": n_sobol_trials,
                    "tpe_trials": n_tpe_trials,
                    "best_mape": best_mape,
                    "best_params": {
                        "channel_betas": best_params.channel_betas,
                        "channel_rs": best_params.channel_rs
                    } if best_params else None
                })

            # Check if we found valid parameters
            if best_params is None:
                raise ValueError(f"Hybrid Sobol/TPE optimization failed to find valid parameters for fold {fold_idx + 1}")

            logger.info(
                f"Hybrid optimization complete for fold {fold_idx + 1}: "
                f"Best MAPE = {best_mape:.2f}% from {len(study.trials)} trials"
            )

            return best_params

        except optuna.exceptions.OptunaError as e:
            if "cancelled" in str(e).lower():
                raise InterruptedError("Training cancelled by user")
            raise ValueError(f"Bayesian optimization failed: {str(e)}")
        except Exception as e:
            logger.error(f"Bayesian optimization failed for fold {fold_idx + 1}: {str(e)}")
            raise ValueError(f"Optimization failed: {str(e)}")

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

        # Calculate statistics to detect outliers
        all_mapes = [f.mape for f in cv_folds]
        mean_mape = np.mean(all_mapes)
        std_mape = np.std(all_mapes)

        # Strategy for handling divergent folds:
        # 1. If standard deviation is high (>50% of mean), be more selective
        # 2. Check for outliers using z-score
        high_variance = std_mape > (mean_mape * 0.5) if mean_mape > 0 else False

        if high_variance and len(sorted_folds) >= 2:
            # High variance detected - be more selective
            logger.warning(f"High variance in fold MAPEs detected (mean: {mean_mape:.2f}%, std: {std_mape:.2f}%)")

            # Use only folds within 1.5 standard deviations of the best fold
            best_mape = sorted_folds[0].mape
            threshold = best_mape + (1.5 * std_mape)

            top_folds = [f for f in sorted_folds if f.mape <= threshold]

            # Ensure we have at least 1 fold but not more than top 50%
            if len(top_folds) == 0:
                top_folds = [sorted_folds[0]]  # Use only the best fold
            elif len(top_folds) > len(sorted_folds) // 2:
                top_folds = sorted_folds[:len(sorted_folds) // 2]

            n_folds_to_average = len(top_folds)
            logger.info(f"High variance strategy: Using {n_folds_to_average} folds within threshold (MAPEs: {[f.mape for f in top_folds]})")
        else:
            # Normal variance - use standard strategy (top 30% or minimum 3)
            n_folds_to_average = max(3, len(sorted_folds) // 3)
            n_folds_to_average = min(n_folds_to_average, len(sorted_folds))
            top_folds = sorted_folds[:n_folds_to_average]
            logger.info(f"Standard strategy: Averaging parameters from top {n_folds_to_average} folds (MAPEs: {[f.mape for f in top_folds]})")

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