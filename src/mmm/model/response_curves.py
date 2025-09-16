"""
Response curve generation for Media Mix Model.

This module generates response curves that show the relationship between
media spend and incremental profit for each channel.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import structlog
from scipy import optimize
import asyncio

from mmm.utils.cache import cache_manager
from mmm.config.settings import settings

logger = structlog.get_logger()


class ResponseCurveGenerator:
    """Generates response curves for MMM model results."""
    
    def __init__(self, model_parameters: Dict[str, Any]):
        """Initialize with trained model parameters."""
        self.model_parameters = model_parameters
        self.channel_alphas = model_parameters.get("channel_alphas", {})
        self.channel_betas = model_parameters.get("channel_betas", {})
        self.channel_rs = model_parameters.get("channel_rs", {})
        self.alpha_baseline = model_parameters.get("alpha_baseline", 1000.0)
    
    def generate_response_curve(self,
                              channel: str,
                              min_spend: float = 0,
                              max_spend: float = None,
                              num_points: int = 100,
                              current_spend: float = None,
                              include_confidence_intervals: bool = True,
                              avg_daily_spend_28d: float = None) -> Dict[str, Any]:
        """
        Generate response curve for a specific channel.
        
        Args:
            channel: Channel name
            min_spend: Minimum spend level
            max_spend: Maximum spend level (auto-calculated if None)
            num_points: Number of points in the curve
            current_spend: Current spend level for highlighting
            
        Returns:
            Dictionary containing curve data and metrics
        """
        if channel not in self.channel_alphas:
            raise ValueError(f"Channel {channel} not found in model parameters")
        
        # Get model parameters for this channel
        alpha = self.channel_alphas[channel]
        beta = self.channel_betas[channel]
        r = self.channel_rs[channel]
        
        # Auto-calculate max spend if not provided
        if max_spend is None:
            if avg_daily_spend_28d and isinstance(avg_daily_spend_28d, (int, float)) and avg_daily_spend_28d > 0:
                max_spend = avg_daily_spend_28d * 2  # 2x 28-day average spend
            elif current_spend and isinstance(current_spend, (int, float)) and current_spend > 0:
                max_spend = current_spend * 3  # 3x current spend
            else:
                max_spend = 50000  # Default max spend
        
        # Generate spend levels
        spend_levels = np.linspace(min_spend, max_spend, num_points)
        
        # Calculate incremental profit for each spend level
        incremental_profits = []
        marginal_roas = []
        confidence_intervals = {"lower": [], "upper": []} if include_confidence_intervals else None

        for spend in spend_levels:
            # Apply adstock transformation (simplified for curve generation)
            adstocked_spend = spend / (1 - r) if r < 1 else spend

            # Apply saturation transformation
            saturated_spend = np.power(adstocked_spend, beta) if beta > 0 else adstocked_spend

            # Calculate incremental profit
            incremental_profit = alpha * saturated_spend
            incremental_profits.append(incremental_profit)

            # Calculate marginal ROAS (derivative)
            if spend > 0:
                marginal_roas.append(self._calculate_marginal_roas(spend, alpha, beta, r))
            else:
                marginal_roas.append(0)

            # Calculate 95% confidence intervals if requested
            if include_confidence_intervals:
                ci_lower, ci_upper = self._calculate_confidence_intervals(
                    incremental_profit, alpha, beta, r
                )
                confidence_intervals["lower"].append(ci_lower)
                confidence_intervals["upper"].append(ci_upper)
        
        # Find key points
        saturation_point = self._find_saturation_point(spend_levels, incremental_profits)
        optimal_spend = self._find_optimal_spend(spend_levels, marginal_roas, target_roas=1.0)
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            spend_levels, incremental_profits, current_spend
        )
        
        result = {
            "channel": channel,
            "spend_levels": spend_levels.tolist(),
            "incremental_profits": incremental_profits,
            "marginal_roas": marginal_roas,
            "saturation_point": saturation_point,
            "optimal_spend": optimal_spend,
            "current_spend": current_spend,
            "efficiency_metrics": efficiency_metrics,
            "model_parameters": {
                "alpha": alpha,
                "beta": beta,
                "r": r
            }
        }

        # Add confidence intervals if calculated
        if confidence_intervals:
            result["confidence_intervals"] = confidence_intervals

        return result
    
    def generate_all_response_curves(self,
                                   current_spend: Dict[str, float] = None,
                                   spend_multiplier: float = 2.0,
                                   num_points: int = 100) -> Dict[str, Dict[str, Any]]:
        """Generate response curves for all channels."""
        curves = {}
        current_spend = current_spend or {}
        
        for channel in self.channel_alphas.keys():
            try:
                channel_current = current_spend.get(channel) if current_spend else None
                # Ensure channel_current is a number, not a dict
                if isinstance(channel_current, dict):
                    channel_current = None

                max_spend = channel_current * spend_multiplier if channel_current and isinstance(channel_current, (int, float)) else None

                curve = self.generate_response_curve(
                    channel=channel,
                    max_spend=max_spend,
                    num_points=num_points,
                    current_spend=channel_current,
                    avg_daily_spend_28d=channel_current
                )
                curves[channel] = curve
                
            except Exception as e:
                logger.warning("Failed to generate response curve", 
                             channel=channel, error=str(e))
        
        return curves
    
    def _calculate_marginal_roas(self, spend: float, alpha: float, beta: float, r: float) -> float:
        """Calculate marginal ROAS at a given spend level."""
        if spend <= 0:
            return 0
        
        # Derivative of the adstock and saturation function
        # d/dx[alpha * (x/(1-r))^beta] = alpha * beta * (1/(1-r))^beta * x^(beta-1)
        adstock_factor = 1 / (1 - r) if r < 1 else 1
        marginal_profit = alpha * beta * (adstock_factor ** beta) * (spend ** (beta - 1))
        
        return marginal_profit / spend if spend > 0 else 0

    def _calculate_confidence_intervals(self, incremental_profit: float,
                                      alpha: float, beta: float, r: float) -> Tuple[float, float]:
        """Calculate 95% confidence intervals for the incremental profit prediction."""
        # Simulate parameter uncertainty (simplified approach)
        # In practice, you'd use the actual parameter covariance matrix from model fitting

        # Assume 10% uncertainty in alpha (main driver of confidence intervals)
        alpha_std = alpha * 0.1

        # Assume 5% uncertainty in beta and r
        beta_std = beta * 0.05
        r_std = r * 0.05

        # Use normal approximation with 1.96 * std for 95% CI
        z_score = 1.96

        # Calculate uncertainty in incremental profit
        # Main uncertainty comes from alpha parameter
        profit_std = incremental_profit * (alpha_std / alpha) if alpha > 0 else incremental_profit * 0.1

        # Add small amount of uncertainty from beta and r parameters
        profit_std += incremental_profit * 0.02  # Additional 2% uncertainty

        ci_lower = incremental_profit - z_score * profit_std
        ci_upper = incremental_profit + z_score * profit_std

        # Ensure lower bound is non-negative
        ci_lower = max(0, ci_lower)

        return ci_lower, ci_upper

    def _find_saturation_point(self, spend_levels: np.ndarray, profits: List[float]) -> Dict[str, float]:
        """Find the saturation point where marginal returns significantly diminish."""
        profits_array = np.array(profits)
        
        # Calculate second derivative (acceleration)
        if len(profits_array) < 3:
            return {"spend": spend_levels[-1], "profit": profits[-1]}
        
        second_derivative = np.gradient(np.gradient(profits_array))
        
        # Find where second derivative becomes most negative (inflection point)
        min_accel_idx = np.argmin(second_derivative)
        
        return {
            "spend": spend_levels[min_accel_idx],
            "profit": profits[min_accel_idx]
        }
    
    def _find_optimal_spend(self, spend_levels: np.ndarray, 
                          marginal_roas: List[float], target_roas: float = 1.0) -> Dict[str, float]:
        """Find optimal spend where marginal ROAS equals target."""
        roas_array = np.array(marginal_roas)
        
        # Find closest point to target ROAS
        diff = np.abs(roas_array - target_roas)
        optimal_idx = np.argmin(diff)
        
        return {
            "spend": spend_levels[optimal_idx],
            "marginal_roas": marginal_roas[optimal_idx]
        }
    
    def _calculate_efficiency_metrics(self, spend_levels: np.ndarray, 
                                    profits: List[float], 
                                    current_spend: Optional[float]) -> Dict[str, Any]:
        """Calculate efficiency metrics for the channel."""
        profits_array = np.array(profits)
        
        # Overall ROAS curve
        roas_curve = profits_array / spend_levels
        roas_curve[spend_levels == 0] = 0  # Handle division by zero
        
        # Peak efficiency point
        max_roas_idx = np.argmax(roas_curve[spend_levels > 0])
        peak_efficiency = {
            "spend": spend_levels[max_roas_idx],
            "roas": roas_curve[max_roas_idx],
            "profit": profits[max_roas_idx]
        }
        
        metrics = {
            "peak_efficiency": peak_efficiency,
            "max_profit": {
                "spend": spend_levels[-1],
                "profit": profits[-1],
                "roas": roas_curve[-1]
            }
        }
        
        # Current efficiency if current spend provided
        if current_spend and current_spend > 0:
            # Find closest spend level to current
            current_idx = np.argmin(np.abs(spend_levels - current_spend))
            metrics["current_efficiency"] = {
                "spend": current_spend,
                "profit": profits[current_idx],
                "roas": roas_curve[current_idx]
            }
        
        return metrics


class CachedResponseCurveGenerator:
    """Response curve generator with caching support."""
    
    def __init__(self, run_id: str, model_parameters: Dict[str, Any]):
        self.run_id = run_id
        self.generator = ResponseCurveGenerator(model_parameters)
    
    async def get_response_curve(self, channel: str, **kwargs) -> Dict[str, Any]:
        """Get response curve with caching."""
        # Try to get from cache first
        cached_curve = await cache_manager.get_response_curve(self.run_id, channel)
        
        if cached_curve:
            logger.info("Response curve served from cache", 
                       run_id=self.run_id, channel=channel)
            return cached_curve
        
        # Generate curve
        curve = await asyncio.to_thread(
            self.generator.generate_response_curve, 
            channel, 
            **kwargs
        )
        
        # Cache the result
        await cache_manager.cache_response_curve(self.run_id, channel, curve)
        
        logger.info("Response curve generated and cached", 
                   run_id=self.run_id, channel=channel)
        return curve
    
    async def get_all_response_curves(self, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Get all response curves with caching."""
        curves = {}

        # Extract current_spend if it's a dictionary
        current_spend_dict = kwargs.get('current_spend', {})

        # Get curves for each channel
        for channel in self.generator.channel_alphas.keys():
            try:
                # Extract individual channel value from current_spend dictionary
                channel_kwargs = kwargs.copy()
                if isinstance(current_spend_dict, dict):
                    channel_kwargs['current_spend'] = current_spend_dict.get(channel)
                    channel_kwargs['avg_daily_spend_28d'] = current_spend_dict.get(channel)

                curve = await self.get_response_curve(channel, **channel_kwargs)
                curves[channel] = curve
            except Exception as e:
                logger.warning("Failed to get response curve",
                             channel=channel, error=str(e))

        return curves
    
    async def invalidate_cache(self):
        """Invalidate cached response curves for this run."""
        deleted = await cache_manager.clear_pattern(f"response_curve:*run_id={self.run_id}*")
        logger.info("Response curve cache invalidated", 
                   run_id=self.run_id, deleted_keys=deleted)
        return deleted


def create_response_curve_generator(run_id: str, 
                                  model_parameters: Dict[str, Any]) -> CachedResponseCurveGenerator:
    """Factory function to create a cached response curve generator."""
    return CachedResponseCurveGenerator(run_id, model_parameters)


# Utility functions for response curve analysis

def compare_response_curves(curves: Dict[str, Dict[str, Any]], 
                          metric: str = "peak_efficiency") -> List[Dict[str, Any]]:
    """Compare channels by a specific efficiency metric."""
    comparisons = []
    
    for channel, curve in curves.items():
        efficiency = curve["efficiency_metrics"][metric]
        comparisons.append({
            "channel": channel,
            "metric": metric,
            "value": efficiency,
            **efficiency
        })
    
    # Sort by ROAS descending
    return sorted(comparisons, key=lambda x: x.get("roas", 0), reverse=True)


def calculate_portfolio_efficiency(curves: Dict[str, Dict[str, Any]], 
                                 current_spend: Dict[str, float]) -> Dict[str, Any]:
    """Calculate overall portfolio efficiency metrics."""
    total_spend = sum(current_spend.values())
    total_profit = 0
    weighted_roas = 0
    
    channel_contributions = {}
    
    for channel, curve in curves.items():
        channel_spend = current_spend.get(channel, 0)
        
        if channel_spend > 0 and "current_efficiency" in curve["efficiency_metrics"]:
            current_eff = curve["efficiency_metrics"]["current_efficiency"]
            channel_profit = current_eff["profit"]
            channel_roas = current_eff["roas"]
            
            total_profit += channel_profit
            weighted_roas += channel_roas * (channel_spend / total_spend)
            
            channel_contributions[channel] = {
                "spend": channel_spend,
                "profit": channel_profit,
                "roas": channel_roas,
                "profit_share": channel_profit / total_profit if total_profit > 0 else 0,
                "spend_share": channel_spend / total_spend if total_spend > 0 else 0
            }
    
    return {
        "total_spend": total_spend,
        "total_profit": total_profit,
        "overall_roas": total_profit / total_spend if total_spend > 0 else 0,
        "weighted_avg_roas": weighted_roas,
        "channel_contributions": channel_contributions,
        "efficiency_score": min(weighted_roas, 10.0)  # Cap at 10 for scoring
    }