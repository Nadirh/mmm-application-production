"""
Budget optimization module for MMM application.
Handles optimization with business constraints and sensitivity analysis.
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from enum import Enum


class ConstraintType(Enum):
    FLOOR = "floor"       # Minimum spend requirement
    CAP = "cap"          # Maximum spend limit
    LOCK = "lock"        # Fixed spend level
    RAMP = "ramp"        # Maximum change percentage


@dataclass
class Constraint:
    channel: str
    type: ConstraintType
    value: float
    description: str = ""


@dataclass
class OptimizationResult:
    optimal_spend: Dict[str, float]
    optimal_profit: float
    current_profit: float
    profit_uplift: float
    shadow_prices: Dict[str, float]
    constraints_binding: List[str]
    response_curves: Dict[str, Tuple[np.ndarray, np.ndarray]]  # (spend_range, profit_values)
    scenario_analysis: Dict[str, Any]


@dataclass
class ResponseCurve:
    channel: str
    spend_range: np.ndarray
    profit_values: np.ndarray
    current_spend: float
    optimal_spend: float
    marginal_efficiency: np.ndarray


class BudgetOptimizer:
    """Optimizes budget allocation subject to business constraints."""
    
    def __init__(self, 
                 model_params: Any,  # ModelParameters from mmm_model
                 min_spend_resolution: float = 1000.0):
        self.model_params = model_params
        self.min_spend_resolution = min_spend_resolution
        
    def optimize(self,
                current_spend: Dict[str, float],
                total_budget: float,
                constraints: List[Constraint] = None,
                optimization_window_days: int = 365) -> OptimizationResult:
        """
        Optimizes budget allocation to maximize profit.
        
        Args:
            current_spend: Current spend levels by channel
            total_budget: Total budget constraint
            constraints: List of business constraints
            optimization_window_days: Time horizon for optimization
            
        Returns:
            OptimizationResult with optimal allocation and analysis
        """
        constraints = constraints or []
        channels = list(current_spend.keys())
        
        # Set up optimization problem
        bounds, constraint_funcs = self._setup_constraints(
            channels, current_spend, total_budget, constraints
        )
        
        # Objective function (negative profit for minimization)
        def objective(spend_array):
            spend_dict = dict(zip(channels, spend_array))
            return -self._calculate_profit(spend_dict, optimization_window_days)
        
        # Initial guess (current spend)
        x0 = np.array([current_spend[ch] for ch in channels])
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraint_funcs
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        # Extract results
        optimal_spend = dict(zip(channels, result.x))
        optimal_profit = -result.fun
        current_profit = self._calculate_profit(current_spend, optimization_window_days)
        profit_uplift = optimal_profit - current_profit
        
        # Calculate shadow prices
        shadow_prices = self._calculate_shadow_prices(
            optimal_spend, total_budget, constraints, optimization_window_days
        )
        
        # Generate response curves
        response_curves = self._generate_response_curves(channels, optimal_spend)
        
        # Scenario analysis
        scenario_analysis = self._perform_scenario_analysis(
            optimal_spend, total_budget, optimization_window_days
        )
        
        # Identify binding constraints
        constraints_binding = self._identify_binding_constraints(
            optimal_spend, constraints, total_budget
        )
        
        return OptimizationResult(
            optimal_spend=optimal_spend,
            optimal_profit=optimal_profit,
            current_profit=current_profit,
            profit_uplift=profit_uplift,
            shadow_prices=shadow_prices,
            constraints_binding=constraints_binding,
            response_curves=response_curves,
            scenario_analysis=scenario_analysis
        )
    
    def _setup_constraints(self,
                          channels: List[str],
                          current_spend: Dict[str, float],
                          total_budget: float,
                          constraints: List[Constraint]) -> Tuple[List[Tuple], List[Dict]]:
        """Sets up optimization constraints and bounds."""
        bounds = []
        constraint_funcs = []
        
        # Default bounds (non-negative spend)
        for channel in channels:
            bounds.append((0, None))
        
        # Process explicit constraints
        for constraint in constraints:
            if constraint.channel not in channels:
                continue
                
            channel_idx = channels.index(constraint.channel)
            
            if constraint.type == ConstraintType.FLOOR:
                # Update lower bound
                bounds[channel_idx] = (constraint.value, bounds[channel_idx][1])
                
            elif constraint.type == ConstraintType.CAP:
                # Update upper bound
                bounds[channel_idx] = (bounds[channel_idx][0], constraint.value)
                
            elif constraint.type == ConstraintType.LOCK:
                # Fix spend level
                bounds[channel_idx] = (constraint.value, constraint.value)
                
            elif constraint.type == ConstraintType.RAMP:
                # Ramp constraint relative to current spend
                current = current_spend[constraint.channel]
                max_change = constraint.value / 100.0  # Convert percentage
                lower = current * (1 - max_change)
                upper = current * (1 + max_change)
                bounds[channel_idx] = (
                    max(bounds[channel_idx][0], lower),
                    min(bounds[channel_idx][1] or float('inf'), upper)
                )
        
        # Budget constraint
        def budget_constraint(spend_array):
            return total_budget - np.sum(spend_array)
        
        constraint_funcs.append({
            'type': 'eq',
            'fun': budget_constraint
        })
        
        return bounds, constraint_funcs
    
    def _calculate_profit(self, spend_dict: Dict[str, float], days: int) -> float:
        """Calculates profit for given spend allocation over specified days."""
        # Simulate daily profit calculation
        # In practice, this would use the MMM model prediction
        
        total_profit = 0.0
        
        # Baseline profit
        baseline_daily = self.model_params.alpha_baseline + \
                        self.model_params.alpha_trend * (days / 2)  # Average trend
        total_profit += baseline_daily * days
        
        # Channel contributions
        for channel, daily_spend in spend_dict.items():
            if channel not in self.model_params.channel_alphas:
                continue
                
            alpha = self.model_params.channel_alphas[channel]
            beta = self.model_params.channel_betas[channel]
            r = self.model_params.channel_rs[channel]
            
            # Simplified calculation - apply adstock and saturation
            # Adstock effect (simplified geometric series)
            adstock_multiplier = 1 / (1 - r) if r < 0.99 else 10.0
            adstocked_spend = daily_spend * adstock_multiplier
            
            # Saturation effect
            saturated_spend = np.power(adstocked_spend, beta)
            
            # Channel contribution
            channel_profit = alpha * saturated_spend * days
            total_profit += channel_profit
        
        return total_profit
    
    def _calculate_shadow_prices(self,
                               optimal_spend: Dict[str, float],
                               total_budget: float,
                               constraints: List[Constraint],
                               days: int) -> Dict[str, float]:
        """Calculates shadow prices (marginal value of budget)."""
        shadow_prices = {}
        epsilon = 1000.0  # Small budget change for numerical derivative
        
        # Shadow price for total budget constraint
        base_profit = self._calculate_profit(optimal_spend, days)
        
        # Increase budget slightly and re-optimize
        increased_spend = {ch: sp * (total_budget + epsilon) / total_budget 
                          for ch, sp in optimal_spend.items()}
        increased_profit = self._calculate_profit(increased_spend, days)
        
        budget_shadow_price = (increased_profit - base_profit) / epsilon
        shadow_prices['total_budget'] = budget_shadow_price
        
        # Shadow prices for individual channels
        for channel in optimal_spend.keys():
            # Marginal efficiency at optimal spend
            current = optimal_spend[channel]
            increased = current + epsilon
            
            spend_copy = optimal_spend.copy()
            spend_copy[channel] = increased
            
            increased_profit = self._calculate_profit(spend_copy, days)
            marginal_efficiency = (increased_profit - base_profit) / epsilon
            
            shadow_prices[channel] = marginal_efficiency
        
        return shadow_prices
    
    def _generate_response_curves(self,
                                channels: List[str],
                                optimal_spend: Dict[str, float]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generates response curves for each channel."""
        response_curves = {}
        
        for channel in channels:
            # Define spend range (0 to 3x optimal spend)
            max_spend = max(optimal_spend[channel] * 3, 50000)
            spend_range = np.linspace(0, max_spend, 100)
            
            # Calculate profit for each spend level
            profit_values = np.zeros(len(spend_range))
            base_spend = optimal_spend.copy()
            
            for i, spend in enumerate(spend_range):
                base_spend[channel] = spend
                profit_values[i] = self._calculate_profit(base_spend, 365)  # Annual basis
            
            response_curves[channel] = (spend_range, profit_values)
        
        return response_curves
    
    def _perform_scenario_analysis(self,
                                 optimal_spend: Dict[str, float],
                                 total_budget: float,
                                 days: int) -> Dict[str, Any]:
        """Performs scenario analysis with different budget levels."""
        scenarios = {}
        base_profit = self._calculate_profit(optimal_spend, days)
        
        # Budget sensitivity scenarios
        budget_multipliers = [0.8, 0.9, 1.1, 1.2, 1.5]
        
        for multiplier in budget_multipliers:
            scenario_budget = total_budget * multiplier
            scenario_spend = {ch: sp * multiplier for ch, sp in optimal_spend.items()}
            scenario_profit = self._calculate_profit(scenario_spend, days)
            
            scenarios[f"budget_{int(multiplier*100)}pct"] = {
                "budget": scenario_budget,
                "spend": scenario_spend,
                "profit": scenario_profit,
                "profit_change_pct": (scenario_profit - base_profit) / base_profit * 100
            }
        
        # Channel elimination scenarios
        for channel in optimal_spend.keys():
            scenario_spend = optimal_spend.copy()
            scenario_spend[channel] = 0
            
            # Redistribute budget proportionally to other channels
            remaining_channels = [ch for ch in optimal_spend.keys() if ch != channel]
            if remaining_channels:
                freed_budget = optimal_spend[channel]
                total_remaining = sum(optimal_spend[ch] for ch in remaining_channels)
                
                for ch in remaining_channels:
                    if total_remaining > 0:
                        proportion = optimal_spend[ch] / total_remaining
                        scenario_spend[ch] += freed_budget * proportion
            
            scenario_profit = self._calculate_profit(scenario_spend, days)
            
            scenarios[f"eliminate_{channel}"] = {
                "eliminated_channel": channel,
                "spend": scenario_spend,
                "profit": scenario_profit,
                "profit_change_pct": (scenario_profit - base_profit) / base_profit * 100
            }
        
        return scenarios
    
    def _identify_binding_constraints(self,
                                    optimal_spend: Dict[str, float],
                                    constraints: List[Constraint],
                                    total_budget: float,
                                    tolerance: float = 0.01) -> List[str]:
        """Identifies which constraints are binding at the optimal solution."""
        binding = []
        
        # Check budget constraint
        if abs(sum(optimal_spend.values()) - total_budget) < tolerance:
            binding.append("total_budget")
        
        # Check explicit constraints
        for constraint in constraints:
            if constraint.channel not in optimal_spend:
                continue
                
            spend = optimal_spend[constraint.channel]
            
            if constraint.type == ConstraintType.FLOOR:
                if abs(spend - constraint.value) < tolerance:
                    binding.append(f"{constraint.channel}_floor")
                    
            elif constraint.type == ConstraintType.CAP:
                if abs(spend - constraint.value) < tolerance:
                    binding.append(f"{constraint.channel}_cap")
                    
            elif constraint.type == ConstraintType.LOCK:
                binding.append(f"{constraint.channel}_lock")
        
        return binding
    
    def get_response_curve(self,
                          channel: str,
                          spend_range: Optional[np.ndarray] = None,
                          resolution: int = 100) -> ResponseCurve:
        """Gets detailed response curve for a specific channel."""
        if spend_range is None:
            max_spend = 100000  # Default max spend
            spend_range = np.linspace(0, max_spend, resolution)
        
        # Calculate profit values
        profit_values = np.zeros(len(spend_range))
        base_spend = {ch: 0 for ch in self.model_params.channel_alphas.keys()}
        
        for i, spend in enumerate(spend_range):
            base_spend[channel] = spend
            profit_values[i] = self._calculate_profit(base_spend, 365)
        
        # Calculate marginal efficiency
        marginal_efficiency = np.gradient(profit_values, spend_range)
        
        return ResponseCurve(
            channel=channel,
            spend_range=spend_range,
            profit_values=profit_values,
            current_spend=0,  # Would be passed in from context
            optimal_spend=0,  # Would be calculated
            marginal_efficiency=marginal_efficiency
        )