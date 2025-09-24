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
    optimal_spend: Dict[str, float]  # Budget-constrained allocation
    optimal_profit: float  # Total profit (includes baseline)
    current_profit: float  # Total profit (includes baseline)
    profit_uplift: float
    shadow_prices: Dict[str, float]
    constraints_binding: List[str]
    response_curves: Dict[str, Tuple[np.ndarray, np.ndarray]]  # (spend_range, profit_values)
    scenario_analysis: Dict[str, Any]
    # New fields for true profit maximization
    true_optimal_spend: Dict[str, float] = None  # True optimal (mROI >= 1)
    true_optimal_profit: float = None
    true_optimal_budget: float = None  # Total spend for true optimal
    budget_reduction_pct: float = None  # How much less to spend for true optimal
    # Media-specific profit fields (excluding baseline)
    media_optimal_profit: float = None  # Media profit for optimal allocation
    media_current_profit: float = None  # Media profit for current allocation
    media_profit_uplift: float = None  # Media profit increase


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
                optimization_window_days: int = 365,
                use_marginal_roi: bool = True) -> OptimizationResult:
        """
        Optimizes budget allocation to maximize profit.

        Args:
            current_spend: Current spend levels by channel
            total_budget: Total budget constraint
            constraints: List of business constraints
            optimization_window_days: Time horizon for optimization
            use_marginal_roi: Use marginal ROI-based allocation (more realistic)

        Returns:
            OptimizationResult with optimal allocation and analysis
        """
        import structlog
        logger = structlog.get_logger()

        # DEBUG: Log all model parameters
        logger.info("=" * 80)
        logger.info("OPTIMIZER DEBUG: Starting optimization")
        logger.info(f"Total budget: ${total_budget:,.0f}")
        logger.info(f"Optimization window: {optimization_window_days} days")
        logger.info(f"Use marginal ROI: {use_marginal_roi}")

        logger.info("-" * 40)
        logger.info("Model parameters:")
        logger.info(f"  alpha_baseline: {self.model_params.alpha_baseline}")
        logger.info(f"  alpha_trend: {self.model_params.alpha_trend}")

        logger.info("Channel parameters:")
        for channel in current_spend.keys():
            alpha = self.model_params.channel_alphas.get(channel, 0)
            beta = self.model_params.channel_betas.get(channel, 0)
            r = self.model_params.channel_rs.get(channel, 0)
            logger.info(f"  {channel}:")
            logger.info(f"    alpha: {alpha:.4f}")
            logger.info(f"    beta: {beta:.4f}")
            logger.info(f"    r: {r:.4f}")
        logger.info("=" * 80)

        constraints = constraints or []
        channels = list(current_spend.keys())

        # Use marginal ROI-based greedy allocation for better results
        if use_marginal_roi:
            optimal_spend = self._optimize_with_marginal_roi(
                channels, total_budget, constraints, optimization_window_days
            )
            result_success = True
        else:
            # Original scipy optimization (kept for compatibility)
            bounds, constraint_funcs = self._setup_constraints(
                channels, current_spend, total_budget, constraints
            )

            def objective(spend_array):
                spend_dict = dict(zip(channels, spend_array))
                return -self._calculate_profit(spend_dict, optimization_window_days)

            x0 = np.array([current_spend[ch] for ch in channels])

            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraint_funcs
            )

            optimal_spend = dict(zip(channels, result.x))
            result_success = result.success

        if not result_success:
            raise ValueError(f"Optimization failed")

        # Extract results (optimal_spend already set above)
        optimal_profit = self._calculate_profit(optimal_spend, optimization_window_days)
        current_profit = self._calculate_profit(current_spend, optimization_window_days)
        profit_uplift = optimal_profit - current_profit

        # DEBUG: Log profit calculation details
        logger.info("-" * 80)
        logger.info("OPTIMIZER DEBUG: Profit calculation results")
        logger.info(f"Optimal profit: ${optimal_profit:,.2f}")
        logger.info(f"Current profit: ${current_profit:,.2f}")
        logger.info(f"Profit uplift: ${profit_uplift:,.2f}")

        # Manual verification
        logger.info("Manual profit verification:")
        baseline = self.model_params.alpha_baseline * optimization_window_days
        logger.info(f"  Baseline profit: ${baseline:,.2f}")

        media_profit = 0
        for channel, spend in optimal_spend.items():
            if spend > 0 and channel in self.model_params.channel_alphas:
                alpha = self.model_params.channel_alphas[channel]
                if alpha > 0:
                    beta = self.model_params.channel_betas[channel]
                    r = self.model_params.channel_rs[channel]
                    adstock_mult = 1 / (1 - r) if r < 0.99 else 10.0
                    adstocked = spend * adstock_mult
                    ch_profit = alpha * np.power(adstocked, beta)
                    media_profit += ch_profit
                    logger.info(f"  {channel}: ${spend:,.0f} -> ${ch_profit:,.2f}")

        logger.info(f"  Total media profit: ${media_profit:,.2f}")
        logger.info(f"  Total (baseline + media): ${baseline + media_profit:,.2f}")

        if abs((baseline + media_profit) - optimal_profit) > 100:
            logger.error(f"⚠️ DISCREPANCY: Manual calc ${baseline + media_profit:,.2f} != Optimal ${optimal_profit:,.2f}")
        logger.info("-" * 80)
        
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

        # Find true optimal allocation (respecting mROI >= 1)
        # First find global optimum by building up from zero
        # Use a fixed increment of $10 for consistency regardless of input budget
        fixed_increment = 10.0
        true_optimal_spend = self._find_global_optimum(channels, fixed_increment)
        true_optimal_budget = sum(true_optimal_spend.values())
        true_optimal_profit = self._calculate_profit(true_optimal_spend, optimization_window_days)
        budget_reduction_pct = ((total_budget - true_optimal_budget) / total_budget * 100) if total_budget > 0 else 0

        # Log comparison
        import structlog
        logger = structlog.get_logger()

        # Calculate media-only profit (excluding baseline)
        baseline_profit = self.model_params.alpha_baseline * optimization_window_days + \
                         self.model_params.alpha_trend * (optimization_window_days * optimization_window_days / 2 / 365)
        media_profit = optimal_profit - baseline_profit
        true_media_profit = true_optimal_profit - baseline_profit

        logger.info("=" * 80)
        logger.info("OPTIMIZATION RESULTS SUMMARY")
        logger.info("-" * 80)
        logger.info(f"Budget-Constrained Allocation (spending full ${total_budget:,.0f}):")
        logger.info(f"  Media-Generated Profit: ${media_profit:,.0f}")
        logger.info(f"  Media Spend: ${total_budget:,.0f}")
        logger.info(f"  Net Profit/Loss from Media: ${media_profit - total_budget:,.0f}")
        logger.info(f"  Media ROI: {media_profit / total_budget:.2f}")

        # Strong warning if media spend loses money
        if media_profit < total_budget:
            logger.warning("=" * 80)
            logger.warning("🚨 CRITICAL WARNING: MEDIA SPEND IS UNPROFITABLE!")
            logger.warning(f"   Spending ${total_budget:,.0f} on media generates only ${media_profit:,.0f} in profit")
            logger.warning(f"   This is a NET LOSS of ${total_budget - media_profit:,.0f}")
            logger.warning(f"   Every dollar spent returns only ${media_profit/total_budget:.2f}")
            logger.warning("=" * 80)

        logger.info("-" * 80)
        logger.info(f"True Optimal Allocation (mROI >= 1 constraint):")
        logger.info(f"  Recommended Budget: ${true_optimal_budget:,.0f} ({budget_reduction_pct:.1f}% reduction)")
        logger.info(f"  Media-Generated Profit: ${true_media_profit:,.0f}")
        logger.info(f"  Net Profit from Media: ${true_media_profit - true_optimal_budget:,.0f}")
        logger.info(f"  Media ROI: {true_media_profit / true_optimal_budget:.2f}" if true_optimal_budget > 0 else "  Media ROI: N/A")

        # Clear recommendation
        if budget_reduction_pct > 10:
            savings = total_budget - true_optimal_budget
            media_net_current = media_profit - total_budget
            media_net_optimal = true_media_profit - true_optimal_budget
            net_benefit = media_net_optimal - media_net_current

            logger.warning("=" * 80)
            logger.warning("💡 STRONG RECOMMENDATION:")
            logger.warning(f"   Reduce budget from ${total_budget:,.0f} to ${true_optimal_budget:,.0f}")
            logger.warning(f"   This would save ${savings:,.0f} in media spend")
            logger.warning(f"   Net benefit to bottom line: ${net_benefit:,.0f}")
            logger.warning("=" * 80)

        logger.info("=" * 80)

        # Calculate media profit for current spend
        current_media_profit = current_profit - (baseline_profit if current_profit > 0 else 0)

        return OptimizationResult(
            optimal_spend=optimal_spend,
            optimal_profit=optimal_profit,
            current_profit=current_profit,
            profit_uplift=profit_uplift,
            shadow_prices=shadow_prices,
            constraints_binding=constraints_binding,
            response_curves=response_curves,
            scenario_analysis=scenario_analysis,
            true_optimal_spend=true_optimal_spend,
            true_optimal_profit=true_optimal_profit,
            true_optimal_budget=true_optimal_budget,
            budget_reduction_pct=budget_reduction_pct,
            media_optimal_profit=media_profit,
            media_current_profit=current_media_profit,
            media_profit_uplift=media_profit - current_media_profit
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
        """Calculates profit for given spend allocation.

        Note: spend_dict contains TOTAL spend for the optimization period, not daily spend.
        The 'days' parameter is kept for backward compatibility but not used in channel calculations.
        """
        total_profit = 0.0

        # Baseline profit (non-media contribution) - this does scale with days
        baseline_daily = self.model_params.alpha_baseline + \
                        self.model_params.alpha_trend * (days / 2)  # Average trend
        total_profit += baseline_daily * days

        # Channel contributions
        for channel, total_spend in spend_dict.items():
            if channel not in self.model_params.channel_alphas:
                continue

            alpha = self.model_params.channel_alphas[channel]
            beta = self.model_params.channel_betas[channel]
            r = self.model_params.channel_rs[channel]

            # Skip channels with no contribution
            if alpha <= 0:
                continue

            # Apply adstock to total spend
            adstock_multiplier = 1 / (1 - r) if r < 0.99 else 10.0
            adstocked_spend = total_spend * adstock_multiplier

            # Saturation effect
            saturated_spend = np.power(adstocked_spend, beta)

            # Channel contribution: profit = alpha * (adstocked_spend)^beta
            # No multiplication by days - spend_dict already contains total spend for the period
            channel_profit = alpha * saturated_spend
            total_profit += channel_profit

        return total_profit

    def _calculate_marginal_roi(self, channel: str, current_spend: float, increment: float = 100.0) -> float:
        """Calculate marginal ROI for a channel at given spend level."""
        if channel not in self.model_params.channel_alphas:
            return 0.0

        alpha = self.model_params.channel_alphas[channel]
        beta = self.model_params.channel_betas[channel]
        r = self.model_params.channel_rs[channel]

        # Apply adstock
        adstock_multiplier = 1 / (1 - r) if r < 0.99 else 10.0

        # Avoid division by zero for zero spend
        if current_spend < 1.0:
            current_spend = 1.0

        adstocked_spend = current_spend * adstock_multiplier

        # Calculate derivative: d/dx[alpha * x^beta] = alpha * beta * x^(beta-1)
        # For adstocked spend, we also multiply by adstock_multiplier
        marginal_roi = alpha * beta * np.power(adstocked_spend, beta - 1) * adstock_multiplier

        return marginal_roi  # ROI per dollar

    def _optimize_with_marginal_roi(self,
                                   channels: List[str],
                                   total_budget: float,
                                   constraints: List[Constraint],
                                   days: int) -> Dict[str, float]:
        """Optimize using greedy marginal ROI allocation starting from zero."""
        # Start all channels at zero allocation
        optimal_spend = {ch: 0.0 for ch in channels}
        allocated_budget = 0.0

        # Apply floor constraints
        for constraint in constraints:
            if constraint.type == ConstraintType.FLOOR and constraint.channel in optimal_spend:
                optimal_spend[constraint.channel] = max(optimal_spend[constraint.channel], constraint.value)
            elif constraint.type == ConstraintType.LOCK and constraint.channel in optimal_spend:
                optimal_spend[constraint.channel] = constraint.value

        allocated_budget = sum(optimal_spend.values())

        if allocated_budget > total_budget:
            # Scale down proportionally if minimum allocation exceeds budget
            scale = total_budget / allocated_budget
            optimal_spend = {ch: spend * scale for ch, spend in optimal_spend.items()}
            return optimal_spend

        # Greedy allocation based on marginal ROI
        remaining_budget = total_budget - allocated_budget
        if remaining_budget <= 0:
            return optimal_spend

        # Use 1% of total budget as increment size (or $10 minimum for very small budgets)
        # This ensures proper granularity: $10 for $1K budget, $100 for $10K budget, etc.
        increment = max(10.0, total_budget * 0.01)
        max_iterations = int(remaining_budget / increment) + 1

        import structlog
        logger = structlog.get_logger()
        logger.info(f"Profit Maximizer: Starting from ZERO allocation")
        logger.info(f"Budget: ${total_budget:,.0f}, Increment: ${increment:,.0f} ({increment/total_budget*100:.1f}%), Max iterations: {max_iterations}")

        for iteration in range(max_iterations):
            if allocated_budget >= total_budget - increment * 0.5:  # Stop when we can't allocate a full increment
                break

            # Calculate marginal ROI for each channel at their CURRENT allocation
            marginal_rois = {}
            for channel in channels:
                # Skip channels with alpha = 0 (they don't contribute to profit)
                if channel in self.model_params.channel_alphas:
                    if self.model_params.channel_alphas[channel] <= 0:
                        continue

                # Check cap constraints
                is_capped = False
                for constraint in constraints:
                    if constraint.type == ConstraintType.CAP and constraint.channel == channel:
                        if optimal_spend[channel] >= constraint.value:
                            is_capped = True
                            break
                    elif constraint.type == ConstraintType.LOCK and constraint.channel == channel:
                        is_capped = True  # Locked channels can't change
                        break

                if not is_capped:
                    # Calculate mROI based on CURRENT spend for this channel
                    marginal_rois[channel] = self._calculate_marginal_roi(
                        channel, optimal_spend[channel], increment
                    )

            if not marginal_rois:
                logger.info("All channels are capped - stopping allocation")
                break  # All channels are capped

            # Allocate to channel with highest marginal ROI
            best_channel = max(marginal_rois, key=marginal_rois.get)
            best_roi = marginal_rois[best_channel]

            if best_roi <= 0:
                logger.info(f"No positive marginal ROI remaining (best was {best_roi:.4f} for {best_channel})")
                break  # No positive marginal ROI remaining

            # Log progress at key points
            if iteration == 0 or iteration % 10 == 0 or iteration == max_iterations - 1:
                logger.info(f"Iteration {iteration}: Allocating ${increment:,.0f} to {best_channel} (mROI={best_roi:.4f})")
                logger.info(f"  Current allocation: {', '.join([f'{ch}: ${optimal_spend[ch]:,.0f}' for ch in channels])}")

            optimal_spend[best_channel] += increment
            allocated_budget += increment

        # Final summary
        logger.info(f"Optimization complete after {iteration + 1} iterations")
        logger.info(f"Final allocation (${allocated_budget:,.0f} of ${total_budget:,.0f} budget):")
        for channel in channels:
            pct = (optimal_spend[channel] / total_budget * 100) if total_budget > 0 else 0
            final_roi = self._calculate_marginal_roi(channel, optimal_spend[channel], increment)
            logger.info(f"  {channel}: ${optimal_spend[channel]:,.0f} ({pct:.1f}%) - Final mROI: {final_roi:.4f}")

        return optimal_spend

    def _find_global_optimum(self, channels: List[str], increment: float) -> Dict[str, float]:
        """
        Finds the global optimum by building up from zero spend until marginal ROI < 1.
        This ensures we find the SINGLE true optimal point regardless of input budget.
        """
        import structlog
        logger = structlog.get_logger()

        logger.info("Finding global optimum (building from zero)")

        # Start with zero allocation
        allocation = {channel: 0.0 for channel in channels}

        # Track marginal ROIs for all channels
        active_channels = set(channels)
        iteration = 0
        max_iterations = 100000  # Safety limit

        while active_channels and iteration < max_iterations:
            iteration += 1

            # Find channel with highest marginal ROI
            best_channel = None
            best_mroi = 0.0

            for channel in active_channels:
                current_spend = allocation[channel]
                # For zero spend, use a small value to avoid numerical issues
                test_spend = max(1.0, current_spend)
                mroi = self._calculate_marginal_roi(channel, test_spend, increment)

                if mroi > best_mroi:
                    best_mroi = mroi
                    best_channel = channel

            # If best mROI < 1, we're done (no profitable additions)
            if best_mroi < 1.0:
                logger.info(f"  Stopping: Best mROI = {best_mroi:.4f} < 1.0")
                break

            # Add increment to best channel
            if best_channel:
                allocation[best_channel] += increment

                # Re-check mROI after adding increment
                new_mroi = self._calculate_marginal_roi(best_channel, allocation[best_channel], increment)

                if iteration % 1000 == 0:
                    total_spend = sum(allocation.values())
                    logger.info(f"  Iteration {iteration}: Added ${increment:.0f} to {best_channel}, "
                              f"new mROI={new_mroi:.4f}, total spend=${total_spend:,.0f}")

                # Remove channel if mROI drops below 1
                if new_mroi < 1.0:
                    active_channels.discard(best_channel)
                    logger.info(f"  {best_channel} reached optimal at ${allocation[best_channel]:,.0f} "
                              f"(mROI={new_mroi:.4f} < 1.0)")
            else:
                break

        # Final summary
        total_spend = sum(allocation.values())
        logger.info(f"Global optimum found at total spend: ${total_spend:,.0f}")

        for channel, spend in allocation.items():
            if spend > 0:
                final_mroi = self._calculate_marginal_roi(channel, spend, increment)
                logger.info(f"  {channel}: ${spend:,.0f} (mROI={final_mroi:.4f})")

        return allocation

    def _find_true_optimal_allocation(self,
                                     initial_allocation: Dict[str, float],
                                     increment: float) -> Dict[str, float]:
        """
        Legacy method - kept for backward compatibility.
        Now just calls _find_global_optimum to ensure consistency.
        """
        channels = list(initial_allocation.keys())
        return self._find_global_optimum(channels, increment)

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