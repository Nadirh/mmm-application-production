"""
Budget optimization endpoints.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import structlog

from mmm.optimization.optimizer import BudgetOptimizer, Constraint, ConstraintType, OptimizationResult
from mmm.api.routes.model import training_runs
from mmm.config.settings import settings

router = APIRouter()
logger = structlog.get_logger()


class ConstraintRequest(BaseModel):
    channel: str
    type: str  # "floor", "cap", "lock", "ramp"
    value: float
    description: Optional[str] = ""


class OptimizationRequest(BaseModel):
    run_id: str
    total_budget: float
    current_spend: Dict[str, float]
    constraints: Optional[List[ConstraintRequest]] = []
    optimization_window_days: Optional[int] = 365


@router.post("/run")
async def run_optimization(request: OptimizationRequest) -> Dict[str, Any]:
    """
    Run budget optimization with constraints.
    
    Args:
        request: Optimization request with budget and constraints
        
    Returns:
        OptimizationResult with optimal allocation and analysis
    """
    logger.info("Starting budget optimization", run_id=request.run_id)
    
    # Validate training run exists and is completed
    if request.run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    run = training_runs[request.run_id]
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    try:
        # Get model results
        model_params = run["results"].parameters
        
        # Convert constraint requests to constraint objects
        constraints = []
        for constraint_req in request.constraints:
            try:
                constraint_type = ConstraintType(constraint_req.type)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid constraint type: {constraint_req.type}"
                )
            
            constraints.append(Constraint(
                channel=constraint_req.channel,
                type=constraint_type,
                value=constraint_req.value,
                description=constraint_req.description
            ))
        
        # Create optimizer
        optimizer = BudgetOptimizer(model_params)
        
        # Run optimization
        result = optimizer.optimize(
            current_spend=request.current_spend,
            total_budget=request.total_budget,
            constraints=constraints,
            optimization_window_days=request.optimization_window_days
        )
        
        # Convert response curves to serializable format
        response_curves_data = {}
        for channel, (spend_range, profit_values) in result.response_curves.items():
            response_curves_data[channel] = {
                "spend_range": spend_range.tolist(),
                "profit_values": profit_values.tolist()
            }
        
        logger.info(
            "Budget optimization completed",
            run_id=request.run_id,
            profit_uplift=result.profit_uplift,
            optimal_profit=result.optimal_profit
        )

        # DEBUG: Log what we're returning
        logger.info("=" * 80)
        logger.info("API RETURNING:")
        logger.info(f"  Optimal profit: ${result.optimal_profit:,.2f}")
        logger.info(f"  Current profit: ${result.current_profit:,.2f}")
        logger.info(f"  Profit uplift: ${result.profit_uplift:,.2f}")
        logger.info("  Optimal spend allocation:")
        for channel, spend in result.optimal_spend.items():
            if spend > 0:
                logger.info(f"    {channel}: ${spend:,.0f}")
        logger.info("=" * 80)

        return {
            "run_id": request.run_id,
            "optimization_results": {
                "optimal_spend": result.optimal_spend,
                "optimal_profit": result.optimal_profit,
                "current_profit": result.current_profit,
                "profit_uplift": result.profit_uplift,
                "profit_uplift_pct": (result.profit_uplift / result.current_profit * 100) if result.current_profit > 0 else 0,
                "shadow_prices": result.shadow_prices,
                "constraints_binding": result.constraints_binding,
                # Media-specific profits (what actually matters for optimization)
                "media_optimal_profit": result.media_optimal_profit,
                "media_current_profit": result.media_current_profit,
                "media_profit_uplift": result.media_profit_uplift,
                "media_roi": result.media_optimal_profit / request.total_budget if request.total_budget > 0 else 0,
                # True optimal allocation (mROI >= 1 constraint)
                "true_optimal_spend": result.true_optimal_spend,
                "true_optimal_profit": result.true_optimal_profit,
                "true_optimal_budget": result.true_optimal_budget,
                "budget_reduction_pct": result.budget_reduction_pct,
                # Net profit/loss calculation
                "net_profit_loss": float(result.media_optimal_profit - request.total_budget),
                "is_profitable": bool(result.media_optimal_profit >= request.total_budget)
            },
            "response_curves": response_curves_data,
            "scenario_analysis": result.scenario_analysis
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is (like 400 for invalid constraints)
        raise
    except Exception as e:
        logger.error("Budget optimization failed", run_id=request.run_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.get("/response-curve/{run_id}/{channel}")
async def get_channel_response_curve(
    run_id: str,
    channel: str,
    max_spend: Optional[float] = None,
    resolution: Optional[int] = 100
) -> Dict[str, Any]:
    """Get response curve for a specific channel."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    run = training_runs[run_id]
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    model_params = run["results"].parameters
    
    if channel not in model_params.channel_alphas:
        raise HTTPException(status_code=404, detail="Channel not found in model")
    
    try:
        optimizer = BudgetOptimizer(model_params)
        
        if max_spend is None:
            max_spend = 100000  # Default max spend
        
        import numpy as np
        spend_range = np.linspace(0, max_spend, resolution)
        
        response_curve = optimizer.get_response_curve(channel, spend_range, resolution)
        
        return {
            "run_id": run_id,
            "channel": channel,
            "response_curve": {
                "spend_range": response_curve.spend_range.tolist(),
                "profit_values": response_curve.profit_values.tolist(),
                "marginal_efficiency": response_curve.marginal_efficiency.tolist(),
                "current_spend": response_curve.current_spend,
                "optimal_spend": response_curve.optimal_spend
            }
        }
        
    except Exception as e:
        logger.error("Response curve generation failed", run_id=run_id, channel=channel, error=str(e))
        raise HTTPException(status_code=500, detail=f"Response curve generation failed: {str(e)}")


@router.post("/scenario-analysis/{run_id}")
async def run_scenario_analysis(
    run_id: str,
    scenarios: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run custom scenario analysis.
    
    Args:
        run_id: Training run ID
        scenarios: Dictionary of scenario configurations
        
    Returns:
        Results for each scenario
    """
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    run = training_runs[run_id]
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    try:
        model_params = run["results"].parameters
        optimizer = BudgetOptimizer(model_params)
        
        scenario_results = {}
        
        for scenario_name, scenario_config in scenarios.items():
            current_spend = scenario_config.get("current_spend", {})
            total_budget = scenario_config.get("total_budget", 0)
            
            # Simple profit calculation for scenario
            scenario_optimizer = BudgetOptimizer(model_params)
            
            profit = scenario_optimizer._calculate_profit(
                current_spend, 
                scenario_config.get("optimization_window_days", 365)
            )
            
            scenario_results[scenario_name] = {
                "spend": current_spend,
                "total_budget": total_budget,
                "projected_profit": profit
            }
        
        return {
            "run_id": run_id,
            "scenario_results": scenario_results
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error("Scenario analysis failed", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Scenario analysis failed: {str(e)}")


@router.get("/constraints/defaults")
async def get_default_constraints() -> Dict[str, Any]:
    """Get default constraint values and recommendations."""
    return {
        "constraint_types": [
            {
                "type": "floor",
                "name": "Minimum Spend",
                "description": "Set minimum spend requirements for channels"
            },
            {
                "type": "cap",
                "name": "Maximum Spend",
                "description": "Set maximum spend limits for channels"
            },
            {
                "type": "lock",
                "name": "Fixed Spend",
                "description": "Lock spend at specific levels"
            },
            {
                "type": "ramp",
                "name": "Change Limit",
                "description": "Limit percentage change from current spend"
            }
        ],
        "default_ramp_limits": {
            "max_increase_pct": settings.optimization.default_max_increase_pct,
            "max_decrease_pct": settings.optimization.default_max_decrease_pct
        },
        "recommendations": {
            "ramp_constraints": "Consider limiting changes to ±20% for gradual optimization",
            "floor_constraints": "Set minimum spends to maintain brand presence",
            "cap_constraints": "Set maximum spends based on market saturation points"
        }
    }


@router.get("/shadow-prices/{run_id}")
async def get_shadow_prices(
    run_id: str,
    total_budget: float,
    current_spend: Dict[str, float]
) -> Dict[str, Any]:
    """Calculate shadow prices for budget allocation."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    run = training_runs[run_id]
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    try:
        model_params = run["results"].parameters
        optimizer = BudgetOptimizer(model_params)
        
        shadow_prices = optimizer._calculate_shadow_prices(
            current_spend, total_budget, [], 365
        )
        
        return {
            "run_id": run_id,
            "shadow_prices": shadow_prices,
            "interpretation": {
                "total_budget": "Marginal profit per additional dollar of total budget",
                "channels": "Marginal profit per additional dollar spent on each channel"
            }
        }
        
    except Exception as e:
        logger.error("Shadow price calculation failed", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Shadow price calculation failed: {str(e)}")