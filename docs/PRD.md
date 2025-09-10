Media Mix Model (MMM) — Complete Implementation PRD (MVP)
Section 1: What the model does
Purpose: Estimate the incremental contribution of each marketing channel to Profit. Recommend budget allocations that maximize Profit under business constraints.
Note: "Profit" is defined by each client (e.g., Leads × Value_per_Lead for lead-gen, Revenue - COGS for e-commerce).
Model Structure:
Profit_t = Baseline_t + sum_c(alpha_c * (Adstock_t_c ^ beta_c)) + error_t

Where:
Baseline_t = alpha_baseline + alpha_trend * days_since_start_t
alpha_baseline = intercept (initial baseline profit level)
alpha_trend = linear trend coefficient (daily organic growth/decline rate)
days_since_start_t = number of days since first date in dataset
alpha_c = channel incremental strength (alpha_c >= 0 for each channel c)
Adstock_t_c = adstocked spend for channel c at time t
beta_c = saturation parameter for channel c (0 < beta_c <= 1)
Outputs:
By-channel contributions and response curves
Elasticities and mROI (marginal ROI)
Budget optimizer: proposed spend by channel, expected delta_Profit, binding constraints
Scenarios: reallocation at fixed budget, scaling (±10% or custom), rule stress tests
Section 2: What data we need (daily)
Required Format (wide, one row per date, no gaps):
date (YYYY-MM-DD)
profit (daily profit as defined by client, cannot be negative, calculated before ad spend)
Per-channel spend columns: Search_Brand, Search_Non_Brand, Social, TV_Video, Display, etc. (numbers >= 0)
Optional Fields:
is_holiday (0/1 flag)
promo_flag (0/1 flag)
site_outage (0/1 flag)
Data Validation Error Codes:
ERROR_001: Negative spend detected in channel [channel_name] on [date]
ERROR_002: Missing date gaps detected between [start_date] and [end_date]
ERROR_003: Day-over-day spend jump >300% in [channel_name] on [date]
ERROR_004: Negative Profit detected on [date]
WARNING_001: Zero spend across all channels on [date]
Missing Data Handling:
Missing optional flags: Default to 0
Missing spend values: Convert to 0 and log WARNING_002
Enhanced Business Viability Tiers & Requirements:
Enterprise Tier (Target Market):
Minimum: 365+ days
CV Folds: 17+
Annual spend: >$2MM
Channel threshold: Individual channels must exceed $25K annually
Status: Green light - "High confidence recommendations with superior accuracy"
Reliability: Very high (Target MAPE: 5-15%)
Mid-Market Tier (Acceptable with Warnings):
Minimum: 280+ days (40+ weeks)
CV Folds: 11-16
Annual spend: $500K-$2MM
Channel threshold: Individual channels must exceed $15K annually
Status: Amber warning - "Reliable foundation with moderate confidence intervals"
Reliability: High with increased uncertainty (Target MAPE: 10-20%)
Small Business Tier (Directional Insights Only):
Minimum: 182+ days (26+ weeks)
CV Folds: 4-10
Annual spend: $200K-$500K
Channel threshold: Individual channels must exceed $8K annually
Status: Red warning - "Directional insights only. NOT suitable for major budget decisions"
Reliability: Low to moderate (Target MAPE: <=35%)
Prototype Tier (Limited Reliability):
Minimum: 182+ days
Annual spend: $50K-$200K
Status: Orange flag - "Prototype analysis with high uncertainty. Educational purposes only"
Reliability: Very limited (Expected MAPE: >25%)
Insufficient Data (Not Viable):
Less than 180 days OR annual spend <$50K
Status: Rejection - "MMM requires minimum 6 months of data and $50K annual spend for any meaningful analysis"
Recommendation: "Consider simpler measurement methods or gather more data before implementing MMM"
File Upload Requirements:
Accepted formats: CSV only
Maximum file size: 50MB
Required headers: exact column name matching (case-sensitive)
Date format validation: YYYY-MM-DD only
Encoding: UTF-8 required
Section 3: How the model thinks (media response)
Core Model:
Profit_t = Baseline_t + sum_c(alpha_c * (Adstock_t_c ^ beta_c)) + error_t

Baseline Components:
Organic Business Trend: Captures natural business growth/decline independent of paid media
Prevents misattribution: Organic trends not attributed to paid channels
Estimated as: Baseline_t = alpha_baseline + alpha_trend * days_since_start_t
Parameter Constraints:
alpha_baseline: Unconstrained (can be positive/negative)
alpha_trend: Unconstrained (allows growth or decline)
alpha_c >= 0 (no negative incremental channel effects)
Adstock Transformation (memory effect):
Adstock_t_c = Spend_t_c + r_c * Adstock_(t-1)_c

Initial condition: Adstock_0_c = 0 for all channels
0 <= r_c < 0.99 (prevent numerical instability)
Saturation Transformation (diminishing returns):
Transformed_Spend_t_c = Adstock_t_c ^ beta_c

0.1 <= beta_c <= 1.0 (prevent extreme transformations)
Channel Contribution:
Contribution_t_c = alpha_c * (Adstock_t_c ^ beta_c)

alpha_c >= 0 (no negative incremental channel effects)
Represents incremental profit above baseline trend
Enhanced Parameter Estimation Strategy:
Enterprise Tier: Individual beta_c and r_c for channels above threshold 
Mid-Market Tier: Individual parameters for channels above threshold 
Small Business/Prototype: Small Business/Prototype: Individual parameter estimation for all channels above threshold, with strong directional-only warnings due to limited data reliability.
Industry-Informed Default Parameters by Channel Type:
Search Brand: beta = 0.6, r = 0.15 (faster saturation, moderate memory)
Search Non-Brand: beta = 0.7, r = 0.25 (keyword auction dynamics)
Social: beta = 0.45, r = 0.35 (strong saturation, social proof carryover)
TV/Video: beta = 0.35, r = 0.45 (heavy saturation, strong carryover)
Display: beta = 0.5, r = 0.3 (moderate saturation and memory)
Unknown: beta = 0.5, r = 0.3 (conservative middle ground)
Channel Type Classification (case-insensitive matching):
Search Brand: contains "search" + ("brand" OR "branded")
Search Non-Brand: contains "search" but NOT ("brand" OR "branded")
Social: contains {"facebook", "instagram", "meta", "tiktok", "twitter", "linkedin", "pinterest", "snapchat"}
TV/Video: contains {"youtube", "tv", "video", "ctv", "ott", "connected"}
Display: contains {"display", "banner", "programmatic", "gdn", "google display"}
Unknown: All others use Unknown defaults
For naming precedence:
def channel_type_classification(channel_name): 
""" Classify channel by name with platform-specific precedence 
Priority order: Specific platforms > Search patterns > Generic terms """ 
name_lower = channel_name.lower()
# Priority 1: Specific social platforms (highest precedence) 
social_platforms = ['facebook', 'instagram', 'meta', 'tiktok', 'twitter', 'linkedin', 'pinterest', 'snapchat'] 
if any(platform in name_lower for platform in social_platforms): 
return 'social' 
# Priority 2: Specific video/TV platforms 
video_platforms = ['youtube', 'ctv', 'ott', 'connected'] 
if any(platform in name_lower for platform in video_platforms): 
return 'tv_video' 
# Priority 3: Generic video/TV terms (only if no specific platform matched) 
if any(term in name_lower for term in ['tv', 'video']): 
return 'tv_video' 
# Priority 4: Search classification (after platform-specific checks) 
if 'search' in name_lower: 
if any(brand_term in name_lower for brand_term in ['brand', 'branded']): return 'search_brand' 
else: return 'search_non_brand' 
# Priority 5: Display terms 
display_terms = ['display', 'banner', 'programmatic', 'gdn', 'google display'] 
if any(term in name_lower for term in display_terms): 
return 'display' 
# Default: Unknown 
return 'unknown'

Section 4: Cross-Validation & Search Strategy (COMPLETE ALGORITHM)
Business Context: What Cross-Validation Accomplishes
Cross-validation is the foundation of model reliability. It answers the critical business question: "How do we know this model will actually work when making future predictions?"
The Business Problem: Any model can be made to perfectly "predict" historical data it was trained on - that's called overfitting. But overfitted models fail catastrophically when making real business decisions about future budget allocation.
Why This Matters for Your Budget:
Model Validation: Ensures recommendations are based on genuinely predictive patterns, not statistical noise
Parameter Selection: Chooses model settings that work consistently across different time periods
Risk Management: Identifies when model predictions are unreliable before costly budget mistakes
Confidence Building: Provides evidence that the model's recommendations are trustworthy
How Cross-Validation Actually Works
Step 1: Create Multiple "Exams" for the Model We divide your historical data into multiple time periods, each serving as a test of how well the model predicts unseen future periods.
Example Timeline:
Fold 1: Train on Jan 1 - May 6 (126 days) → Test on May 7 - May 20 (14 days)
Fold 2: Train on Jan 15 - May 20 (126 days) → Test on May 21 - June 3 (14 days)
Fold 3: Train on Jan 29 - June 3 (126 days) → Test on June 4 - June 17 (14 days)
Continue until no more test periods remain...
Why 126 Training Days? This provides enough data for stable parameter estimation while ensuring we can create multiple test periods. It's roughly 18 weeks - long enough to capture seasonal patterns but short enough to create many validation folds.
Why 14 Test Days? This matches typical business planning cycles (2 weeks) and provides enough data points for reliable accuracy measurement without using up too much data for testing.
Step 2: Test Every Parameter Combination on Every Time Period For each fold, we:
Train models using different parameter combinations on the training period
Make predictions for the test period
Measure accuracy by comparing predictions to actual results
Record performance for each parameter combination
Step 3: Find Parameters That Work Consistently After testing across all time periods, we select the parameter combination with the best average performance across all folds, not just the most recent period.
Why Average Performance Matters:
Seasonal Robustness: Parameters must work in Q1 and Q4, not just holiday season
Market Condition Reliability: Must work during both boom and recession periods
Event Independence: Can't rely on parameters that only work during promotional periods
The Business Logic Behind Our Approach
Fixed Training Windows (126 Days)
Fair Comparison: Every parameter combination gets the same amount of training data
Prevents Cheating: Later folds can't use information earlier folds didn't have
Realistic Constraints: Mimics real-world scenario where you have limited historical data
Walk-Forward Testing
Time Series Respect: Honors the fact that you can't predict the past using the future
Business Reality: Simulates how the model would actually be used for planning
Progressive Learning: Tests how well the model adapts as new data becomes available
Global Winner Selection
Stability Focus: Prioritizes parameters that work consistently over time
Risk Mitigation: Avoids parameters that work brilliantly once but fail everywhere else
Forward-Looking: Optimizes for future performance, not historical fit
Performance Standards by Business Tier
Enterprise Tier (Annual Spend >$2MM)
Target Accuracy: ≤15% prediction error (MAPE)
Alert Threshold: >20% prediction error triggers investigation
Minimum Folds: 17+ for robust parameter estimation
Business Confidence: High reliability for major budget decisions
Mid-Market Tier (Annual Spend $500K-$2MM)
Target Accuracy: ≤20% prediction error (MAPE)
Alert Threshold: >25% prediction error triggers investigation
Minimum Folds: 11-16 folds
Business Confidence: Good reliability with moderate uncertainty
Small Business Tier (Annual Spend $200K-$500K)
Target Accuracy: ≤25% prediction error (MAPE)
Alert Threshold: >30% prediction error triggers investigation
Minimum Folds: 4-10 folds
Business Confidence: Directional insights only, not suitable for major decisions
What These Numbers Mean:
15% MAPE: If actual profit was $100K, model predictions typically within $85K-$115K
25% MAPE: If actual profit was $100K, model predictions typically within $75K-$125K
Computational Strategy & Business Trade-offs
Primary Approach: Comprehensive Validation
Accuracy Focus: Maximum testing for best possible parameter selection
Time Investment: 2-4 hours training time for highest confidence
Business Value: Optimal for organizations making major budget allocation decisions
Backup Approach: Efficient Validation
Speed Focus: 60-80% time reduction with minimal accuracy loss
Practical Trade-off: 90% of the benefit in 20% of the time
Business Value: Suitable when speed matters more than perfection
When to Use Each Approach:
Comprehensive: Annual budget planning, major strategic decisions, first-time model development
Efficient: Monthly refreshes, tactical adjustments, proof-of-concept development
Fixed-Window Walk-Forward Cross-Validation Framework
Critical Fix: Fixed training window prevents data leakage where later folds have unfair advantage on earlier test periods.
CV Configuration:
Training window: 126 days (18 weeks) - FIXED for all folds
Test window: 14 days (2 weeks)
Step size: 14 days (no overlap between test periods)
Complete Cross-Validation Algorithm
def walk_forward_cross_validation(data, min_training_days=126, test_days=14, step_days=14):
    """
    Implement fixed-window walk-forward CV with global parameter selection
    """
    # Initialize CV configuration
    cv_folds = []
    start_date = data['date'].min()
    end_date = data['date'].max()
    total_days = (end_date - start_date).days + 1
    
    # Generate fold definitions
    current_start = 0
    fold_id = 1
    
    while True:
        train_end = current_start + min_training_days
        test_start = train_end
        test_end = test_start + test_days
        
        # Check if enough data remains
        if test_end > total_days:
            break
            
        cv_folds.append({
            'fold_id': fold_id,
            'train_start_day': current_start,
            'train_end_day': train_end,
            'test_start_day': test_start,
            'test_end_day': test_end,
            'train_dates': (start_date + pd.Timedelta(days=current_start), 
                          start_date + pd.Timedelta(days=train_end)),
            'test_dates': (start_date + pd.Timedelta(days=test_start),
                         start_date + pd.Timedelta(days=test_end))
        })
        
        current_start += step_days
        fold_id += 1
    
    return cv_folds

def coordinate_search_per_fold(fold_data, channel_types, parameter_grids):
    """
    Coordinate search: optimize one channel type at a time
    """
    # Initialize with channel-type defaults
    current_best_params = initialize_default_parameters(channel_types)
    current_best_mape = float('inf')
    
    max_passes = 3
    converged = False
    
    for pass_num in range(max_passes):
        pass_changed = False
        
        # Iterate through each channel type
        for channel_type in channel_types:
            best_params_for_type = None
            best_mape_for_type = float('inf')
            
            # Try all parameter combinations for this channel type
            for beta, r in parameter_grids[channel_type]['stage1']:
                # Update parameters for this channel type only
                test_params = current_best_params.copy()
                test_params[channel_type] = {'beta': beta, 'r': r}
                
                # Fit model and evaluate MAPE on fold training data
                mape = fit_and_evaluate_model(fold_data, test_params)
                
                if mape < best_mape_for_type:
                    best_mape_for_type = mape
                    best_params_for_type = {'beta': beta, 'r': r}
            
            # Update if improvement found
            if best_mape_for_type < current_best_mape:
                current_best_params[channel_type] = best_params_for_type
                current_best_mape = best_mape_for_type
                pass_changed = True
        
        # Check convergence
        if not pass_changed:
            converged = True
            break
    
    # Stage 2: Local refinement around best parameters
    refined_params = local_refinement(fold_data, current_best_params, channel_types)
    
    return refined_params, current_best_mape

def global_parameter_selection(all_fold_results):
    """
    Select global winner based on average MAPE across all folds
    """
    # Aggregate results across all folds
    parameter_performance = defaultdict(list)
    
    for fold_result in all_fold_results:
        for param_combo, mape in fold_result['tested_combinations'].items():
            parameter_performance[param_combo].append(mape)
    
    # Calculate average MAPE for each parameter combination
    avg_performance = {}
    for param_combo, mape_list in parameter_performance.items():
        if len(mape_list) >= len(all_fold_results) * 0.8:  # Must appear in 80% of folds
            avg_performance[param_combo] = np.mean(mape_list)
    
    # Select global winner
    global_winner = min(avg_performance.items(), key=lambda x: x[1])
    return global_winner[0], global_winner[1]

Process Flow:
Fold 1: Train on days 1-126 → Test on days 127-140 → Calculate MAPE₁
Fold 2: Train on days 15-140 → Test on days 141-154 → Calculate MAPE₂
Fold 3: Train on days 29-154 → Test on days 155-168 → Calculate MAPE₃
Continue until <14 test days remain
Global Winner Selection: Parameter combination with lowest average MAPE across all folds (not most recent fold winner)
Enhanced Performance Targets (Business-Tier Adjusted):
Enterprise Tier: Target ≤20% CV MAPE, Alert if >30%
Mid-Market Tier: Target ≤25% CV MAPE, Alert if >35%
Small Business: Target ≤35% CV MAPE, Alert if >40%
Prototype: Target ≤30% CV MAPE, Alert if >35%
Additional Performance Requirements:
Minimum R²: ≥ 0.25 (model explains at least 25% of profit variance)
Shadow price validity: 0.3 ≤ shadow_price ≤ 8.0 (realistic marginal ROI range)
Media attribution: 20% ≤ media_contribution ≤ 80% of total modeled profit
Computational Efficiency Backup Approach
For cases where training time becomes prohibitive:
EFFICIENT_PARAMETER_GRIDS = {
    'search_brand': {
        'stage1': [(b, r) for b in [0.5, 0.6, 0.7] for r in [0.1, 0.15, 0.2]],  # 9 combinations
        'stage2_size': 3  # 3x3 refinement
    },
    # Similar for other channel types
}

def early_stopping_cv(fold_results, min_folds=8, improvement_threshold=0.005):
    """
    Stop CV early if MAPE improvement plateaus
    """
    if len(fold_results) < min_folds:
        return False
    
    recent_mapes = [fold['best_mape'] for fold in fold_results[-3:]]
    improvement = max(recent_mapes) - min(recent_mapes)
    
    return improvement < improvement_threshold

Expected Time Reduction: 60-80% of current training time Expected Accuracy Impact: 0.5-2% MAPE increase in worst cases


Section 5: Parameter Grids & Selection (COMPLETE SPECIFICATION)
Business Context: What This Section Accomplishes
This section defines how the system finds the best parameters (beta and r values) for each marketing channel. Think of this as "tuning" the model to get the most accurate predictions possible.
Why This Matters for Business:
Accuracy: Better parameter selection = more accurate profit predictions and budget recommendations
Reliability: Systematic testing ensures the model works consistently across different time periods
Channel-Specific Insights: Each channel type gets parameters that reflect its unique behavior (e.g., search responds differently than TV)
The Business Challenge: Every marketing channel behaves differently. Search advertising might have immediate impact with little carryover, while TV advertising might have delayed impact that builds over weeks. The model needs to "learn" these patterns from your data to make accurate predictions.
What We're Optimizing:
Beta (Saturation): How quickly does additional spend in this channel become less effective?
r (Memory/Adstock): How long does the impact of today's advertising last?
Real Business Impact: Getting these parameters right is the difference between a model that says "increase Search by 50%" versus "increase Search by 15%" - potentially millions of dollars in allocation differences.
How the Parameter Search Actually Works
Step 1: Create the Testing Framework Imagine you have 17 different time periods from your historical data. For each time period, we'll train a model using earlier data and test it on that specific period. This is like having 17 different "exams" for our parameter choices.
Step 2: Test Every Parameter Combination on Every Time Period For Search Brand channels, we test 81 different combinations of beta and r values:
Beta values from 0.4 to 0.8 (testing how quickly Search Brand saturates)
r values from 0.0 to 0.2 (testing how long Search Brand impact lasts)
The Process:
Pick a parameter combination (say beta=0.7, r=0.15 for Search Brand)
Train the model using this combination on each time period's training data
Test the prediction accuracy on that time period's test data
Record how well it performed (the prediction error percentage)
Repeat for all 81 combinations across all 17 time periods
Calculate average performance for each parameter combination
Why This Takes Time: With 5 channel types × 81 combinations × 17 time periods = about 6,885 individual model training runs. Each one takes a few seconds, but they add up to hours of computation.
Step 3: The "Coordinate Search" Strategy Rather than testing every possible combination simultaneously (which would be millions of possibilities), we use a smarter approach:
Round 1: Optimize Search Channels
Set all other channel types to their current best estimated parameters
Test all 81 combinations for Search Brand, find the best one
Test all 81 combinations for Search Non-Brand, find the best one
Keep these winners for Search channels
Round 2: Optimize Social Channels
Keep the Search winners from Round 1
Test all 81 combinations for Social channels with the Search winners fixed
Find the best Social parameters
Round 3: Optimize TV/Video and Display
Continue the same process for remaining channel types
Round 4: Check for Improvements
With all "first-round winners" selected, go back through each channel type
See if any channel type wants to change its choice given the new winners
Stop when no channel type wants to change (convergence)
Business Analogy: This is like optimizing your marketing organization structure. You can't change everyone's role simultaneously - you optimize the Search team first, then optimize Social given the new Search structure, then optimize TV given the new Search and Social structure, and keep iterating until no team wants to change their structure.
Step 4: Fine-Tuning (Stage 2) Once we have our "approximate winners," we test 25 additional combinations around each winner:
If Search Brand's winner was beta=0.65, r=0.15, we test values like beta=0.63, r=0.13 or beta=0.67, r=0.17
This catches cases where the "true optimum" is between our original test points
Step 5: Global Winner Selection After all this testing, we have performance data for every parameter combination across all time periods. The "global winner" is the combination with the lowest average prediction error across all 17 time periods.
How Many Parameter Combinations Get Evaluated for Global Winner?
This is a common question that highlights an important aspect of the coordinate search process.
The Mathematical Reality: With 3 channels each having 81 possible parameter combinations (9 beta × 9 r values), the theoretical maximum is 81³ = 531,441 total combinations. However, coordinate search is much more efficient.
What Actually Gets Tested:
Per Fold: Approximately 560 unique parameter combinations
Across 17 Folds: These combinations have significant overlap, but not 100% overlap
Why There's Variation Across Folds: Each fold's data has different patterns, so the coordinate search finds different "local winners" in Stage 1. This means the Stage 2 refinement grids center around different points in each fold, causing each fold to explore slightly different regions of the parameter space.
Realistic Numbers for Global Winner Selection:
Minimum candidates: 560 (if every fold tested identical combinations)
Maximum candidates: 560 * 17 (if every fold tested completely different combinations)
Realistic estimate: 1,500-3,000 unique parameter combinations tested across all 17 folds
Example of How This Works:
Fold 1 might find Search Brand winner at β=0.65, r=0.15, so Stage 2 tests around that area
Fold 8 might find Search Brand winner at β=0.70, r=0.12, so Stage 2 tests around that different area
Result: Some overlap between folds, but also some unique combinations tested
For Global Winner Selection: We evaluate all unique parameter combinations that were tested across any fold. Each combination gets an average MAPE score based on the folds where it was tested. The combination with the lowest average MAPE becomes our global winner.
Business Implication: This approach ensures we find parameters that work consistently across different time periods and market conditions, rather than just those that happened to work well in one specific period.

Computational Efficiency Strategy
Primary Approach: Comprehensive testing for maximum accuracy (recommended for overnight training)
Backup Approach: If training takes too long, we can reduce the search space by 60-80% with minimal accuracy loss:
Smaller grids: Test 9 combinations instead of 81 per channel type
Early stopping: If 8 time periods all prefer the same parameters, stop testing
Fewer refinement steps: Skip the 25-combination fine-tuning stage
The Trade-off: Comprehensive approach might find parameters that improve profit recommendations by an extra 2-3%. Efficient approach gets you 90% of the benefit in 20% of the time.
Business Decision Point: Most businesses prefer "very good and reliable" over "theoretically perfect but time-consuming." The backup approach is often sufficient for decision-making purposes, especially when you can retrain monthly with new data.
Primary Approach: Two-Stage Grid Search
PRIMARY_PARAMETER_GRIDS = {
    'search_brand': {
        'stage1': [
            (beta, r) for beta in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
                      for r in [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
        ],  # 81 combinations
        'defaults': {'beta': 0.6, 'r': 0.15}
    },
    'search_non_brand': {
        'stage1': [
            (beta, r) for beta in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
                      for r in [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
        ],  # 81 combinations
        'defaults': {'beta': 0.7, 'r': 0.25}
    },
    'social': {
        'stage1': [
            (beta, r) for beta in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
                      for r in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        ],  # 81 combinations
        'defaults': {'beta': 0.45, 'r': 0.35}
    },
    'tv_video': {
        'stage1': [
            (beta, r) for beta in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
                      for r in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        ],  # 81 combinations
        'defaults': {'beta': 0.35, 'r': 0.45}
    },
    'display': {
        'stage1': [
            (beta, r) for beta in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
                      for r in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        ],  # 81 combinations
        'defaults': {'beta': 0.5, 'r': 0.3}
    },
    'unknown': {
        'stage1': [
            (beta, r) for beta in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
                      for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        ],  # 81 combinations
        'defaults': {'beta': 0.5, 'r': 0.3}
    }
}

def generate_stage2_grid(winner_beta, winner_r, grid_size=5, step_size=0.02):
    """
    Generate local refinement grid around Stage 1 winner
    """
    half_size = grid_size // 2
    
    beta_values = [
        max(0.1, min(1.0, winner_beta + (i - half_size) * step_size))
        for i in range(grid_size)
    ]
    
    r_values = [
        max(0.0, min(0.99, winner_r + (i - half_size) * step_size))
        for i in range(grid_size)
    ]
    
    return [(b, r) for b in beta_values for r in r_values]

def channel_type_classification(channel_name):
    """
    Classify channel by name to assign appropriate parameter grid
    """
    name_lower = channel_name.lower()
    
    if 'search' in name_lower:
        if any(brand_term in name_lower for brand_term in ['brand', 'branded']):
            return 'search_brand'
        else:
            return 'search_non_brand'
    elif any(social_term in name_lower for social_term in 
             ['facebook', 'instagram', 'meta', 'tiktok', 'twitter', 'linkedin', 'pinterest', 'snapchat']):
        return 'social'
    elif any(video_term in name_lower for video_term in 
             ['youtube', 'tv', 'video', 'ctv', 'ott', 'connected']):
        return 'tv_video'
    elif any(display_term in name_lower for display_term in 
             ['display', 'banner', 'programmatic', 'gdn', 'google display']):
        return 'display'
    else:
        return 'unknown'

Winner Selection Algorithm:
def select_parameter_winner(fold_results, stability_threshold=0.60):
    """
    Select winner with stability tie-breaking
    """
    # Primary criterion: Lowest average MAPE
    avg_mapes = calculate_average_mapes(fold_results)
    min_mape = min(avg_mapes.values())
    
    # Find candidates within 0.05% of best MAPE
    candidates = {
        params: mape for params, mape in avg_mapes.items()
        if mape <= min_mape + 0.0005
    }
    
    if len(candidates) == 1:
        return list(candidates.keys())[0]
    
    # Tie-breaking: Bootstrap stability
    stability_scores = calculate_bootstrap_stability(candidates.keys())
    
    # Select most stable candidate
    winner = max(candidates.keys(), key=lambda p: stability_scores[p])
    return winner


Section 6: Budget Optimization & Response Curves (COMPLETE ALGORITHM)
Business Context: What Budget Optimization Accomplishes
Budget optimization answers the most critical business question: "Given what we've learned about how each channel performs, how should we allocate our marketing budget to maximize profit?"
The Business Problem: You have a fixed marketing budget and multiple channels competing for those dollars. Most traditional approaches don't account for diminishing returns, carryover effects, or the complex interactions between your marketing mix and business outcomes.
Why This Matters:
Profit Maximization: Find the allocation that generates the highest total profit
Scientific Approach: Use mathematical optimization
Constraint Management: Respect real-world business constraints while finding the best possible solution
Measurable Impact: Quantify exactly how much additional profit the optimized allocation should generate
Real Business Impact: Organizations typically see 5-15% improvement in marketing efficiency through optimization. For a $10M annual budget, this translates to $500K-$1.5M in additional profit from the same spending.
What We're Optimizing: The Economics Behind the Math
Diminishing Returns Mathematics: Every channel follows a predictable pattern: the first dollar spent generates more impact than the 1,000th dollar. We use power-law functions to model this:
Strong Diminishing Returns (β = 0.5): Need 4x spend for 2x impact
Minimal Diminishing Returns (β = 0.9): Nearly linear response
The Optimization Principle: Equalizing Marginal ROI
Core Economic Principle: The optimal allocation is reached when the marginal ROI (return on the last dollar spent) is equal across all unconstrained channels.
Business Translation: You can't move $1 from any channel to another channel and increase total profit. If you could, the allocation wouldn't be optimal.
Why This Works:
If Search has marginal ROI of 3.0 and Social has marginal ROI of 1.5, move money from Search to Social
Continue until both channels have the same marginal ROI
When marginal ROIs are equalized, you've found the maximum profit allocation
Real Example:
Before Optimization: Search mROI = 2.8, Social mROI = 1.2, TV mROI = 3.1
After Optimization: Search mROI = 2.4, Social mROI = 2.4, TV mROI = 2.4
Result: Total profit increased because money moved from low-efficiency to high-efficiency channels
How Business Constraints Work
Real businesses can't simply move money anywhere they want. The optimization respects these practical limitations:
Floor Constraints (Minimum Spend):
Business Reason: Brand maintenance, competitive presence, contractual obligations
Example: "Search Brand must have at least $50K to maintain market share"
Optimization Impact: Channel may operate below its optimal marginal ROI
Cap Constraints (Maximum Spend):
Business Reason: Channel capacity, market saturation, operational limits
Example: "Local radio can't effectively spend more than $30K/month in our market"
Optimization Impact: Channel may have higher marginal ROI but can't spend more
Lock Constraints (Fixed Spend):
Business Reason: Contractual commitments, strategic imperatives, political considerations
Example: "TV spend locked at $100K due to annual contract"
Optimization Impact: Channel removed from optimization, others adjust around it
Ramp Constraints (Change Limits):
Business Reason: Operational feasibility, agency capacity, gradual scaling
Example: "No channel can change by more than ±20% from current levels"
Optimization Impact: May prevent reaching true optimum but ensures implementable changes
The Shadow Price: Your Budget's Value
What Shadow Price Means: The shadow price represents the profit you'd gain from one additional dollar of budget, allocated optimally.
Business Interpretation Examples:
Shadow Price = 2.3: An extra $1,000 of budget would generate $2,300 in additional profit
Shadow Price = 0.8: An extra $1,000 would only generate $800 in profit (you're overspending)
Shadow Price = 4.5: An extra $1,000 would generate $4,500 in profit (strong case for budget increase)
Decision Guidelines:
λ < 1.0: Consider reducing budget - you're past the point of positive ROI
1.0 ≤ λ ≤ 3.0: Healthy efficiency range for most businesses
λ > 3.0: Strong business case to increase budget
The Optimization Process: How We Find the Best Allocation
Step 1: Build the Mathematical Model
Convert each channel's parameters (alpha, beta, r) into response functions
Calculate marginal ROI functions for each channel (by taking the first derivative of the function).  
Set up the constraint system (floors, caps, locks, budget limit)
Step 2: Check Feasibility Before starting optimization, validate that constraints don't conflict:
Budget Check: Sum of floors must be ≤ total budget
Channel Check: Floor must be ≤ cap for each channel
Lock Check: Locked amounts must respect floor/cap constraints
Step 3: Mathematical Optimization Use advanced mathematical algorithms to efficiently find the budget allocation that maximizes profit while respecting all business constraints." to:
Maximize: Total profit function across all channels
Subject to: All business constraints
Method: Gradient-based optimization that efficiently finds the optimal solution
Step 4: Validate and Interpret Results
Convergence Check: Ensure the algorithm found a stable solution
Constraint Analysis: Identify which constraints are "binding" (limiting further optimization)
Sensitivity Testing: Verify results are robust to small parameter changes
Understanding Optimization Results
The Allocation Table: Shows current vs. proposed spend for each channel with the business rationale:
Increase Recommendations: Channels with mROI above shadow price (shadow price being the additional profit for $1 spent regardless of channel).  
Decrease Recommendations: Channels with mROI below shadow price
No Change: Channels constrained by floors, caps, or locks
Expected Profit Impact:
Point Estimate: Best guess of profit improvement
Confidence Interval: Range accounting for parameter uncertainty (typically 90% confidence)
Incremental Profit: Additional profit from optimized allocation vs. current allocation
Binding Constraints Analysis: Identifies which business rules are preventing further optimization:
Budget-Bound: Using full budget, shadow price > 1.0 (increase budget)
Channel-Bound: Specific channels at floors/caps preventing reallocation
Ramp-Bound: Change limits preventing optimal moves
Scenario Analysis:
±10% Budget: How allocation would change with budget increase/decrease
Constraint Relaxation: Impact of removing or loosening specific constraints
Sensitivity Testing: How allocation changes with different parameter assumptions
Risk Management and Practical Implementation
Confidence Intervals on Recommendations: Every allocation recommendation includes uncertainty bounds based on parameter confidence:
High Confidence: Narrow confidence intervals, reliable recommendations
Medium Confidence: Moderate uncertainty, monitor performance closely
Low Confidence: Wide confidence intervals, consider as directional guidance only
Implementation Considerations:
Gradual Rollout: Consider implementing changes over 1 month rather than immediately
Performance Monitoring: Track actual results vs. predicted improvements
Constraint Adjustment: Be prepared to modify constraints based on operational reality
When Optimization May Not Work:
Insufficient Data: Channels below spend thresholds have unreliable parameters
External Changes: Market conditions shifting faster than model refresh cycles
Operational Constraints: Real-world limitations not captured in mathematical constraints


Mathematical Implementation
def steady_state_response_function(spend, alpha, beta, r):
    """
    Calculate steady-state response for given spend level
    """
    if spend <= 0:
        return 0
    
    steady_state_adstock = spend / (1 - r)
    response = alpha * (steady_state_adstock ** beta)
    return response

def marginal_roi_function(spend, alpha, beta, r):
    """
    Calculate marginal ROI at given spend level
    """
    if spend <= 0:
        return alpha * beta * ((1 - r) ** (-beta))
    
    marginal_roi = alpha * beta * ((1 - r) ** (-beta)) * (spend ** (beta - 1))
    return marginal_roi

def total_profit_function(spend_allocation, baseline_profit, channel_params):
    """
    Calculate total profit for given spend allocation
    """
    total_profit = baseline_profit
    
    for channel, spend in spend_allocation.items():
        if channel in channel_params:
            alpha = channel_params[channel]['alpha']
            beta = channel_params[channel]['beta']
            r = channel_params[channel]['r']
            
            contribution = steady_state_response_function(spend, alpha, beta, r)
            total_profit += contribution
    
    return total_profit

L-BFGS-B Optimization Implementation
from scipy.optimize import minimize

def optimize_budget_allocation(total_budget, channel_params, constraints, baseline_profit):
    """
    Optimize budget allocation using L-BFGS-B algorithm
    """
    channels = list(channel_params.keys())
    n_channels = len(channels)
    
    # Objective function (negative because minimize() minimizes)
    def objective(x):
        spend_allocation = dict(zip(channels, x))
        profit = total_profit_function(spend_allocation, baseline_profit, channel_params)
        return -profit  # Negative for minimization
    
    # Gradient function
    def gradient(x):
        grad = np.zeros(n_channels)
        for i, channel in enumerate(channels):
            spend = x[i]
            alpha = channel_params[channel]['alpha']
            beta = channel_params[channel]['beta']
            r = channel_params[channel]['r']
            
            # Marginal ROI = negative gradient for minimization
            grad[i] = -marginal_roi_function(spend, alpha, beta, r)
        
        return grad
    
    # Budget constraint
    budget_constraint = {
        'type': 'eq',
        'fun': lambda x: np.sum(x) - total_budget,
        'jac': lambda x: np.ones(n_channels)
    }
    
    # Build bounds from constraints
    bounds = []
    for channel in channels:
        lower_bound = constraints.get(channel, {}).get('floor', 0)
        upper_bound = constraints.get(channel, {}).get('cap', total_budget)
        
        # Handle locked channels
        if 'lock' in constraints.get(channel, {}):
            lock_value = constraints[channel]['lock']
            bounds.append((lock_value, lock_value))
        else:
            bounds.append((lower_bound, upper_bound))
    
    # Starting point: current allocation or proportional distribution
    x0 = get_starting_allocation(channels, constraints, total_budget)
    
    # Optimize
    result = minimize(
        fun=objective,
        x0=x0,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds,
        constraints=[budget_constraint],
        options={
            'ftol': 1e-6,
            'gtol': 1e-6,
            'maxiter': 1000
        }
    )
    
    return result

def calculate_shadow_price(optimization_result, channel_params):
    """
    Calculate shadow price from optimization result
    """
    if not optimization_result.success:
        return None
    
    # Shadow price equals the common mROI of unconstrained channels
    optimal_allocation = optimization_result.x
    marginal_rois = []
    
    for i, channel in enumerate(channel_params.keys()):
        spend = optimal_allocation[i]
        alpha = channel_params[channel]['alpha']
        beta = channel_params[channel]['beta']
        r = channel_params[channel]['r']
        
        mroi = marginal_roi_function(spend, alpha, beta, r)
        marginal_rois.append(mroi)
    
    # Shadow price is the common mROI level
    return np.mean(marginal_rois)

Constraint Validation and Conflict Resolution
def validate_constraints(constraints, total_budget):
    """
    Validate constraint feasibility and detect conflicts
    """
    errors = []
    warnings = []
    
    # Calculate minimum required budget
    total_floors = sum(
        constraint.get('floor', 0) 
        for constraint in constraints.values()
    )
    
    if total_floors > total_budget:
        errors.append({
            'type': 'BUDGET_INSUFFICIENT',
            'message': f'Floor constraints require ${total_floors:,.0f} but budget is ${total_budget:,.0f}',
            'required_action': f'Reduce floors by ${total_floors - total_budget:,.0f} minimum'
        })
    
    # Check individual channel conflicts
    for channel, constraint in constraints.items():
        floor = constraint.get('floor', 0)
        cap = constraint.get('cap', float('inf'))
        lock = constraint.get('lock')
        
        if floor > cap:
            errors.append({
                'type': 'CHANNEL_CONFLICT',
                'channel': channel,
                'message': f'{channel} floor (${floor:,.0f}) exceeds cap (${cap:,.0f})',
                'required_action': 'Adjust floor or cap constraints'
            })
        
        if lock is not None and (lock < floor or lock > cap):
            errors.append({
                'type': 'LOCK_CONFLICT',
                'channel': channel,
                'message': f'{channel} locked at ${lock:,.0f} violates floor/cap constraints',
                'required_action': 'Adjust lock value or floor/cap constraints'
            })
    
    return {'errors': errors, 'warnings': warnings}

Section 7: Dashboard & User Interface (COMPLETE API SPECIFICATIONS)
Business Context: What the Dashboard Accomplishes
The dashboard transforms complex statistical analysis into actionable business insights. It's designed for marketing teams who need to make data-driven budget decisions without getting lost in mathematical details.
The Business Challenge: MMM analysis generates thousands of data points - parameter estimates, confidence intervals, response curves, optimization results. Without proper visualization, this becomes overwhelming rather than helpful. The dashboard distills this complexity into clear, actionable insights.
Primary User Workflow:
Upload & Validate → 2. Monitor Training → 3. Review Results → 4. Optimize Budget → 5. Export Decisions
This workflow mirrors how marketing teams actually work - from data preparation through strategic planning to tactical execution.
What Each Chart and Table Shows You
Channel Performance Overview Table: This is your "executive summary" of channel effectiveness. Each row represents one marketing channel with the key metrics that drive budget decisions:
Current Spend: What you're spending now
Contribution: How much profit this channel actually generates (not just revenue)
mROI: Return on the last dollar spent (your efficiency indicator)
Confidence Level: How reliable these estimates are (High/Medium/Low)
Status: What the optimization recommends (Optimized/Underutilized/At Cap/Low Signal)
Business Translation:
High mROI + Underutilized Status = "Spend more here"
Low mROI + Optimized Status = "Current level is right"
Low Confidence + Low Signal = "Need more data before making major changes"
Budget Optimization Panel: Shows the financial impact of following the model's recommendations:
Expected Profit Increase: How much additional profit the optimal allocation should generate
Confidence Interval: Range of possible outcomes (90% confidence means you can be quite sure the actual result will fall within this range)
Shadow Price: The value of adding $1,000 more to your total budget
Binding Constraints: Which business rules are preventing even better results
Model Performance Time Series Chart: This chart shows actual profit vs. what the model predicted over time. It's your "trust indicator" for the model:
Tight Fit: Predicted line close to actual line = reliable model
Widening Gap: Predictions diverging from reality = model needs attention
Confidence Bands: Blue shaded area showing 90% prediction uncertainty
CV Windows: Shaded periods showing how the model performed on "unseen" data during validation
Response Curves (Channel Detail View): When you click on a specific channel, you see its response curve - how profit contribution changes with different spending levels:
Current Spend: Vertical red line showing where you are now
Proposed Spend: Vertical green line showing optimization recommendation
Saturation Point: Where the curve flattens out (diminishing returns become severe)
Confidence Bands: Blue shaded area showing 90% uncertainty in the curve shape
Business Insight from Response Curves:
Steep curve at current spend = "Room to increase"
Flat curve at current spend = "Already saturated, don't increase"
Wide confidence bands = "Uncertain about this channel's behavior"
Real-Time Training Progress Display
Why You Need Live Progress Updates: Model training takes 1-4 hours. Without progress visibility, you don't know if the system is working, stuck, or experiencing problems.
Training Status Panel Elements:
Overall Progress Bar:
Step Identification: "Cross-Validation Fold 8 of 17"
Time Estimates: "47 minutes remaining"
Current Activity: "Testing parameter combination 45/81 for Social channels"
Performance Tracking:
Current Best MAPE: How accurate the best parameter combination is so far
Parameter Stability: Whether the algorithm is finding consistent answers
Convergence Indicators: Whether the search is making progress or getting stuck
Live Log Stream: Real-time updates showing what the system is doing:
"Starting parameter grid search for TV channels..."
"Fold 12/17 complete. Current best MAPE: 18.4%"
"Bootstrap confidence intervals: 89% complete"
Model Health and Diagnostic Indicators
Model Health Status Bar: Traffic light system showing overall model reliability:
Green: High confidence, reliable recommendations
Amber: Good model with some uncertainty, monitor closely
Red: Significant issues, recommendations unreliable
Diagnostic Test Results: Shows which statistical tests passed or failed:
Autocorrelation Test: PASS (residuals don't show time patterns)
Attribution Reasonableness: PASS (media attribution is 45% - realistic)
Parameter Stability: FAIL (parameters vary significantly across time periods)
Business Translation of Diagnostics:
All Tests Pass: Trust the recommendations
1-2 Tests Fail: Use recommendations but monitor performance closely
3+ Tests Fail: Consider gathering more data or adjusting approach
Scenario Planning and What-If Analysis
Scenario Builder Interface: Allows you to test "what if" questions before making actual budget changes:
Budget Sliders: Drag to test different total budget levels
Constraint Adjustments: Modify floors, caps, and locks in real-time
Live Updates: See allocation changes immediately as you adjust constraints
Feasibility Indicators: Color-coded system showing whether your scenario is possible:
Green: Feasible scenario, optimization can find a solution
Red: Conflicting constraints (e.g., floors exceed total budget)
Yellow: Tight constraints, limited optimization flexibility
Scenario Comparison Table: Shows how different scenarios compare:
Current vs. Optimized: Baseline comparison
+10% Budget: Impact of budget increase
-10% Budget: Impact of budget decrease
Custom Constraints: Your specific "what if" scenario
Alert and Monitoring System Display
Performance Alert Dashboard: Shows when the model needs attention:
Model Drift Alerts: "Prediction accuracy has decreased 12% over past 30 days"
Attribution Stability Warnings: "Social media attribution changed 25% month-over-month"
Constraint Violation Notices: "Current spend exceeds optimal allocation by $50K"
Alert Severity Levels:
Green Info: Informational updates, no action needed
Yellow Warning: Monitor situation, consider investigation
Red Critical: Immediate attention required, recommendations may be unreliable
Export and Reporting Capabilities
Executive Summary Export: One-page summary suitable for leadership presentations:
Key findings and recommendations
Expected profit impact with confidence intervals
Top 3 action items with business rationale
Detailed Analysis Export: Complete data export for deeper analysis:
Channel-level parameters and confidence intervals
Response curve data points
Optimization results with multiple scenarios
Model diagnostics and validation results
Implementation Playbook: Step-by-step guidance for executing recommendations:
Prioritized channel adjustments
Timeline for implementation
Performance monitoring checkpoints
Rollback procedures if results don't match predictions


API Endpoint Specifications
# API Response Time Requirements
API_PERFORMANCE_TARGETS = {
    'model_status': {'target_ms': 200, 'max_ms': 500},
    'optimization_run': {'target_ms': 2000, 'max_ms': 5000},
    'response_curves_cached': {'target_ms': 500, 'max_ms': 1000},
    'response_curves_computed': {'target_ms': 3000, 'max_ms': 8000},
    'dashboard_interactions': {'target_ms': 200, 'max_ms': 500}
}

# Complete API Schema Definitions
@app.route('/api/upload/data', methods=['POST'])
def upload_data():
    """
    Handle CSV data upload with validation
    
    Request: multipart/form-data with CSV file
    Response: {
        "status": "success|error",
        "data_summary": {
            "date_range": {"start": "2025-01-01", "end": "2025-12-31"},
            "total_days": 365,
            "channels_detected": 8,
            "channel_list": ["Search_Brand", "Social_Facebook", ...],
            "total_spend": 450000.00,
            "business_tier": "enterprise",
            "data_quality_score": 92,
            "cv_folds_possible": 17
        },
        "validation_errors": [
            {
                "code": "ERROR_001", 
                "message": "Negative spend detected in Social_Facebook on 2025-03-15", 
                "severity": "error",
                "line_number": 74,
                "suggested_fix": "Check data source for Social_Facebook on 2025-03-15"
            }
        ]
    }
    """
    pass

@app.route('/api/model/train', methods=['POST'])
def train_model():
    """
    Start model training with configuration
    
    Request: {
        "client_id": "uuid",
        "training_config": {
            "cv_approach": "primary|efficient",
            "bootstrap_samples": 100,
            "early_stopping": false,
            "parameter_search": "full|quick"
        }
    }
    Response: {
        "run_id": "uuid",
        "status": "training",
        "estimated_completion": "2025-10-15T14:30:00Z",
        "progress_webhook": "/api/training/progress/{run_id}",
        "total_folds": 17,
        "estimated_duration_minutes": 156
    }
    """
    pass

@app.route('/api/training/progress/<run_id>', methods=['GET'])
def get_training_progress(run_id):
    """
    Get current training progress
    
    Response: {
        "run_id": "uuid",
        "status": "training|completed|failed|cancelled",
        "progress": {
            "current_step": "Cross-Validation",
            "current_fold": 8,
            "total_folds": 17,
            "current_channel_type": "social",
            "current_combination": 45,
            "total_combinations": 81,
            "step_progress": 0.47,
            "overall_progress": 0.65,
            "estimated_completion": "2025-10-15T14:30:00Z",
            "elapsed_minutes": 89
        },
        "current_best_mape": 23.4,
        "parameter_stability": "stable|moderate|unstable",
        "logs": [
            {
                "timestamp": "2025-10-15T13:15:23Z", 
                "level": "INFO",
                "message": "Starting parameter grid search for Social channels..."
            },
            {
                "timestamp": "2025-10-15T13:18:45Z",
                "level": "INFO", 
                "message": "Fold 5/17 complete. Current best MAPE: 23.4%"
            }
        ]
    }
    """
    pass

@app.route('/api/model/results/<run_id>', methods=['GET'])
def get_model_results(run_id):
    """
    Get complete model results after training
    
    Response: {
        "run_id": "uuid",
        "model_performance": {
            "cv_mape": 21.8,
            "cv_mape_std": 3.2,
            "r_squared": 0.67,
            "shadow_price": 2.34,
            "business_tier": "enterprise",
            "confidence_level": "high",
            "diagnostic_tests": {
                "total_passed": 9,
                "total_tests": 11,
                "failed_tests": ["heteroscedasticity", "attribution_stability"],
                "test_details": {
                    "autocorrelation": {"status": "pass", "statistic": 1.87, "threshold": "1.5-2.5"},
                    "normality": {"status": "pass", "p_value": 0.12, "threshold": ">0.05"},
                    "heteroscedasticity": {"status": "fail", "p_value": 0.03, "threshold": ">0.05"}
                }
            }
        },
        "channel_performance": [
            {
                "channel_name": "Search_Brand",
                "channel_type": "search_brand",
                "current_spend": 45200.00,
                "contribution": 87400.00,
                "contribution_pct": 23.5,
                "marginal_roi": 1.93,
                "confidence": "high",
                "confidence_interval": {"lower": 1.64, "upper": 2.22},
                "parameters": {
                    "alpha": 234.5,
                    "beta": 0.62,
                    "r": 0.15,
                    "is_estimated": true
                },
                "annual_spend": 541200.00,
                "meets_threshold": true
            }
        ],
        "baseline_attribution": {
            "baseline_profit": 125000.00,
            "baseline_pct": 58.4,
            "media_profit": 89000.00,
            "media_pct": 41.6,
            "total_modeled_profit": 214000.00
        },
        "parameter_stability": {
            "overall_score": "stable",
            "winner_agreement_rate": 0.76,
            "bootstrap_cv_avg": 0.12,
            "flip_rate": 0.18
        }
    }
    """
    pass

@app.route('/api/data/curves/<run_id>/<channel_name>', methods=['GET'])
def get_response_curves(run_id, channel_name):
    """
    Get response and mROI curves for specific channel
    
    Query parameters:
    - n_points: Number of points in curve (default: 30, max: 100)
    - max_spend: Maximum spend for curve (default: auto-calculated)
    - include_confidence: Include confidence bands (default: true)
    
    Response: {
        "channel_name": "Search_Brand",
        "channel_type": "search_brand", 
        "curve_data": {
            "spend_points": [0, 1000, 2000, ..., 75000],
            "contribution_points": [0, 1834.5, 3421.2, ..., 45231.8],
            "marginal_roi_points": [2.45, 2.12, 1.93, ..., 0.87],
            "confidence_bands": {
                "contribution_lower": [0, 1654.1, 3089.8, ...],
                "contribution_upper": [0, 2014.9, 3752.6, ...],
                "marginal_roi_lower": [2.21, 1.91, 1.74, ...],
                "marginal_roi_upper": [2.69, 2.33, 2.12, ...]
            }
        },
        "current_spend": 45200.00,
        "current_contribution": 87400.00,
        "current_marginal_roi": 1.93,
        "optimal_spend_range": {"min": 35000, "max": 65000},
        "saturation_point": 120000.00,
        "observed_data": [
            {"spend": 42000.00, "contribution": 82100.00, "date": "2025-10-10"},
            {"spend": 47000.00, "contribution": 91200.00, "date": "2025-10-11"}
        ],
        "cache_info": {
            "cached": true,
            "generated_at": "2025-10-15T14:22:33Z",
            "expires_at": "2025-10-16T14:22:33Z"
        }
    }
    """
    pass

@app.route('/api/optimization/run', methods=['POST'])
def run_optimization():
    """
    Execute budget optimization with constraints
    
    Request: {
        "run_id": "uuid",
        "scenario_config": {
            "scenario_name": "Q4_2025_Plan",
            "total_budget": 500000.00,
            "constraints": [
                {"channel": "Search_Brand", "type": "floor", "value": 30000.00},
                {"channel": "Social_Facebook", "type": "cap", "value": 75000.00},
                {"channel": "TV_YouTube", "type": "lock", "value": 45000.00}
            ],
            "ramp_limit_pct": 0.20,
            "planning_horizon": "steady_state"
        }
    }
    
    Response: {
        "scenario_id": "uuid",
        "optimization_status": "completed|failed|infeasible",
        "computation_time_ms": 1247,
        "results": {
            "feasibility": {
                "is_feasible": true,
                "constraint_violations": [],
                "warnings": []
            },
            "performance": {
                "expected_profit_delta": 12500.00,
                "confidence_interval": {"lower": 8900.00, "upper": 16100.00},
                "current_total_profit": 214000.00,
                "optimized_total_profit": 226500.00,
                "improvement_pct": 5.8
            },
            "shadow_price_analysis": {
                "shadow_price": 2.1,
                "interpretation": "healthy_range",
                "budget_recommendations": {
                    "increase_profitable_up_to": 750000.00,
                    "decrease_acceptable_down_to": 400000.00
                }
            },
            "binding_constraints": [
                {
                    "constraint": "Search_Brand_floor",
                    "channel": "Search_Brand",
                    "type": "floor",
                    "value": 30000.00,
                    "shadow_price_impact": 0.15
                }
            ],
            "channel_allocations": [
                {
                    "channel": "Search_Brand",
                    "current": 45200.00,
                    "proposed": 52000.00,
                    "delta_abs": 6800.00,
                    "delta_pct": 15.0,
                    "marginal_roi": 2.1,
                    "status": "optimized",
                    "confidence": "high",
                    "constraint_status": "unconstrained"
                }
            ]
        },
        "sensitivity_analysis": {
            "parameter_robustness": {
                "flip_rate": 0.12,
                "allocation_stability": "high"
            },
            "budget_stress_test": {
                "minus_10_pct": {"profit_delta": 11250.00, "feasible": true},
                "plus_10_pct": {"profit_delta": 13750.00, "feasible": true}
            }
        }
    }
    """
    pass

# WebSocket Event Schemas
WEBSOCKET_EVENT_SCHEMAS = {
    'training_progress': {
        'event': 'training_progress',
        'data': {
            'run_id': 'uuid',
            'overall_progress': 0.65,
            'current_step': 'Cross-Validation',
            'current_fold': 8,
            'total_folds': 17,
            'step_detail': 'Testing parameter combination 45/81 for Social channels',
            'estimated_completion': ''2025-10-15T14:30:00Z',
'current_best_mape': 23.4, 
'parameter_stability': 'stable', 
'bootstrap_progress': 0.89
        }
    },
    'fold_complete': {
        'event': 'fold_complete',
        'data': {
            'run_id': 'uuid',
            'fold_id': 8,
            'fold_mape': 22.1,
            'current_best_mape': 21.8,
            'winning_parameters': {
                'search_brand': {'beta': 0.62, 'r': 0.15},
                'social': {'beta': 0.45, 'r': 0.35}
            }
        }
    },
    'training_complete': {
        'event': 'training_complete',
        'data': {
            'run_id': 'uuid',
            'status': 'success',
            'final_mape': 21.8,
            'total_duration_minutes': 142,
            'model_quality': 'excellent'
        }
    },
    'training_error': {
        'event': 'training_error',
        'data': {
            'run_id': 'uuid',
            'error_type': 'convergence_failure|insufficient_data|computation_error',
            'error_message': 'Parameter search failed to converge for TV_Video channels',
            'recovery_suggestions': ['Reduce parameter grid density', 'Check data quality for TV channels']
        }
    }
}

Dashboard Component Specifications
# React Component Props and State Management
DASHBOARD_COMPONENT_SPECS = {
    'TrainingProgressComponent': {
        'props': {
            'runId': 'string',
            'autoRefresh': 'boolean',
            'onComplete': 'function',
            'onError': 'function'
        },
        'state': {
            'progress': 'ProgressState',
            'logs': 'LogEntry[]',
            'isConnected': 'boolean',
            'error': 'string|null'
        },
        'websocket_events': ['training_progress', 'fold_complete', 'training_complete', 'training_error']
    },
    'ChannelPerformanceTable': {
        'props': {
            'channels': 'ChannelPerformance[]',
            'sortBy': 'string',
            'filterBy': 'string',
            'showConfidence': 'boolean'
        },
        'interactions': ['sort', 'filter', 'view_curve', 'adjust_constraints'],
        'update_frequency': '5_seconds'
    },
    'OptimizationPanel': {
        'props': {
            'currentAllocation': 'Allocation',
            'constraints': 'Constraint[]',
            'onOptimize': 'function'
        },
        'real_time_validation': true,
        'response_time_target': '200ms'
    }
}

Section 8: Acceptance Criteria (UPDATED) - Business-Context-Aware Validation
8.1: Business-Tiered Predictive Performance Standards
Enterprise Tier (>$2MM annual spend):
Excellent: CV MAPE ≤ 20% AND directional accuracy >85%
Good: CV MAPE 20-30% AND directional accuracy >75%
Poor: CV MAPE >30% OR directional accuracy ≤75%
Mid-Market Tier ($500K-$2MM annual spend):
Excellent: CV MAPE ≤ 25% AND directional accuracy >80%
Good: CV MAPE 25-35% AND directional accuracy >70%
Poor: CV MAPE >35% OR directional accuracy ≤70%
Small Business Tier ($200K-$500K annual spend):
Acceptable: CV MAPE ≤ 35% AND directional accuracy >70%
Poor: CV MAPE >35% OR directional accuracy ≤70%
Directional Accuracy Definition: Percentage of CV test periods where predicted change direction (increase/decrease vs. previous period) matches actual change direction.
Additional Performance Requirements:
Minimum R²: ≥ 0.25 (model explains at least 25% of profit variance)
Baseline vs. Media Attribution: 20% ≤ media_contribution ≤ 80% of total modeled profit
Monthly Trend Stability: No systematic degradation >5 percentage points over 3+ consecutive months
8.2: MMM-Specific Diagnostic Tests
Enhanced Residual Analysis (7 Tests Total):
Standard Regression Diagnostics:
Autocorrelation Test: Durbin-Watson statistic 1.5-2.5
Fail: Residuals show time-based patterns model isn't capturing
Normality Test: Shapiro-Wilk p-value >0.05
Fail: Non-normal residuals suggest model misspecification
Heteroscedasticity Test: Breusch-Pagan p-value >0.05
Fail: Error variance changes with fitted values
Systematic Bias Test: |mean(residuals)| < 2% of mean(actual_profit)
Fail: Systematic over/under-prediction
Time Trend Test: Linear regression of residuals vs. time p-value >0.05
Fail: Model degrading over time
MMM-Specific Diagnostics: 6. Channel Multicollinearity Test: Variance Inflation Factor <3.0 for all transformed channels
Calculation: VIF on [Adstock_t_c ^ beta_c] for all channels
Fail: High correlation between adstocked/saturated channels prevents reliable attribution
Attribution Reasonableness Test: 20% ≤ (media_contribution / total_modeled_profit) ≤ 80%
Calculation: sum(channel_contributions) / (baseline + sum(channel_contributions))
Fail: Model attributes unrealistic proportion to paid media vs. organic factors
Diagnostic Quality Score:
Pass 6-7 tests: "Excellent model diagnostics"
Pass 4-5 tests: "Good model diagnostics with minor issues"
Pass 2-3 tests: "Acceptable diagnostics - monitor closely"
Pass 0-1 tests: "Poor model diagnostics - review required"
8.3: Statistically-Founded Parameter Stability
Cross-Validation Parameter Stability:
Winner Agreement Rate (with confidence intervals):
Calculation: % of CV folds selecting same (beta, r) combination as global winner
Statistical Testing: Binomial test against null hypothesis of random selection
Thresholds:
Stable: Agreement rate significantly >random (p<0.05) AND rate ≥60%
Moderate: Agreement rate ≥40% but not significantly >random
Unstable: Agreement rate <40% OR significantly ≤random
Bootstrap Parameter Confidence:
Method: 100 bootstrap samples per channel type
Metric: Confidence interval width relative to point estimate
Calculation: (upper_ci - lower_ci) / point_estimate for beta and r parameters
Thresholds:
High confidence: Relative CI width <0.30 for both beta and r
Medium confidence: Relative CI width 0.30-0.60 for either parameter
Low confidence: Relative CI width >0.60 for any parameter
Allocation Robustness (with statistical foundations):
Flip Rate Test: % of channels changing direction under parameter jitter


Stable: <20% (model recommendations robust to parameter uncertainty)
Moderate: 20-40% (some sensitivity, monitor allocation changes)
Unstable: >40% (high sensitivity, low confidence in recommendations)
Churn Magnitude Test: Average |allocation_change| / current_allocation under jitter


Low sensitivity: <15% (minor adjustments)
Moderate sensitivity: 15-30% (significant but manageable)
High sensitivity: >30% (major shifts, review model reliability)
8.4: Business-Context Shadow Price Validation
Industry-Informed Shadow Price Ranges:
B2B/Enterprise (longer sales cycles, higher LTV):
Healthy range: 1.0 - 3.0
Investigate if: <1.0 (potential over-spending) or >3.0 (potential under-spending)
E-commerce/DTC (shorter cycles, volume-driven):
Healthy range: 1.0 - 3.0
Investigate if: <1.0 (over-spending likely) or >3.0 (significant opportunity)
Lead Generation (cost-per-lead focus):
Healthy range: 1.0 - 3.0
Investigate if: <1.0 (efficiency issues) or >3.0 (capacity constraints)
Growth-Stage Businesses (market share focus):
Acceptable range: 0.5 - 10.0 (wider tolerance for strategic growth spending)
Red flag only if: <0.3 or >15.0
Shadow Price Interpretation Checks:
Economic Consistency: Shadow price represents realistic marginal ROI for the business model
Temporal Stability: Shadow price doesn't fluctuate >100% month-over-month without business explanation
Cross-Channel Validity: Unconstrained channels converge to shadow price within ±10%
8.5: Channel Response Realism Validation
Saturation Curve Sanity Checks:
Beta Parameter Realism:


Search Brand: 0.4 ≤ beta ≤ 0.8 (moderate to low saturation expected)
Search Non-Brand: 0.5 ≤ beta ≤ 0.9 (should be more linear than brand)
Social/Display: 0.3 ≤ beta ≤ 0.7 (significant saturation expected)
TV/Video: 0.2 ≤ beta ≤ 0.6 (strong saturation due to reach limits)
Adstock Parameter Realism:


Search: 0.0 ≤ r ≤ 0.3 (short memory, immediate response)
Social: 0.2 ≤ r ≤ 0.5 (moderate memory, social proof effects)
TV/Video: 0.3 ≤ r ≤ 0.7 (longer memory, brand building)
Display: 0.1 ≤ r ≤ 0.4 (varies by format and audience)
Response Magnitude Check:


No channel should contribute >60% of total media attribution (diversification check)
Contribution per dollar should decrease monotonically with spend level (diminishing returns)
mROI at current spend should be >0.2 for included channels (minimum efficiency)
8.6: Temporal Model Stability
Parameter Drift Detection:
Rolling Window Analysis: Compare parameters estimated on first 60% vs. last 60% of data
Drift Tolerance: Parameter changes <20% indicate stable model
Structural Break Test: Chow test for parameter stability across time periods
Seasonality Validation: Model performance consistent across different seasonal periods
Performance Degradation Monitoring:
Out-of-Time Validation: Latest 30 days performance vs. historical CV performance
Alert Threshold: >10 percentage point MAPE degradation triggers retraining recommendation
Trend Detection: Monitor for systematic performance decline over rolling 90-day windows
8.7: Business Impact Validation
Recommendation Reasonableness:
Change Magnitude: Proposed allocation changes should be operationally feasible


Flag recommendations requiring >50% change in any channel
Validate that proposed changes align with channel capacity constraints
ROI Ordering: Higher mROI channels should receive preference in allocation increases


Exception: Channels at capacity constraints may have artificially high mROI
Historical Context: Recommendations should align with successful historical patterns


Flag if optimization suggests dramatically different allocation vs. best historical periods
Economic Logic Validation:
Profit Impact Realism: Projected profit improvements should align with business economics
Market Share Implications: Large budget increases should consider market saturation
Competitive Response: Major allocation shifts should consider competitive dynamics
8.8: Production Readiness Gates
Model Confidence Requirements:
Overall Grade: Must achieve "Good" or better in all validation categories
Red Flag Tolerance: No more than 1 "Poor" rating across all acceptance criteria
Business Tier Compliance: Must meet performance standards for client's business tier
Stakeholder Sign-off: Model outputs reviewed and approved by client business stakeholders
Monitoring & Alerting Setup:
Automated Diagnostics: Weekly runs of all acceptance tests with email alerts
Performance Tracking: Monthly reports on model degradation metrics
Parameter Stability: Quarterly assessment of parameter drift and recommendation consistency
Business Impact Tracking: Ongoing validation of actual vs. predicted performance
Documentation Requirements:
Model Card: Summary of performance, limitations, and appropriate use cases
Validation Report: Detailed results of all acceptance criteria tests
Business Context: Client-specific factors affecting model interpretation
Update Schedule: Recommended retraining frequency based on model stability metrics
Section 9: Simplifying Assumptions (v1)
Data & Input Simplifications:
Spend-only inputs: No impressions, clicks, or reach data required
Daily granularity: No sub-daily or weekly aggregation options
Single KPI: Profit only (client-defined) - no multi-objective optimization
Limited external factors: Only basic event flags (holiday, promo, outage) - no weather, competitors, macroeconomic variables
Mathematical Model Simplifications:
Single curve family: Power law saturation (x^beta) with geometric adstock
No interaction effects: Channels operate independently
Additive profit structure: Profit = baseline + sum(channel_contributions)
Linear baseline: Simple trend component (no seasonality beyond event flags)
Parameter Estimation Simplifications:
Classical regression: No Bayesian methods or prior incorporation
Individual channel parameters: Each channel gets own beta/r if above spend threshold
Static parameters: Fixed coefficients during each model fit - no seasonal parameter shifts
Grid search only: No continuous optimization or gradient-based parameter search
Fixed cross-validation scheme: 126/14 day train/test split
Business Logic Simplifications:
Hard constraints only: Floors, caps, locks, ramp limits
Steady-state planning: Single planning mode - no finite-horizon or dynamic planning
Equal channel treatment: No channel priorities, strategic importance weights, or business preferences
Single budget period: Daily optimization focus
No portfolio constraints: No requirements like "Digital ≥ 60% of budget"
Technical Implementation Simplifications:
Single-tenant deployment: Separate instances per client - no multi-tenant architecture
Batch processing: Monthly model refresh - no real-time parameter updates
Limited export formats: CSV/PNG/JSON only - no API integrations or automated reporting
Basic user management: Simple authentication - no role-based permissions or team collaboration
File-based data input: Manual upload - no automated data connectors
Validation & Uncertainty Simplifications:
Bootstrap confidence intervals: 90% confidence bands on key metrics
Simple sensitivity analysis: Parameter jitter testing
Basic stability checks: Cross-validation agreement metrics
Standard diagnostics: Regression diagnostics without advanced time series tests
User Interface Simplifications:
Desktop-optimized: Limited mobile responsiveness
English only: No internationalization
Manual scenario management: Save/load scenarios - no automated A/B testing
What We're Explicitly NOT Building in v1:
Multi-geo modeling: No regional or market-level attribution
Product-level attribution: Single business/product focus
Incrementality testing integration: No geo-lift or holdout experiment connections
Competitive intelligence: No competitive spend tracking
Creative-level insights: No creative fatigue or message-level attribution
Channel sub-attribution: No keyword-level or campaign-level granularity
Time-varying coefficients: No seasonal parameters or concept drift
Multi-objective optimization: No balancing of profit vs. growth vs. brand metrics
Section 10: Glossary (Plain Words)
Core Model Terms:
Adstock (r): "Memory" of advertising - how long the effect of today's ad spend lasts. 0 = immediate effect only, closer to 1 = very long memory. Example: r = 0.3 means 30% of today's impact carries over to tomorrow.
Alpha (channel strength): How effective each channel is at generating profit. Higher alpha = more profit per dollar spent. Always positive (no negative ROI channels allowed).
Beta (diminishing returns): How quickly a channel becomes less effective as you spend more. Beta = 1 is linear (double spend = double impact), beta = 0.5 means you need 4x spend to get 2x impact.
Baseline: Profit expected without any paid advertising. Includes organic growth trends and seasonal factors. We model this as organic business trajectory rather than assuming it's constant.
Confidence Interval: Range showing uncertainty in our estimates. "90% confidence" means we're 90% sure the true value falls within this range. Wider ranges = more uncertainty.
Shadow Price: The profit you'd get from one more dollar of budget, allocated optimally. If shadow price = 2.3, the next $1,000 optimally spent would generate $2,300 in profit.
Model Performance Terms:
CV MAPE (Cross-Validation Mean Absolute Percentage Error): How accurate our predictions are on data the model hasn't seen. Lower is better. 15% MAPE means predictions are typically within 15% of actual results.
Directional Accuracy: Percentage of time periods where we correctly predict whether profit will go up or down compared to the previous period.
R-squared: Percentage of profit variation explained by the model. R² = 0.6 means the model explains 60% of why profit goes up and down.
Residuals: Difference between actual profit and model predictions. Good models have small, random residuals with no patterns.
Business Terms:
Contribution: How much profit each channel generated. Calculated using the channel's spend after applying adstock and saturation transformations.
Free Channels: Channels not constrained by floors, caps, or locks during optimization. These channels can have their budgets adjusted freely.
mROI (Marginal Return on Investment): Extra profit from spending one more dollar in a channel at its current spending level. Used to compare channel efficiency.
Steady-State: Planning assumption that spending continues at the same level long-term, allowing full adstock effects to build up.
Optimization Terms:
Binding Constraint: A business rule (floor, cap, lock, ramp limit) that's actively limiting the optimal allocation. Example: "Search capped at $10k" if the optimizer wants to spend more.
Equalization: In optimal allocation, all free channels should have the same mROI. This is the mathematical principle behind budget optimization.
Ramp Limit: Maximum allowed change in channel spend vs. current levels. Default ±20% prevents dramatic budget shifts that might be operationally difficult.
Data Quality Terms:
Business Tier: Categories based on annual spend that determine reliability expectations: Enterprise (>$500K), Mid-Market ($200K-$500K), Small Business ($50K-$200K).
Channel Type: Categories for setting default parameters. Examples: Search Brand, Search Non-Brand, Social, TV/Video, Display.
Parameter Stability: How consistent parameter estimates are across different data samples. Stable parameters = reliable model.
Spend Threshold: Minimum spend required for individual parameter estimation. Channels below threshold use default parameters.
Technical Terms (Simplified):
Bootstrap: Statistical method for estimating confidence intervals by repeatedly resampling data and refitting the model.
Cross-Validation: Testing model accuracy by training on part of the data and testing on the remaining part. Prevents overfitting.
Walk-Forward Validation: Time series cross-validation that respects temporal order - never use future data to predict the past.
Dashboard Terms:
Churn Rate: Percentage of channels that change direction (increase vs decrease) when model parameters are slightly adjusted. High churn = unstable recommendations.
Flip Rate: Similar to churn rate - measures allocation stability under parameter uncertainty.
Model Diagnostics: Statistical tests that check whether the model assumptions are reasonable and the results are trustworthy.
Common Business Questions:
"Should I increase my budget?" Look at shadow price - if >1.0, yes. If <1.0, consider reducing unless focused on growth over profit.
"Which channels should get more money?" Channels with highest mROI that aren't at their caps.
"How confident should I be in these recommendations?" Check confidence intervals, parameter stability scores, and data quality indicators.
"Why did my channel's ROI estimate change?" Could be new data, parameter updates from monthly retraining, or changes in other channels affecting the overall model.
Section 11: What We're Not Doing in v1
Advanced Attribution Methods:
Incrementality Testing Integration: No direct connection to geo-lift tests, holdout experiments, or A/B testing frameworks
Creative-Level Attribution: No analysis of ad creative performance, message testing, or creative fatigue effects
Competitive Intelligence: No tracking of competitor spend levels, market share impacts, or competitive response modeling
Multi-Touch Attribution: No customer journey modeling or path-to-conversion analysis
Advanced Modeling Techniques:
Hierarchical Modeling: No multi-level models for regions, products, or customer segments
Time-Varying Coefficients: No seasonal parameter shifts, concept drift modeling, or adaptive coefficients
Bayesian Methods: No prior incorporation, uncertainty quantification through Bayesian inference
Interaction Effects: No synergy or cannibalization modeling between channels
Advanced Saturation Curves: No Hill transformation, S-curves, or threshold effects
Business Logic Extensions:
Multi-Objective Optimization: No balancing of profit vs. growth vs. brand metrics
Portfolio Constraints: No rules like "Digital must be ≥60% of budget"
Dynamic Planning: No finite-horizon optimization, seasonal budget allocation, or campaign scheduling
Strategic Constraints: No brand safety considerations, competitive response factors
Data & Integration:
Real-Time Data Ingestion: No automated data connectors, API integrations, or streaming data processing
Multi-Source Data Integration: No automatic joining of spend data from different platforms
Advanced External Variables: No weather data, macroeconomic indicators, competitive intelligence
Sub-Channel Granularity: No keyword-level, audience-level, campaign-level, or creative-level analysis
Platform & Architecture:
Multi-Tenant Architecture: No shared infrastructure - each client gets separate deployment
Mobile Applications: No native mobile apps or mobile-optimized interfaces
API Access: No programmatic access, webhook integrations, or third-party tool connections
Advanced User Management: No role-based permissions, approval workflows, or team collaboration
Enterprise Features:
Single Sign-On (SSO): No enterprise authentication integration
Advanced Security: Basic authentication only - no advanced encryption or compliance frameworks
Data Governance: No data lineage tracking, retention policies, or regulatory compliance tools
Custom Modeling: No client-specific model modifications or bespoke algorithm development
Integration Ecosystem:
BI Tool Connections: No direct integration with Tableau, PowerBI, or other business intelligence platforms
Marketing Platform APIs: No direct connections to Google Ads, Facebook Ads Manager, or other advertising platforms
CRM Integration: No customer data platform connections or customer lifetime value modeling
Data Warehouse Connectivity: No direct database connections or cloud data warehouse integration
Why These Limitations:
MVP Focus: Prioritizing core MMM functionality over advanced features
Market Validation: Testing demand for core features before investing in advanced capabilities
Technical Debt Avoidance: Building solid foundation before adding sophisticated features
User Experience: Keeping interface simple for marketing teams
Section 12: Complete Bootstrap Implementation
Residual Bootstrap Algorithm
def compute_bootstrap_confidence_intervals(model_data, n_bootstrap=100, confidence_level=0.90):
    """
    Compute confidence intervals using residual bootstrap method
    """
    # Fit base model and extract components
    base_model = fit_mmm_model(model_data)
    residuals = base_model.residuals
    fitted_values = base_model.fitted_values
    
    # Storage for bootstrap results
    bootstrap_params = []
    bootstrap_allocations = []
    bootstrap_curves = []
    
    # Progress tracking
    print(f"Starting bootstrap with {n_bootstrap} samples...")
    
    for i in range(n_bootstrap):
        try:
            # Resample residuals (preserving time order)
            resampled_residuals = np.random.choice(
                residuals, 
                size=len(residuals), 
                replace=True
            )
            
            # Create bootstrap response variable
            bootstrap_profit = fitted_values + resampled_residuals
            
            # Ensure bootstrap profit is non-negative
            bootstrap_profit = np.maximum(bootstrap_profit, 0.01)
            
            # Create bootstrap dataset
            bootstrap_data = model_data.copy()
            bootstrap_data['profit'] = bootstrap_profit
            
            # Refit model with bootstrap data
            bootstrap_model = fit_mmm_model(bootstrap_data)
            bootstrap_params.append(bootstrap_model.parameters)
            
            # Compute allocation for this bootstrap sample
            allocation = optimize_allocation(
                bootstrap_model, 
                current_constraints,
                total_budget
            )
            bootstrap_allocations.append(allocation)
            
            # Generate response curves
            curves = generate_response_curves(bootstrap_model)
            bootstrap_curves.append(curves)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Bootstrap progress: {i + 1}/{n_bootstrap}")
                
        except Exception as e:
            print(f"Bootstrap sample {i} failed: {str(e)}")
            continue
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Parameter confidence intervals
    param_ci = {}
    for param_name in bootstrap_params[0].keys():
        param_values = [params[param_name] for params in bootstrap_params]
        param_ci[param_name] = {
            'mean': np.mean(param_values),
            'lower': np.percentile(param_values, lower_percentile),
            'upper': np.percentile(param_values, upper_percentile),
            'std': np.std(param_values),
            'cv': np.std(param_values) / np.mean(param_values)
        }
    
    # Allocation confidence intervals
    allocation_ci = {}
    for channel in bootstrap_allocations[0].keys():
        channel_allocations = [alloc[channel] for alloc in bootstrap_allocations]
        allocation_ci[channel] = {
            'mean': np.mean(channel_allocations),
            'lower': np.percentile(channel_allocations, lower_percentile),
            'upper': np.percentile(channel_allocations, upper_percentile),
            'std': np.std(channel_allocations)
        }
    
    # Response curve confidence bands
    curve_ci = {}
    for channel in bootstrap_curves[0].keys():
        channel_curves = [curves[channel] for curves in bootstrap_curves]
        curve_ci[channel] = {
            'contribution_lower': np.percentile(
                [curve['contribution'] for curve in channel_curves], 
                lower_percentile, axis=0
            ),
            'contribution_upper': np.percentile(
                [curve['contribution'] for curve in channel_curves], 
                upper_percentile, axis=0
            ),
            'marginal_roi_lower': np.percentile(
                [curve['marginal_roi'] for curve in channel_curves], 
                lower_percentile, axis=0
            ),
            'marginal_roi_upper': np.percentile(
                [curve['marginal_roi'] for curve in channel_curves], 
                upper_percentile, axis=0
            )
        }
    
    return BootstrapResults(
        parameter_confidence=param_ci,
        allocation_confidence=allocation_ci,
        curve_confidence=curve_ci,
        n_successful_samples=len(bootstrap_params),
        n_failed_samples=n_bootstrap - len(bootstrap_params)
    )

# Enhanced Bootstrap Methods (Beyond MVP)
def stratified_temporal_bootstrap(model_data, n_bootstrap=50, season_months=3):
    """
    Bootstrap with temporal stratification to preserve seasonal patterns
    """
    # Group data by season
    model_data['month'] = model_data['date'].dt.month
    model_data['season'] = model_data['month'] // season_months
    
    bootstrap_results = []
    
    for i in range(n_bootstrap):
        bootstrap_data = []
        
        # Sample from each season proportionally
        for season in model_data['season'].unique():
            season_data = model_data[model_data['season'] == season]
            n_season_samples = len(season_data)
            
            # Bootstrap sample within season
            season_sample = season_data.sample(
                n=n_season_samples, 
                replace=True
            )
            bootstrap_data.append(season_sample)
        
        # Combine seasonal samples
        full_bootstrap = pd.concat(bootstrap_data).sort_values('date')
        
        # Fit model and store results
        bootstrap_model = fit_mmm_model(full_bootstrap)
        bootstrap_results.append(bootstrap_model.parameters)
    
    return bootstrap_results

def block_bootstrap(residuals, block_size=7, n_bootstrap=50):
    """
    Block bootstrap to preserve autocorrelation structure
    """
    n_obs = len(residuals)
    n_blocks = n_obs // block_size
    
    bootstrap_residuals = []
    
    for i in range(n_bootstrap):
        # Sample blocks with replacement
        sampled_blocks = []
        
        for j in range(n_blocks):
            block_start = np.random.randint(0, n_obs - block_size + 1)
            block = residuals[block_start:block_start + block_size]
            sampled_blocks.append(block)
        
        # Concatenate blocks
        bootstrap_sample = np.concatenate(sampled_blocks)[:n_obs]
        bootstrap_residuals.append(bootstrap_sample)
    
    return bootstrap_residuals

Section 13: Enhanced Diagnostic Testing
Complete MMM-Specific Diagnostic Implementation
def run_mmm_diagnostic_tests(model_results, data, transformed_channels):
    """
    Run complete suite of MMM-specific diagnostic tests
    """
    diagnostic_results = {}
    
    # Standard regression diagnostics
    diagnostic_results.update(run_standard_diagnostics(model_results, data))
    
    # MMM-specific advanced diagnostics
    diagnostic_results.update(run_mmm_specific_diagnostics(
        model_results, data, transformed_channels
    ))
    
    # Calculate overall diagnostic score
    total_tests = len(diagnostic_results)
    passed_tests = sum(1 for result in diagnostic_results.values() if result['status'] == 'pass')
    
    diagnostic_score = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': total_tests - passed_tests,
        'score_percentage': (passed_tests / total_tests) * 100,
        'overall_grade': calculate_diagnostic_grade(passed_tests, total_tests)
    }
    
    return {
        'individual_tests': diagnostic_results,
        'summary': diagnostic_score
    }

def run_standard_diagnostics(model_results, data):
    """
    Standard regression diagnostic tests
    """
    residuals = model_results.residuals
    fitted_values = model_results.fitted_values
    actual_profit = data['profit']
    
    tests = {}
    
    # 1. Autocorrelation Test
    dw_statistic = durbin_watson(residuals)
    tests['autocorrelation'] = {
        'status': 'pass' if 1.5 <= dw_statistic <= 2.5 else 'fail',
        'statistic': dw_statistic,
        'threshold': '1.5-2.5',
        'interpretation': 'Tests for serial correlation in residuals'
    }
    
    # 2. Normality Test
    shapiro_stat, shapiro_p = shapiro(residuals)
    tests['normality'] = {
        'status': 'pass' if shapiro_p > 0.05 else 'fail',
        'statistic': shapiro_stat,
        'p_value': shapiro_p,
        'threshold': '>0.05',
        'interpretation': 'Tests if residuals are normally distributed'
    }
    
    # 3. Heteroscedasticity Test
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, fitted_values.reshape(-1, 1))
    tests['heteroscedasticity'] = {
        'status': 'pass' if bp_p > 0.05 else 'fail',
        'statistic': bp_stat,
        'p_value': bp_p,
        'threshold': '>0.05',
        'interpretation': 'Tests for constant error variance'
    }
    
    # 4. Systematic Bias Test
    mean_residual = np.mean(residuals)
    mean_profit = np.mean(actual_profit)
    bias_percentage = abs(mean_residual) / mean_profit * 100
    tests['systematic_bias'] = {
        'status': 'pass' if bias_percentage < 2.0 else 'fail',
        'bias_percentage': bias_percentage,
        'threshold': '<2.0%',
        'interpretation': 'Tests for systematic over/under-prediction'
    }
    
    # 5. Time Trend Test
    time_trend = np.arange(len(residuals))
    trend_corr, trend_p = pearsonr(residuals, time_trend)
    tests['time_trend'] = {
        'status': 'pass' if trend_p > 0.05 else 'fail',
        'correlation': trend_corr,
        'p_value': trend_p,
        'threshold': '>0.05',
        'interpretation': 'Tests for time-based trends in residuals'
    }
    
    return tests

def run_mmm_specific_diagnostics(model_results, data, transformed_channels):
    """
    MMM-specific advanced diagnostic tests
    """
    tests = {}
    parameters = model_results.parameters
    
    # 6. Channel Multicollinearity Test
    vif_scores = calculate_vif(transformed_channels)
    max_vif = max(vif_scores.values())
    tests['multicollinearity'] = {
        'status': 'pass' if max_vif < 3.0 else 'fail',
        'max_vif': max_vif,
        'vif_scores': vif_scores,
        'threshold': '<3.0',
        'interpretation': 'Tests for excessive correlation between transformed channels'
    }
    
    # 7. Attribution Reasonableness Test
    baseline_contribution = model_results.baseline_contribution
    media_contribution = model_results.total_media_contribution
    total_modeled = baseline_contribution + media_contribution
    media_percentage = media_contribution / total_modeled * 100
    
    tests['attribution_reasonableness'] = {
        'status': 'pass' if 20 <= media_percentage <= 80 else 'fail',
        'media_percentage': media_percentage,
        'baseline_percentage': 100 - media_percentage,
        'threshold': '20-80%',
        'interpretation': 'Tests if media attribution is within reasonable bounds'
    }
    
    # 8. Adstock Decay Validation
    adstock_validity = {}
    for channel, params in parameters.items():
        if 'r' in params:
            r_value = params['r']
            # Calculate days for 90% decay
            decay_days = np.log(0.1) / np.log(r_value) if r_value > 0 else 0
            
            # Check if decay is realistic (within 60 days for most channels)
            is_realistic = 1 <= decay_days <= 60
            adstock_validity[channel] = {
                'r_value': r_value,
                'decay_days_90pct': decay_days,
                'is_realistic': is_realistic
            }
    
    all_realistic = all(cv['is_realistic'] for cv in adstock_validity.values())
    tests['adstock_decay'] = {
        'status': 'pass' if all_realistic else 'fail',
        'channel_details': adstock_validity,
        'threshold': '1-60 days for 90% decay',
        'interpretation': 'Tests if adstock parameters create realistic carryover patterns'
    }
    
    # 9. Cross-Channel Correlation Matrix
    correlation_matrix = np.corrcoef([transformed_channels[ch] for ch in transformed_channels.columns], rowvar=False)
    max_correlation = np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
    
    tests['cross_channel_correlation'] = {
        'status': 'pass' if max_correlation < 0.7 else 'fail',
        'max_correlation': max_correlation,
        'correlation_matrix': correlation_matrix.tolist(),
        'threshold': '<0.7',
        'interpretation': 'Tests for excessive correlation between transformed channels'
    }
    
    # 10. Attribution Stability Test
    if len(data) >= 60:  # Need at least 2 months of data
        monthly_attributions = calculate_monthly_attributions(model_results, data)
        attribution_variance = np.std(monthly_attributions) / np.mean(monthly_attributions)
        
        tests['attribution_stability'] = {
            'status': 'pass' if attribution_variance < 0.2 else 'fail',
            'coefficient_of_variation': attribution_variance,
            'monthly_attributions': monthly_attributions,
            'threshold': '<20% month-over-month variance',
            'interpretation': 'Tests if attribution is stable across time periods'
        }
    else:
        tests['attribution_stability'] = {
            'status': 'skip',
            'reason': 'Insufficient data for monthly comparison',
            'interpretation': 'Requires at least 60 days of data'
        }
    
    # 11. Parameter Bounds Realism
    bounds_validity = {}
    channel_type_bounds = {
        'search_brand': {'beta': (0.4, 0.8), 'r': (0.0, 0.3)},
        'search_non_brand': {'beta': (0.5, 0.9), 'r': (0.1, 0.4)},
        'social': {'beta': (0.3, 0.7), 'r': (0.2, 0.6)},
        'tv_video': {'beta': (0.2, 0.6), 'r': (0.3, 0.8)},
        'display': {'beta': (0.3, 0.7), 'r': (0.1, 0.6)},
        'unknown': {'beta': (0.3, 0.7), 'r': (0.1, 0.5)}
    }
    
    for channel, params in parameters.items():
        channel_type = classify_channel_type(channel)
        expected_bounds = channel_type_bounds.get(channel_type, channel_type_bounds['unknown'])
        
        beta_in_bounds = expected_bounds['beta'][0] <= params.get('beta', 0.5) <= expected_bounds['beta'][1]
        r_in_bounds = expected_bounds['r'][0] <= params.get('r', 0.3) <= expected_bounds['r'][1]
        
        bounds_validity[channel] = {
            'beta_in_bounds': beta_in_bounds,
            'r_in_bounds': r_in_bounds,
            'both_in_bounds': beta_in_bounds and r_in_bounds
        }
    
    all_in_bounds = all(bv['both_in_bounds'] for bv in bounds_validity.values())
    tests['parameter_bounds_realism'] = {
        'status': 'pass' if all_in_bounds else 'fail',
        'channel_details': bounds_validity,
        'interpretation': 'Tests if estimated parameters align with channel-type research expectations'
    }
    
    return tests

def calculate_diagnostic_grade(passed_tests, total_tests):
    """
    Calculate overall diagnostic grade
    """
    pass_rate = passed_tests / total_tests
    
    if pass_rate >= 0.82:  # 9-11 tests passed
        return 'excellent'
    elif pass_rate >= 0.64:  # 7-8 tests passed
        return 'good'
    elif pass_rate >= 0.45:  # 5-6 tests passed
        return 'acceptable'
    else:  # <5 tests passed
        return 'poor'

Section 14: Real-Time Monitoring & Alerting System
Complete Monitoring Architecture
class MMModelMonitor:
    """
    Real-time monitoring system for MMM model performance
    """
    
    def __init__(self, alert_config):
        self.alert_thresholds = alert_config
        self.monitoring_active = True
        self.alert_handlers = []
        
    def setup_monitoring(self, model_id, baseline_performance):
        """
        Initialize monitoring for a specific model
        """
        self.model_id = model_id
        self.baseline_mape = baseline_performance['cv_mape']
        self.baseline_attribution = baseline_performance['media_attribution']
        self.last_check_time = datetime.utcnow()
        
        # Create monitoring schedule
        schedule.every(1).hours.do(self.check_performance_drift)
        schedule.every(1).days.do(self.check_attribution_stability)
        schedule.every(1).weeks.do(self.check_parameter_drift)
        
    def check_performance_drift(self):
        """
        Monitor model performance degradation
        """
        current_time = datetime.utcnow()
        
        # Get recent 30-day performance
        recent_performance = self.calculate_rolling_performance(days=30)
        
        if recent_performance is None:
            return  # Insufficient recent data
        
        # Calculate performance degradation
        mape_degradation = (recent_performance['mape'] - self.baseline_mape) / self.baseline_mape
        
        if mape_degradation > self.alert_thresholds['mape_degradation']:
            self.trigger_alert({
                'type': 'PERFORMANCE_DEGRADATION',
                'severity': 'HIGH',
                'current_mape': recent_performance['mape'],
                'baseline_mape': self.baseline_mape,
                'degradation_pct': mape_degradation * 100,
                'recommendation': 'Consider model retraining or data quality review',
                'timestamp': current_time
            })
    
    def check_attribution_stability(self):
        """
        Monitor attribution stability across time periods
        """
        # Get monthly attribution data
        monthly_attributions = self.get_monthly_attributions(months=3)
        
        if len(monthly_attributions) < 2:
            return  # Need at least 2 months
        
        # Calculate month-over-month variance
        attribution_changes = []
        for i in range(1, len(monthly_attributions)):
            change = abs(monthly_attributions[i] - monthly_attributions[i-1]) / monthly_attributions[i-1]
            attribution_changes.append(change)
        
        max_change = max(attribution_changes)
        
        if max_change > self.alert_thresholds['attribution_stability']:
            self.trigger_alert({
                'type': 'ATTRIBUTION_INSTABILITY',
                'severity': 'MEDIUM',
                'max_month_over_month_change': max_change * 100,
                'threshold': self.alert_thresholds['attribution_stability'] * 100,
                'monthly_attributions': monthly_attributions,
                'recommendation': 'Review recent spend changes or external factors',
                'timestamp': datetime.utcnow()
            })
    
    def check_parameter_drift(self):
        """
        Monitor parameter drift across model versions
        """
        # Get parameter history
        parameter_history = self.get_parameter_history(weeks=12)
        
        if len(parameter_history) < 2:
            return  # Need at least 2 parameter sets
        
        # Calculate parameter drift
        drift_detected = False
        drift_details = {}
        
        latest_params = parameter_history[-1]
        baseline_params = parameter_history[0]
        
        for channel in latest_params:
            if channel in baseline_params:
                beta_drift = abs(latest_params[channel]['beta'] - baseline_params[channel]['beta'])
                r_drift = abs(latest_params[channel]['r'] - baseline_params[channel]['r'])
                
                if beta_drift > self.alert_thresholds['parameter_drift'] or r_drift > self.alert_thresholds['parameter_drift']:
                    drift_detected = True
                    drift_details[channel] = {
                        'beta_drift': beta_drift,
                        'r_drift': r_drift,
                        'significant': True
                    }
        
        if drift_detected:
            self.trigger_alert({
                'type': 'PARAMETER_DRIFT',
                'severity': 'MEDIUM',
                'channels_affected': list(drift_details.keys()),
                'drift_details': drift_details,
                'recommendation': 'Consider parameter stability analysis or model retraining',
                'timestamp': datetime.utcnow()
            })
    
    def monitor_api_performance(self, endpoint, response_time):
        """
        Monitor API response time performance
        """
        target_time = API_PERFORMANCE_TARGETS.get(endpoint, {}).get('target_ms', 1000)
        max_time = API_PERFORMANCE_TARGETS.get(endpoint, {}).get('max_ms', 5000)
        
        if response_time > max_time:
            self.trigger_alert({
                'type': 'API_PERFORMANCE_DEGRADATION',
                'severity': 'HIGH',
                'endpoint': endpoint,
                'response_time_ms': response_time,
                'target_ms': target_time,
                'max_threshold_ms': max_time,
                'recommendation': 'Check system resources and database performance',
                'timestamp': datetime.utcnow()
            })
    
    def trigger_alert(self, alert_data):
        """
        Trigger alert through configured channels
        """
        alert = {
            'id': str(uuid.uuid4()),
            'model_id': self.model_id,
            'alert_data': alert_data,
            'created_at': datetime.utcnow(),
            'status': 'active'
        }
        
        # Store alert in database
        self.store_alert(alert)
        
        # Send notifications through configured handlers
        for handler in self.alert_handlers:
            try:
                handler.send_notification(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {str(e)}")

# Alert Configuration
ALERT_THRESHOLDS = {
    'mape_degradation': 0.10,  # 10% increase from baseline
    'attribution_stability': 0.20,  # 20% month-over-month variance
    'api_response_time': 5000,  # 5 seconds for optimization endpoints
    'parameter_drift': 0.15,  # 15% coefficient change
    'data_quality_score': 0.70  # Below 70% quality score
}

# Alert Handler Classes
class EmailAlertHandler:
    def __init__(self, smtp_config, recipients):
        self.smtp_config = smtp_config
        self.recipients = recipients
    
    def send_notification(self, alert):
        subject = f"MMM Alert: {alert['alert_data']['type']}"
        body = self.format_alert_email(alert)
        
        # Send email implementation
        send_email(
            recipients=self.recipients,
            subject=subject,
            body=body,
            smtp_config=self.smtp_config
        )

class SlackAlertHandler:
    def __init__(self, webhook_url, channel):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send_notification(self, alert):
        slack_message = self.format_slack_message(alert)
        
        # Send Slack notification
        requests.post(self.webhook_url, json={
            'channel': self.channel,
            'text': slack_message,
            'username': 'MMM Monitor'
        })

Section 15: Complete Database Schema & Transactions
Required imports for database operations
from contextlib import contextmanager 
from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker import logging 
from .exceptions import DataValidationError, ModelTrainingError, OptimizationError


Enhanced Database Schema with Constraints
-- Enhanced database schema with performance optimization
CREATE TABLE clients (
    client_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_name VARCHAR(255) NOT NULL,
    business_tier ENUM('enterprise', 'mid_market', 'small_business', 'prototype') NOT NULL,
    annual_spend_estimate DECIMAL(12,2),
    currency_code CHAR(3) DEFAULT 'USD',
    timezone VARCHAR(50) DEFAULT 'UTC',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_annual_spend CHECK (annual_spend_estimate >= 0),
    CONSTRAINT valid_currency CHECK (currency_code ~ '^[A-Z]{3})
);

CREATE TABLE historical_data (
    client_id UUID REFERENCES clients(client_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    profit DECIMAL(12,2) NOT NULL,
    is_holiday BOOLEAN DEFAULT FALSE,
    promo_flag BOOLEAN DEFAULT FALSE,
    site_outage BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (client_id, date),
    CONSTRAINT positive_profit CHECK (profit >= 0),
    CONSTRAINT reasonable_date CHECK (date >= '2020-01-01' AND date <= CURRENT_DATE + INTERVAL '1 year')
);

CREATE TABLE channel_spend (
    client_id UUID REFERENCES clients(client_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    channel_name VARCHAR(100) NOT NULL,
    spend DECIMAL(10,2) NOT NULL,
    channel_type ENUM('search_brand', 'search_non_brand', 'social', 'tv_video', 'display', 'unknown') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (client_id, date, channel_name),
    CONSTRAINT positive_spend CHECK (spend >= 0),
    CONSTRAINT reasonable_spend CHECK (spend <= 1000000), -- Max $1M daily spend per channel
    
    FOREIGN KEY (client_id, date) REFERENCES historical_data(client_id, date) ON DELETE CASCADE
);

CREATE TABLE model_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID REFERENCES clients(client_id) ON DELETE CASCADE,
    model_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    training_start_date DATE NOT NULL,
    training_end_date DATE NOT NULL,
    cv_mape DECIMAL(5,2),
    cv_mape_std DECIMAL(5,2),
    r_squared DECIMAL(4,3),
    shadow_price DECIMAL(8,4),
    model_status ENUM('training', 'completed', 'failed', 'cancelled') NOT NULL DEFAULT 'training',
    diagnostic_score INTEGER CHECK (diagnostic_score BETWEEN 0 AND 11),
    parameter_stability_score ENUM('stable', 'moderate', 'unstable'),
    total_folds INTEGER,
    completed_folds INTEGER DEFAULT 0,
    training_config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    
    CONSTRAINT valid_mape CHECK (cv_mape >= 0 AND cv_mape <= 100),
    CONSTRAINT valid_r_squared CHECK (r_squared BETWEEN 0 AND 1),
    CONSTRAINT valid_shadow_price CHECK (shadow_price BETWEEN 0 AND 50),
    CONSTRAINT valid_date_range CHECK (training_end_date >= training_start_date),
    CONSTRAINT valid_fold_counts CHECK (completed_folds <= total_folds)
);

CREATE TABLE model_parameters (
    run_id UUID REFERENCES model_runs(run_id) ON DELETE CASCADE,
    channel_name VARCHAR(100) NOT NULL,
    channel_type VARCHAR(50) NOT NULL,
    alpha DECIMAL(10,4) NOT NULL,
    beta DECIMAL(4,2) NOT NULL,
    r_value DECIMAL(4,2) NOT NULL,
    alpha_baseline DECIMAL(12,2), -- Only populated for baseline parameters
    alpha_trend DECIMAL(8,4),     -- Only populated for baseline parameters
    is_estimated BOOLEAN NOT NULL DEFAULT FALSE,
    confidence_interval_lower DECIMAL(10,4),
    confidence_interval_upper DECIMAL(10,4),
    bootstrap_cv DECIMAL(4,3), -- Coefficient of variation from bootstrap
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (run_id, channel_name),
    CONSTRAINT positive_alpha CHECK (alpha >= 0),
    CONSTRAINT valid_beta CHECK (beta BETWEEN 0.1 AND 1.0),
    CONSTRAINT valid_r CHECK (r_value BETWEEN 0.0 AND 0.99),
    CONSTRAINT valid_confidence_interval CHECK (
        (confidence_interval_lower IS NULL AND confidence_interval_upper IS NULL) OR
        (confidence_interval_lower IS NOT NULL AND confidence_interval_upper IS NOT NULL AND
         confidence_interval_lower <= confidence_interval_upper)
    )
);

CREATE TABLE optimization_scenarios (
    scenario_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID REFERENCES clients(client_id) ON DELETE CASCADE,
    run_id UUID REFERENCES model_runs(run_id) ON DELETE CASCADE,
    scenario_name VARCHAR(255) NOT NULL,
    total_budget DECIMAL(12,2) NOT NULL,
ramp_limit_pct DECIMAL(4,2) DEFAULT 0.20 CHECK (ramp_limit_pct BETWEEN 0.0 AND 1.0),
    planning_mode ENUM('steady_state') DEFAULT 'steady_state',
    optimization_status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    computation_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    
    CONSTRAINT positive_budget CHECK (total_budget > 0),
    CONSTRAINT reasonable_budget CHECK (total_budget <= 100000000) -- Max $100M budget
);

CREATE TABLE scenario_constraints (
    scenario_id UUID REFERENCES optimization_scenarios(scenario_id) ON DELETE CASCADE,
    channel_name VARCHAR(100) NOT NULL,
    constraint_type ENUM('floor', 'cap', 'lock') NOT NULL,
    constraint_value DECIMAL(10,2) NOT NULL,
    
    PRIMARY KEY (scenario_id, channel_name, constraint_type),
    CONSTRAINT positive_constraint_value CHECK (constraint_value >= 0)
);

CREATE TABLE optimization_results (
    scenario_id UUID REFERENCES optimization_scenarios(scenario_id) ON DELETE CASCADE,
    channel_name VARCHAR(100) NOT NULL,
    current_spend DECIMAL(10,2) NOT NULL,
    proposed_spend DECIMAL(10,2) NOT NULL,
    current_contribution DECIMAL(12,2) NOT NULL,
    proposed_contribution DECIMAL(12,2) NOT NULL,
    marginal_roi DECIMAL(6,3) NOT NULL,
    is_binding_constraint BOOLEAN DEFAULT FALSE,
    constraint_type VARCHAR(50),
    confidence_interval_lower DECIMAL(12,2),
    confidence_interval_upper DECIMAL(12,2),
    
    PRIMARY KEY (scenario_id, channel_name),
    CONSTRAINT positive_spend CHECK (current_spend >= 0 AND proposed_spend >= 0),
    CONSTRAINT reasonable_roi CHECK (marginal_roi BETWEEN -10 AND 50)
);

-- Monitoring and alerting tables
CREATE TABLE model_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID REFERENCES clients(client_id) ON DELETE CASCADE,
    run_id UUID REFERENCES model_runs(run_id) ON DELETE SET NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') NOT NULL,
    alert_data JSONB NOT NULL,
    status ENUM('active', 'acknowledged', 'resolved') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    
    INDEX idx_alerts_client_status (client_id, status, created_at),
    INDEX idx_alerts_type_severity (alert_type, severity)
);

CREATE TABLE performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID REFERENCES clients(client_id) ON DELETE CASCADE,
    run_id UUID REFERENCES model_runs(run_id) ON DELETE CASCADE,
    metric_date DATE NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- 'daily_mape', 'attribution_stability', etc.
    metric_value DECIMAL(10,4) NOT NULL,
    baseline_value DECIMAL(10,4),
    threshold_value DECIMAL(10,4),
    is_within_threshold BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (client_id, run_id, metric_date, metric_type),
    INDEX idx_performance_date_type (metric_date, metric_type)
);

-- Critical performance indexes
CREATE INDEX idx_channel_spend_date_client ON channel_spend(client_id, date, channel_name);
CREATE INDEX idx_historical_data_date_range ON historical_data(client_id, date) 
  WHERE date >= '2025-01-01';
CREATE INDEX idx_model_runs_status ON model_runs(client_id, model_status, created_at);
CREATE INDEX idx_optimization_scenarios_client ON optimization_scenarios(client_id, created_at);
CREATE INDEX idx_model_parameters_run_channel ON model_parameters(run_id, channel_name);
CREATE INDEX idx_optimization_results_scenario ON optimization_results(scenario_id, channel_name);

-- Partitioning for large tables (if needed)
-- CREATE TABLE historical_data_2025 PARTITION OF historical_data 
-- FOR VALUES FROM ('2025-10-01') TO ('2025-01-01');

Transaction Management and Error Handling
class DatabaseManager:
    """
    Database transaction manager with proper error handling
    """
    
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)
        self.session_factory = sessionmaker(bind=self.engine)
    
    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions with automatic rollback
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except (DataValidationError, ModelTrainingError, OptimizationError) as mmm_error: 
session.rollback() 
logging.error(f"MMM business logic error during database operation: {str(mmm_error)}") 
raise # Re-raise the original MMM exception
except Exception as e:
            session.rollback()
            logging.error(f"Database transaction failed: {str(e)}")
            raise
        finally:
            session.close()
    
    def upload_client_data(self, client_id, csv_data, validation_results):
        """
        Atomic upload of client data with validation
        """
        with self.transaction() as session:
            # Clear existing data for date range
            date_range = (csv_data['date'].min(), csv_data['date'].max())
            
            session.query(ChannelSpend).filter(
                ChannelSpend.client_id == client_id,
                ChannelSpend.date.between(*date_range)
            ).delete()
            
            session.query(HistoricalData).filter(
                HistoricalData.client_id == client_id,
                HistoricalData.date.between(*date_range)
            ).delete()
            
            # Insert new data
            historical_records = []
            channel_records = []
            
            for _, row in csv_data.iterrows():
                # Historical data record
                historical_records.append(HistoricalData(
                    client_id=client_id,
                    date=row['date'],
                    profit=row['profit'],
                    is_holiday=row.get('is_holiday', False),
                    promo_flag=row.get('promo_flag', False),
                    site_outage=row.get('site_outage', False)
                ))
                
                # Channel spend records
                for channel_col in csv_data.columns:
                    if channel_col not in ['date', 'profit', 'is_holiday', 'promo_flag', 'site_outage']:
                        channel_type = classify_channel_type(channel_col)
                        channel_records.append(ChannelSpend(
                            client_id=client_id,
                            date=row['date'],
                            channel_name=channel_col,
                            spend=row[channel_col],
                            channel_type=channel_type
                        ))
            
            # Bulk insert
            session.bulk_save_objects(historical_records)
            session.bulk_save_objects(channel_records)
            
            # Log upload success
            logging.info(f"Successfully uploaded {len(historical_records)} days of data for client {client_id}")
    
    def save_model_training_progress(self, run_id, fold_id, progress_data):
        """
        Save training progress with conflict resolution
        """
        with self.transaction() as session:
            # Update model run progress
            model_run = session.query(ModelRun).filter_by(run_id=run_id).first()
            if model_run:
                model_run.completed_folds = progress_data['completed_folds']
                model_run.updated_at = datetime.utcnow()
                
                if progress_data.get('current_best_mape'):
                    model_run.cv_mape = progress_data['current_best_mape']
    
    def save_model_results(self, run_id, model_results):
        """
        Save complete model results atomically
        """
        with self.transaction() as session:
            # Update model run status
            model_run = session.query(ModelRun).filter_by(run_id=run_id).first()
            model_run.model_status = 'completed'
            model_run.cv_mape = model_results['cv_mape']
            model_run.r_squared = model_results['r_squared']
            model_run.shadow_price = model_results['shadow_price']
            model_run.diagnostic_score = model_results['diagnostic_score']
            model_run.parameter_stability_score = model_results['stability_score']
            model_run.completed_at = datetime.utcnow()
            
            # Save parameters
            for channel, params in model_results['parameters'].items():
                model_param = ModelParameter(
                    run_id=run_id,
                    channel_name=channel,
                    channel_type=params['channel_type'],
                    alpha=params['alpha'],
                    beta=params['beta'],
                    r_value=params['r'],
                    is_estimated=params['is_estimated'],
                    confidence_interval_lower=params.get('ci_lower'),
                    confidence_interval_upper=params.get('ci_upper'),
                    bootstrap_cv=params.get('bootstrap_cv')
                )
                session.add(model_param)
            
            # Save baseline parameters
            baseline_param = ModelParameter(
                run_id=run_id,
                channel_name='baseline',
                channel_type='baseline',
                alpha=0,  # Not applicable for baseline
                beta=0,   # Not applicable for baseline
                r_value=0, # Not applicable for baseline
                alpha_baseline=model_results['baseline']['alpha_baseline'],
                alpha_trend=model_results['baseline']['alpha_trend'],
                is_estimated=True
            )
            session.add(baseline_param)

Section 16: Configuration Management & Environment Setup
Complete Configuration System
# config.py - Environment-specific configuration management
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    
    @property
    def connection_string(self):
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class TrainingConfig:
    """Configuration for model training parameters"""
    # Cross-validation settings
    training_window_days: int = 126
    test_window_days: int = 14
    step_days: int = 14
    min_folds: int = 4
    
    # Parameter search settings
    use_efficient_grids: bool = False
    enable_early_stopping: bool = False
    early_stopping_threshold: float = 0.005
    min_folds_before_stopping: int = 8
    
    # Bootstrap settings
    bootstrap_samples: int = 100
    bootstrap_confidence_level: float = 0.90
    enable_stratified_bootstrap: bool = False  # Beyond MVP
    
    # Performance settings
    max_training_time_hours: int = 6
    parallel_processing: bool = True
    max_workers: int = 4

@dataclass
class OptimizationConfig:
    """Configuration for budget optimization"""
    algorithm: str = "L-BFGS-B"
    max_iterations: int = 1000
    tolerance: float = 1e-6
    gradient_tolerance: float = 1e-6
    
    # Business constraints
    default_ramp_limit_pct: float = 0.20
    min_channel_spend: float = 100.0
    max_channel_spend: float = 10000000.0
    
    # Shadow price validation
    min_valid_shadow_price: float = 0.3
    max_valid_shadow_price: float = 8.0

@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring"""
    enable_monitoring: bool = True
    alert_thresholds: Dict[str, float] = None
    check_frequency_hours: int = 1
    
    # Alert handlers
    email_alerts: bool = True
    slack_alerts: bool = False
    webhook_alerts: bool = False
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'mape_degradation': 0.10,
                'attribution_stability': 0.20,
                'api_response_time': 5000,
                'parameter_drift': 0.15,
                'data_quality_score': 0.70
            }

@dataclass
class APIConfig:
    """Configuration for API performance and behavior"""
    # Performance targets (milliseconds)
    target_response_times: Dict[str, int] = None
    max_response_times: Dict[str, int] = None
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 20
    
    # Caching
    enable_response_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size_mb: int = 512
    
    def __post_init__(self):
        if self.target_response_times is None:
            self.target_response_times = {
                'model_status': 200,
                'optimization_run': 2000,
                'response_curves_cached': 500,
                'response_curves_computed': 3000,
                'dashboard_interactions': 200
            }
        
        if self.max_response_times is None:
            self.max_response_times = {
                'model_status': 500,
                'optimization_run': 5000,
                'response_curves_cached': 1000,
                'response_curves_computed': 8000,
                'dashboard_interactions': 500
            }

class ConfigManager:
    """
    Centralized configuration management with environment-specific overrides
    """
    
    def __init__(self, environment: str = None):
        self.environment = environment or os.getenv('MMM_ENVIRONMENT', 'development')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration based on environment"""
        base_config = self._get_base_config()
        env_config = self._get_environment_config()
        
        # Merge configurations (environment overrides base)
        merged_config = {**base_config, **env_config}
        
        return merged_config
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Base configuration applicable to all environments"""
        return {
            'database': DatabaseConfig(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', 5432)),
                database=os.getenv('DB_NAME', 'mmm_database'),
                username=os.getenv('DB_USERNAME', 'mmm_user'),
                password=os.getenv('DB_PASSWORD', 'password')
            ),
            'training': TrainingConfig(),
            'optimization': OptimizationConfig(),
            'monitoring': MonitoringConfig(),
            'api': APIConfig()
        }
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Environment-specific configuration overrides"""
        if self.environment == 'production':
            return {
                'training': TrainingConfig(
                    parallel_processing=True,
                    max_workers=8,
                    bootstrap_samples=100
                ),
                'monitoring': MonitoringConfig(
                    enable_monitoring=True,
                    email_alerts=True,
                    slack_alerts=True
                ),
                'api': APIConfig(
                    rate_limit_requests_per_minute=1000,
                    enable_response_caching=True,
                    max_cache_size_mb=2048
                )
            }
        
        elif self.environment == 'staging':
            return {
                'training': TrainingConfig(
                    use_efficient_grids=True,
                    bootstrap_samples=50
                ),
                'monitoring': MonitoringConfig(
                    enable_monitoring=True,
                    email_alerts=True
                )
            }
        
        elif self.environment == 'development':
            return {
                'training': TrainingConfig(
                    use_efficient_grids=True,
                    bootstrap_samples=25,
                    enable_early_stopping=True
                ),
                'monitoring': MonitoringConfig(
                    enable_monitoring=False
                ),
                'api': APIConfig(
                    enable_response_caching=False,
                    rate_limit_requests_per_minute=50
                )
            }
        
        else:
            return {}
    
    def get(self, key: str, default=None):
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = getattr(value, k, None)
            
            if value is None:
                return default
        
        return value

# Environment configuration files
# .env.development
"""
MMM_ENVIRONMENT=development
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mmm_dev
DB_USERNAME=dev_user
DB_PASSWORD=dev_password
LOG_LEVEL=DEBUG
ENABLE_PROFILING=true
"""

# .env.production
"""
MMM_ENVIRONMENT=production
DB_HOST=prod-db.example.com
DB_PORT=5432
DB_NAME=mmm_production
DB_USERNAME=prod_user
DB_PASSWORD=secure_production_password
LOG_LEVEL=INFO
ENABLE_PROFILING=false
SENTRY_DSN=https://your-sentry-dsn
SLACK_WEBHOOK_URL=https://hooks.slack.com/your-webhook
EMAIL_SMTP_HOST=smtp.example.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=alerts@example.com
EMAIL_PASSWORD=email_password
"""

# Usage example
config = ConfigManager()

# Access configuration values
db_config = config.get('database')
training_config = config.get('training')
enable_monitoring = config.get('monitoring.enable_monitoring', False)
api_timeout = config.get('api.target_response_times.optimization_run', 2000)

Section 17: Error Handling & Recovery Procedures
Comprehensive Error Handling Framework
    """Base exception for MMM-specific errors"""
    def __init__(self, message, error_code=None, recovery_suggestions=None):
        super().__init__(message)
        self.error_code = error_code
        self.recovery_suggestions = recovery_suggestions or []

class DataValidationError(MMModelingError):
    """Errors during data validation phase"""
    pass

class ModelTrainingError(MMModelingError):
    """Errors during model training phase"""
    pass

class OptimizationError(MMModelingError):
    """Errors during budget optimization phase"""
    pass

class ModelConvergenceError(ModelTrainingError):
    """Specific error when model fails to converge"""
    pass

def handle_training_errors(func):
    """Decorator for handling training errors with recovery"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        
        except ModelConvergenceError as e:
            logging.error(f"Model convergence failed: {str(e)}")
            
            # Attempt recovery with simplified parameters
            recovery_suggestions = [
                "Retry with efficient parameter grids",
                "Check data quality for outliers",
                "Reduce parameter search space",
                "Enable early stopping"
            ]
            
            if 'run_id' in kwargs:
                update_model_status(kwargs['run_id'], 'failed', str(e), recovery_suggestions)
            
            raise MMModelingError(
                f"Model training failed to converge: {str(e)}",
                error_code="CONVERGENCE_FAILURE",
                recovery_suggestions=recovery_suggestions
            )
        
        except DataValidationError as e:
            logging.error(f"Data validation failed: {str(e)}")
            
            recovery_suggestions = [
                "Check data upload format and encoding",
                "Verify date ranges and profit values",
                "Review spend data for negative values",
                "Ensure minimum data requirements are met"
            ]
            
            raise MMModelingError(
                f"Data validation failed: {str(e)}",
                error_code="DATA_VALIDATION_FAILURE",
                recovery_suggestions=recovery_suggestions
            )




