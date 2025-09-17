# Grid Search Parameter Locations

## Current Values (v2.0)
- **Beta**: [0.7, 0.8, 0.9]
- **R**: [0.1, 0.2, 0.3]

## ✅ FIXED: Single Source of Truth Established

### **src/mmm/config/settings.py** (lines 252-273)
- Method: `get_parameter_grid_config()`
- **THIS IS NOW THE SINGLE SOURCE OF TRUTH**
- Contains all 6 channel types with parameter values
- Has warning comments about not reverting

## Data Flow (After Fix)
```
model.py calls processor.get_parameter_grid()
         ↓
processor.py calls settings.get_parameter_grid_config()
         ↓
settings.py returns the parameter values
```

## Files That Need Updating When Changing Parameters

### Must Update (Code - NOW ONLY ONE PLACE):
1. **src/mmm/config/settings.py** lines 252-273 - **SINGLE SOURCE OF TRUTH**

### Should Update (Documentation):
2. **README.md** line 148-149 - User documentation
3. **PARAMETER_VALIDATION.md** - Validation documentation

### May Need Update (Tests):
4. **tests/test_mathematical_model.py** - Has assertions expecting [0.1, 0.2, 0.3] for R values

## How to Change Parameters
To change grid search parameters in the future:
1. Edit **only** `/src/mmm/config/settings.py` in the `get_parameter_grid_config()` method
2. Update documentation files (README.md, PARAMETER_VALIDATION.md)
3. Update tests if the new values break existing assertions
4. No other code changes needed!

## Implementation Details
The refactored `processor.py` now calls:
```python
channel_grids[channel_name] = settings.get_parameter_grid_config(
    info.type.value
)
```

This ensures all parameter values come from a single source in `settings.py`.