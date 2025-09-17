# Parameter Grid Validation

## CRITICAL: Correct Parameter Values

**As of 2025-09-17**, the CORRECT parameter grid values are:

```python
"beta": [0.6, 0.7, 0.8, 0.9]  # Expanded to include stronger saturation (0.6)
"r": [0.1, 0.2, 0.3, 0.4]     # Expanded to include longer memory (0.4)
```

**These values apply to ALL channel types:**
- search_brand
- search_non_brand
- social
- tv_video
- display
- unknown

## Grid Expansion Impact
- Previous: 3×3 = 9 combinations per channel
- Current: 4×4 = 16 combinations per channel
- Training time increase: ~78% longer

## Validation

Before any training run, verify that `src/mmm/config/settings.py` contains:

```python
grids = {
    "search_brand": {
        "beta": [0.6, 0.7, 0.8, 0.9],
        "r": [0.1, 0.2, 0.3, 0.4]
    },
    # ... (same for all other channel types)
}
```

## Warning Signs of Parameter Reversion

❌ **INCORRECT VALUES (DO NOT USE):**
- Beta ranges like [0.4, 0.45, 0.5, ...] with many decimal increments
- R ranges like [0.0, 0.025, 0.05, ...] with small increments
- Different ranges per channel type

✅ **CORRECT VALUES (USE THESE):**
- Beta: exactly [0.7, 0.8, 0.9]
- R: exactly [0.1, 0.2, 0.3]
- Same ranges for all channel types

## History

- **Issue**: Parameters reverted from optimized values to old complex ranges
- **Root Cause**: Unknown automatic reversion
- **Fix**: Committed bab2032 - restored correct values
- **Prevention**: Added version comments and this validation document