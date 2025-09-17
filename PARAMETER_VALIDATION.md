# Parameter Grid Validation

## CRITICAL: Channel-Specific Parameter Values

**As of 2025-09-17**, the parameter grids are CHANNEL-SPECIFIC:

## Channel Classification Rules (case-insensitive)
1. **search_brand**: Contains 'search' AND 'brand'
2. **search_non_brand**: Contains 'search' AND NOT 'brand'
3. **social**: Contains 'social'
4. **display**: Contains 'display'
5. **tv_video_youtube**: Contains 'tv' OR 'video' OR 'youtube' OR 'yt'
6. **other**: Everything else

## Parameter Grids by Channel Type (5×5 = 25 combinations each)

### Search Brand
- **Beta**: [0.4, 0.5, 0.6, 0.7, 0.8]
- **r**: [0.05, 0.1, 0.15, 0.2, 0.25]
- **Default Half-Life**: 0.3 days

### Search Non-Brand
- **Beta**: [0.5, 0.6, 0.7, 0.8, 0.9]
- **r**: [0.1, 0.15, 0.2, 0.25, 0.3]
- **Default Half-Life**: 0.4 days

### Social
- **Beta**: [0.5, 0.6, 0.7, 0.8, 0.9]
- **r**: [0.1, 0.2, 0.3, 0.4, 0.5]
- **Default Half-Life**: 1.0 days

### Display
- **Beta**: [0.5, 0.6, 0.7, 0.8, 0.9]
- **r**: [0.15, 0.25, 0.35, 0.45, 0.55]
- **Default Half-Life**: 1.5 days

### TV-Video-YouTube
- **Beta**: [0.5, 0.6, 0.7, 0.8, 0.9]
- **r**: [0.3, 0.4, 0.5, 0.6, 0.7]
- **Default Half-Life**: 3.0 days

### Other
- **Beta**: [0.5, 0.6, 0.7, 0.8, 0.9]
- **r**: [0.1, 0.2, 0.3, 0.4, 0.5]
- **Default Half-Life**: 1.0 days

## Half-Life Customization

Users can customize the half-life for each channel, which automatically generates 5 r values using:
```
r = 0.5^(1/half_life_days)
```

The 5 values are generated from 50% to 150% of the specified half-life.

## Grid Expansion Impact
- Previous: 4×4 = 16 combinations per channel
- Current: 5×5 = 25 combinations per channel
- Training time increase: ~56% longer

## Validation

Before any training run, verify that `src/mmm/config/settings.py` contains the `classify_channel_type()` method and channel-specific grids.

## Warning Signs of Parameter Reversion

❌ **INCORRECT VALUES (DO NOT USE):**
- Uniform grids for all channels
- Only 3 or 4 values per parameter
- Missing channel classification logic