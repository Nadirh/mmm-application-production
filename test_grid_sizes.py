#!/usr/bin/env python3
"""
Test script to compare parameter grid sizes between normal and test configurations.
"""
import os
import sys

# Add src to path
sys.path.insert(0, '/workspaces/mmm-application/src')

# Test with normal settings first
print("=" * 60)
print("NORMAL SETTINGS (5x5 grids)")
print("=" * 60)

from mmm.config.settings import settings

channels = ["search_brand", "social_facebook", "display_programmatic", "video_youtube", "affiliate_network"]

total_combinations_normal = 1
for channel in channels:
    config = settings.get_parameter_grid_config(channel)
    beta_count = len(config['beta'])
    r_count = len(config['r'])
    channel_combinations = beta_count * r_count
    total_combinations_normal *= channel_combinations
    print(f"{channel}: {beta_count} betas × {r_count} r values = {channel_combinations} combinations")

print(f"\nTotal combinations for {len(channels)} channels: {total_combinations_normal:,}")
print(f"This means {total_combinations_normal:,} models need to be trained!")

# Test with reduced settings
print("\n" + "=" * 60)
print("TEST SETTINGS (2x2 grids)")
print("=" * 60)

# Set environment variable and reimport
os.environ["USE_SMALL_GRID"] = "true"
from mmm.config.settings_test import get_settings
test_settings = get_settings()

total_combinations_test = 1
for channel in channels:
    config = test_settings.get_parameter_grid_config(channel)
    beta_count = len(config['beta'])
    r_count = len(config['r'])
    channel_combinations = beta_count * r_count
    total_combinations_test *= channel_combinations
    print(f"{channel}: {beta_count} betas × {r_count} r values = {channel_combinations} combinations")

print(f"\nTotal combinations for {len(channels)} channels: {total_combinations_test:,}")
print(f"This is {total_combinations_normal / total_combinations_test:.0f}x fewer models to train!")

# Test with 3 channels
print("\n" + "=" * 60)
print("COMPARISON: 3 channels vs 5 channels")
print("=" * 60)

channels_3 = channels[:3]
total_3_normal = 1
total_3_test = 1

for channel in channels_3:
    config_normal = settings.get_parameter_grid_config(channel)
    config_test = test_settings.get_parameter_grid_config(channel)

    normal_combo = len(config_normal['beta']) * len(config_normal['r'])
    test_combo = len(config_test['beta']) * len(config_test['r'])

    total_3_normal *= normal_combo
    total_3_test *= test_combo

print(f"3 channels with normal grid (5x5): {total_3_normal:,} combinations")
print(f"3 channels with test grid (2x2): {total_3_test:,} combinations")
print(f"5 channels with normal grid (5x5): {total_combinations_normal:,} combinations")
print(f"5 channels with test grid (2x2): {total_combinations_test:,} combinations")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print("For testing with 5 channels, use USE_SMALL_GRID=true to reduce")
print(f"from {total_combinations_normal:,} to {total_combinations_test:,} parameter combinations.")
print("This should make training complete in reasonable time.")