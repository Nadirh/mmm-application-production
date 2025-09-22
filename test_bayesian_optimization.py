#!/usr/bin/env python3
"""
Test script for Bayesian optimization implementation.
"""
import pandas as pd
import numpy as np
import time
from src.mmm.model.mmm_model import MMMModel
from src.mmm.data.processor import DataProcessor
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer()
    ]
)

logger = structlog.get_logger()

def test_bayesian_optimization():
    """Test Bayesian optimization with 4-channel data."""

    print("\n" + "="*60)
    print("Testing Bayesian Optimization for MMM")
    print("="*60 + "\n")

    # Load test data (5 channels, we'll use first 4)
    df = pd.read_csv('test_data_5channels_52weeks.csv')

    # Keep only first 4 channels for testing
    channel_columns = ['search_brand', 'social_facebook', 'display_programmatic', 'video_youtube']
    df = df[['date', 'profit'] + channel_columns].copy()

    # Fill any missing values with 0
    df = df.fillna(0)

    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Channels: {channel_columns}")
    print(f"Total profit: ${df['profit'].sum():,.2f}")
    print(f"Total spend by channel:")
    for channel in channel_columns:
        print(f"  {channel}: ${df[channel].sum():,.2f}")

    # Process data
    processor = DataProcessor()
    processed_df, channel_info = processor.process_data(df)

    print(f"\nProcessed data shape: {processed_df.shape}")
    print(f"Channel info: {len(channel_info)} channels")

    # Get parameter grids (ignored for Bayesian but needed for interface)
    channel_grids = processor.get_parameter_grid(channel_info)

    # Create model
    model = MMMModel(
        training_window_days=126,  # 18 weeks
        test_window_days=14,        # 2 weeks
        n_bootstrap=100,            # Reduced for faster testing
        use_nested_cv=False         # Simple CV for testing
    )

    # Progress tracking
    progress_updates = []

    def progress_callback(update):
        progress_updates.append(update)
        if update.get('type') == 'bayesian_optimization':
            print(f"\rFold {update['fold']}: Trial {update['trial']}/{update['total_trials']} | "
                  f"Best MAPE: {update.get('best_mape', 'N/A'):.2f}%" if update.get('best_mape') else "N/A",
                  end='', flush=True)
        elif update.get('type') == 'bayesian_optimization_complete':
            print(f"\n✓ Fold {update['fold']} complete: Best MAPE = {update['best_mape']:.2f}%")

    # Train model
    print("\nStarting Bayesian optimization training...")
    print("This will use Optuna TPE sampler with full parameter ranges")
    print(f"Beta range: [0.1, 1.0], R range: [0.0, 0.99]")
    print(f"Trials per fold: {min(1000, 100 * len(channel_columns))}")

    start_time = time.time()

    try:
        results = model.fit(
            processed_df,
            channel_grids,
            progress_callback=progress_callback
        )

        elapsed_time = time.time() - start_time

        print(f"\n\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"\nTime taken: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"\nModel Results:")
        print(f"  R-squared: {results.r_squared:.4f}")
        print(f"  Training MAPE: {results.mape:.2f}%")
        print(f"  CV MAPE: {results.cv_mape:.2f}%")
        print(f"\nOptimized Parameters:")

        for channel in channel_columns:
            beta = results.parameters.channel_betas.get(channel, 'N/A')
            r = results.parameters.channel_rs.get(channel, 'N/A')
            alpha = results.parameters.channel_alphas.get(channel, 'N/A')
            print(f"\n  {channel}:")
            print(f"    Beta (saturation): {beta:.3f}")
            print(f"    R (adstock):       {r:.3f}")
            print(f"    Alpha (strength):  {alpha:.3f}")

        # Count progress updates
        bayesian_updates = [u for u in progress_updates if u.get('type') == 'bayesian_optimization']
        print(f"\n\nProgress updates received: {len(bayesian_updates)}")

        # Verify we're using Bayesian optimization
        if any('bayesian' in str(u.get('type', '')).lower() for u in progress_updates):
            print("✅ Confirmed: Bayesian optimization was used!")
        else:
            print("⚠️  Warning: Bayesian optimization may not have been used")

        return results

    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run test
    results = test_bayesian_optimization()

    if results:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")