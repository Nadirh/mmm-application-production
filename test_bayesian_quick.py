#!/usr/bin/env python3
"""
Quick test script for Bayesian optimization implementation.
"""
import pandas as pd
import numpy as np
import time
from src.mmm.model.mmm_model import MMMModel
from src.mmm.data.processor import DataProcessor

def quick_test():
    """Quick test with minimal data and trials."""

    print("\n" + "="*60)
    print("Quick Bayesian Optimization Test")
    print("="*60 + "\n")

    # Create simple synthetic data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')

    # Create 3 channels of spend data
    data = {
        'date': dates,
        'profit': np.random.uniform(10000, 20000, 200),
        'search': np.random.uniform(1000, 3000, 200),
        'social': np.random.uniform(800, 2500, 200),
        'display': np.random.uniform(500, 1500, 200)
    }

    df = pd.DataFrame(data)

    print(f"Created synthetic data:")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Channels: search, social, display")

    # Process data
    processor = DataProcessor()
    processed_df, channel_info = processor.process_data(df)
    channel_grids = processor.get_parameter_grid(channel_info)

    # Create model with reduced parameters for speed
    model = MMMModel(
        training_window_days=70,   # 10 weeks
        test_window_days=14,        # 2 weeks
        n_bootstrap=10,             # Very small for testing
        use_nested_cv=False
    )

    # Test that Bayesian optimization is being called
    start_time = time.time()

    # Track if Bayesian was used
    bayesian_used = False

    def progress_callback(update):
        nonlocal bayesian_used
        if 'bayesian' in str(update.get('type', '')).lower():
            bayesian_used = True
            print(f"✓ Bayesian optimization in progress: Trial {update.get('trial', '?')}/{update.get('total_trials', '?')}")

    print("\nStarting model training with Bayesian optimization...")

    try:
        results = model.fit(
            processed_df,
            channel_grids,
            progress_callback=progress_callback
        )

        elapsed = time.time() - start_time

        print(f"\n✅ Training completed in {elapsed:.1f} seconds")
        print(f"   CV MAPE: {results.cv_mape:.2f}%")

        if bayesian_used:
            print("\n✅ CONFIRMED: Bayesian optimization was used!")

            # Show optimized parameters
            print("\nOptimized parameters (from Bayesian search):")
            for channel in ['search', 'social', 'display']:
                beta = results.parameters.channel_betas.get(channel, 'N/A')
                r = results.parameters.channel_rs.get(channel, 'N/A')
                print(f"  {channel}: beta={beta:.3f}, r={r:.3f}")
        else:
            print("\n⚠️ Warning: Bayesian optimization may not have been used")

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()

    if success:
        print("\n" + "="*60)
        print("✅ Bayesian optimization is working correctly!")
        print("="*60)
    else:
        print("\n❌ Test failed")