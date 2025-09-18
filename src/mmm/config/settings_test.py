"""
Test configuration with reduced parameter grids to avoid combinatorial explosion.
"""
import os
from mmm.config.settings import Settings, Environment


class TestSettings(Settings):
    """Settings with reduced parameter grids for testing."""

    def get_parameter_grid_config(self, channel_name: str, custom_r_values: dict = None) -> dict:
        """Gets reduced parameter grid for testing (2x2 instead of 5x5)."""
        # Classify the channel type
        channel_type = self.classify_channel_type(channel_name)

        # Reduced 2x2 grids for testing
        beta_grids = {
            "search_brand": [0.5, 0.7],
            "search_non_brand": [0.6, 0.8],
            "social": [0.6, 0.8],
            "display": [0.6, 0.8],
            "tv_video_youtube": [0.6, 0.8],
            "other": [0.6, 0.8]
        }

        # Default r values (just 2 values)
        r_grids = {
            "search_brand": [0.1, 0.2],
            "search_non_brand": [0.15, 0.25],
            "social": [0.25, 0.35],
            "display": [0.3, 0.4],
            "tv_video_youtube": [0.4, 0.6],
            "other": [0.25, 0.35]
        }

        # Use custom r values if provided
        if custom_r_values and channel_name in custom_r_values:
            center_r = custom_r_values[channel_name]
            # Generate just 2 values around the center
            if center_r == 0:
                r_values = [0]
            else:
                r_values = [round(center_r * 0.75, 3), round(min(center_r * 1.25, 0.99), 3)]
        else:
            r_values = r_grids[channel_type]

        return {
            "beta": beta_grids[channel_type],
            "r": r_values
        }


# Create test settings instance
test_settings = TestSettings(env=Environment.TESTING)

# Allow environment variable to control grid size
def get_settings():
    """Return test or normal settings based on environment variable."""
    if os.getenv("USE_SMALL_GRID", "false").lower() == "true":
        return test_settings
    else:
        from mmm.config.settings import settings
        return settings