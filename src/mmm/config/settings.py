"""
Configuration management for MMM application.
Handles application settings, environment variables, and business rules.
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration for SQLite and Redis."""
    url: str = "sqlite:///mmm_app.db"
    redis_url: str = "redis://localhost:6379/0"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    # Cache settings
    cache_ttl: int = 3600  # 1 hour default TTL
    response_curve_cache_ttl: int = 7200  # 2 hours for response curves
    session_ttl: int = 86400  # 24 hours for sessions


@dataclass
class ModelConfig:
    """Model training and validation configuration."""
    training_window_days: int = 126
    test_window_days: int = 14
    n_bootstrap: int = 1000
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    
    # Cross-validation settings
    min_training_days: int = 182
    max_cv_folds: int = 20
    
    # Parameter bounds
    min_beta: float = 0.1
    max_beta: float = 1.0
    min_r: float = 0.0
    max_r: float = 0.99
    min_alpha: float = 0.0


@dataclass
class BusinessTierConfig:
    """Business tier classification thresholds."""
    enterprise_days: int = 365
    enterprise_spend: float = 2000000
    enterprise_channel_min: float = 25000
    
    mid_market_days: int = 280
    mid_market_spend: float = 500000
    mid_market_channel_min: float = 15000
    
    small_business_days: int = 182
    small_business_spend: float = 200000
    small_business_channel_min: float = 8000
    
    prototype_days: int = 182
    prototype_spend: float = 50000


@dataclass
class ValidationConfig:
    """Data validation configuration."""
    max_file_size_mb: int = 100
    max_spend_jump_pct: float = 300.0
    min_data_quality_score: float = 50.0
    
    # MAPE targets by business tier
    enterprise_mape_target: float = 20.0
    mid_market_mape_target: float = 25.0
    small_business_mape_target: float = 35.0
    prototype_mape_target: float = 50.0
    
    # Model validation thresholds
    min_r_squared: float = 0.25
    min_media_attribution: float = 20.0
    max_media_attribution: float = 80.0
    min_shadow_price: float = 0.3
    max_shadow_price: float = 8.0


@dataclass
class OptimizationConfig:
    """Budget optimization configuration."""
    default_optimization_days: int = 365
    min_spend_resolution: float = 1000.0
    max_optimization_time: int = 300  # seconds
    
    # Default ramp constraints
    default_max_increase_pct: float = 20.0
    default_max_decrease_pct: float = 20.0
    
    # Response curve settings
    curve_resolution: int = 100
    max_spend_multiplier: float = 3.0


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    
    # File upload settings
    upload_dir: str = "static/uploads"
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: tuple = (".csv",)
    
    # API rate limiting
    rate_limit_calls: int = 100
    rate_limit_period: int = 60  # seconds
    
    # CORS settings
    cors_origins: list = field(default_factory=lambda: ["*"])
    cors_methods: list = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "logs"
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    # Structured logging
    use_json: bool = True
    include_trace: bool = False


class Settings:
    """Main application settings class."""
    
    def __init__(self, env: Optional[Environment] = None):
        self.env = env or Environment(os.getenv("MMM_ENV", "development"))
        self._load_environment_variables()
        self._initialize_configs()
    
    def _load_environment_variables(self):
        """Loads configuration from environment variables."""
        # Load .env file if exists
        env_file = Path(".env")
        if env_file.exists():
            self._load_env_file(env_file)
    
    def _load_env_file(self, env_file: Path):
        """Loads environment variables from .env file."""
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")
    
    def _initialize_configs(self):
        """Initializes configuration objects."""
        self.database = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite:///mmm_app.db"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            echo=self.env == Environment.DEVELOPMENT
        )
        
        self.model = ModelConfig(
            training_window_days=int(os.getenv("TRAINING_WINDOW_DAYS", "126")),
            test_window_days=int(os.getenv("TEST_WINDOW_DAYS", "14")),
            n_bootstrap=int(os.getenv("N_BOOTSTRAP", "1000"))
        )
        
        self.business_tiers = BusinessTierConfig()
        
        self.validation = ValidationConfig(
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100"))
        )
        
        self.optimization = OptimizationConfig(
            default_optimization_days=int(os.getenv("OPTIMIZATION_DAYS", "365"))
        )
        
        self.api = APIConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=self.env == Environment.DEVELOPMENT,
            reload=self.env == Environment.DEVELOPMENT,
            upload_dir=os.getenv("UPLOAD_DIR", "static/uploads")
        )
        
        self.logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            log_dir=os.getenv("LOG_DIR", "logs"),
            use_json=os.getenv("USE_JSON_LOGGING", "true").lower() == "true"
        )
    
    def get_business_tier_config(self, tier: str) -> Dict[str, Any]:
        """Gets configuration for a specific business tier."""
        tier_configs = {
            "enterprise": {
                "days": self.business_tiers.enterprise_days,
                "spend": self.business_tiers.enterprise_spend,
                "channel_min": self.business_tiers.enterprise_channel_min,
                "mape_target": self.validation.enterprise_mape_target
            },
            "mid_market": {
                "days": self.business_tiers.mid_market_days,
                "spend": self.business_tiers.mid_market_spend,
                "channel_min": self.business_tiers.mid_market_channel_min,
                "mape_target": self.validation.mid_market_mape_target
            },
            "small_business": {
                "days": self.business_tiers.small_business_days,
                "spend": self.business_tiers.small_business_spend,
                "channel_min": self.business_tiers.small_business_channel_min,
                "mape_target": self.validation.small_business_mape_target
            },
            "prototype": {
                "days": self.business_tiers.prototype_days,
                "spend": self.business_tiers.prototype_spend,
                "channel_min": 0,
                "mape_target": self.validation.prototype_mape_target
            }
        }
        
        return tier_configs.get(tier, tier_configs["prototype"])
    
    def classify_channel_type(self, channel_name: str) -> str:
        """Classify channel based on name patterns.

        Classification rules (case-insensitive):
        1. search_brand: Contains 'search' AND 'brand'
        2. search_non_brand: Contains 'search' AND NOT 'brand'
        3. social: Contains 'social'
        4. display: Contains 'display'
        5. tv_video_youtube: Contains 'tv' OR 'video' OR 'youtube' OR 'yt'
        6. other: Everything else
        """
        name_lower = channel_name.lower()

        # Check for search types first
        if 'search' in name_lower:
            if 'brand' in name_lower:
                return 'search_brand'
            else:
                return 'search_non_brand'
        elif 'social' in name_lower:
            return 'social'
        elif 'display' in name_lower:
            return 'display'
        elif any(term in name_lower for term in ['tv', 'video', 'youtube', 'yt']):
            return 'tv_video_youtube'
        else:
            return 'other'

    def get_parameter_grid_config(self, channel_name: str, custom_r_values: Dict[str, float] = None) -> Dict[str, list]:
        """Gets parameter grid configuration for a channel.

        Args:
            channel_name: Name of the channel
            custom_r_values: Optional dict of channel_name -> r_value (center value for grid)
        """
        # Classify the channel type
        channel_type = self.classify_channel_type(channel_name)

        # GRID_VERSION: v4.2 - Channel-specific grids with up to 5x5 combinations
        # Beta grids by channel type
        beta_grids = {
            "search_brand": [0.4, 0.5, 0.6, 0.7, 0.8],
            "search_non_brand": [0.5, 0.6, 0.7, 0.8, 0.9],
            "social": [0.5, 0.6, 0.7, 0.8, 0.9],
            "display": [0.5, 0.6, 0.7, 0.8, 0.9],
            "tv_video_youtube": [0.5, 0.6, 0.7, 0.8, 0.9],
            "other": [0.5, 0.6, 0.7, 0.8, 0.9]
        }

        # Default r center values by channel type
        default_r_centers = {
            "search_brand": 0.15,      # Fast decay
            "search_non_brand": 0.2,    # Fast-medium decay
            "social": 0.3,              # Medium decay
            "display": 0.35,            # Medium decay
            "tv_video_youtube": 0.5,    # Slow decay
            "other": 0.3                # Medium decay
        }

        # Get center r value (use custom if provided, otherwise use default)
        if custom_r_values and channel_name in custom_r_values:
            center_r = custom_r_values[channel_name]
        else:
            center_r = default_r_centers[channel_type]

        # Generate r grid from center value
        r_values = self.generate_r_grid_from_center(center_r)

        return {
            "beta": beta_grids[channel_type],
            "r": r_values
        }

    def generate_r_grid_from_center(self, center_r: float) -> list:
        """Generate up to 5 r values from 50% to 150% of center value.

        Special cases:
        - If center_r = 0, return [0]
        - Values are capped at 0.99 maximum
        - May return fewer than 5 values if hitting upper bound
        """
        import numpy as np

        center_r = float(center_r)

        # Special case: if center is 0, return only 0
        if center_r == 0:
            return [0]

        # Generate 5 values from 50% to 150% of center
        percentages = [0.5, 0.75, 1.0, 1.25, 1.5]
        r_values = []

        for pct in percentages:
            r_val = center_r * pct
            # Cap at 0.99 and stop adding if we've reached the cap
            if r_val >= 0.99:
                if 0.99 not in r_values:  # Only add 0.99 once
                    r_values.append(0.99)
                break
            else:
                r_values.append(round(r_val, 3))

        # Ensure we have at least some variation for optimization
        # If only one value, add slight variations if possible
        if len(r_values) == 1 and r_values[0] < 0.99:
            r_values.append(min(0.99, round(r_values[0] * 1.1, 3)))

        return r_values
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.env == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env == Environment.PRODUCTION
    
    def setup_directories(self):
        """Creates necessary directories."""
        directories = [
            self.api.upload_dir,
            self.logging.log_dir,
            "static",
            "static/exports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()