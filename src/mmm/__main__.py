"""
Main entry point for the MMM application.
"""
import sys
import argparse
import uvicorn

from mmm.config.settings import settings
from mmm.utils.logging import setup_logging


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(description="Media Mix Modeling Application")
    
    parser.add_argument(
        "--host", 
        default=settings.api.host, 
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=settings.api.port, 
        help="Port to bind the server to"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true",
        default=settings.api.reload,
        help="Enable auto-reload on code changes"
    )
    
    parser.add_argument(
        "--log-level", 
        default=settings.logging.level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level"
    )
    
    parser.add_argument(
        "--env",
        default=settings.env.value,
        choices=["development", "staging", "production", "testing"],
        help="Environment to run in"
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create directories
    settings.setup_directories()
    
    print(f"Starting MMM API server on {args.host}:{args.port}")
    print(f"Environment: {args.env}")
    print(f"Log level: {args.log_level}")
    
    try:
        uvicorn.run(
            "mmm.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower()
        )
    except KeyboardInterrupt:
        print("\nShutting down MMM API server...")
        sys.exit(0)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()