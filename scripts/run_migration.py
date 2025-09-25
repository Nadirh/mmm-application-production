#!/usr/bin/env python
"""
Script to run database migrations on production.
This script will be executed inside the Docker container.
"""
import os
import sys
import subprocess
from pathlib import Path

# Add src to Python path
sys.path.insert(0, '/app/src')

from mmm.config.settings import settings

def run_migration():
    """Run Alembic migration on the database."""
    print(f"Running migration on {settings.env.value} environment...")
    print(f"Database URL: {settings.database.url}")

    # Set the DATABASE_URL environment variable for Alembic
    os.environ['DATABASE_URL'] = settings.database.url

    # Run alembic upgrade
    try:
        result = subprocess.run(
            ['alembic', 'upgrade', 'head'],
            capture_output=True,
            text=True,
            cwd='/app'
        )

        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            print(f"Migration failed with return code {result.returncode}")
            sys.exit(1)

        print("Migration completed successfully!")

    except Exception as e:
        print(f"Error running migration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_migration()