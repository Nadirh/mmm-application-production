"""
Apply migration to add client_id and organization_id fields to SQLite database.
Safe to run multiple times - checks if columns exist first.
"""
import sqlite3
import os

# Database path
db_path = os.path.join(os.path.dirname(__file__), "mmm_app.db")

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}. It will be created when the app runs.")
    exit(0)

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Function to check if column exists
def column_exists(table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    return any(col[1] == column_name for col in columns)

# Tables to update
tables = ['upload_sessions', 'training_runs', 'optimization_runs']

for table in tables:
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    if not cursor.fetchone():
        print(f"Table {table} doesn't exist yet. Skipping...")
        continue

    # Add client_id if it doesn't exist
    if not column_exists(table, 'client_id'):
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN client_id TEXT DEFAULT 'default'")
        print(f"Added client_id to {table}")
    else:
        print(f"client_id already exists in {table}")

    # Add organization_id if it doesn't exist
    if not column_exists(table, 'organization_id'):
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN organization_id TEXT DEFAULT 'default'")
        print(f"Added organization_id to {table}")
    else:
        print(f"organization_id already exists in {table}")

    # Create indexes (SQLite doesn't have IF NOT EXISTS for indexes, so we'll try-except)
    try:
        cursor.execute(f"CREATE INDEX idx_{table}_client_id ON {table}(client_id)")
        print(f"Created index on {table}.client_id")
    except sqlite3.OperationalError:
        print(f"Index on {table}.client_id already exists")

    try:
        cursor.execute(f"CREATE INDEX idx_{table}_organization_id ON {table}(organization_id)")
        print(f"Created index on {table}.organization_id")
    except sqlite3.OperationalError:
        print(f"Index on {table}.organization_id already exists")

# Commit changes
conn.commit()
print("\nMigration completed successfully!")

# Show current schema
print("\nCurrent schema:")
for table in tables:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    if columns:
        print(f"\n{table}:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")

conn.close()