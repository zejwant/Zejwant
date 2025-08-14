# storage/migrations/001_initial_schema.py

"""
Migration 001: Initial Schema
-----------------------------
Purpose:
- Create foundational tables for the data platform.
- Define primary and foreign keys, and indexes for performance.
- Idempotent: safe to re-run without breaking existing schema.

Version: 1.0
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from storage.sql_manager import SQLManager

# Logger setup
logger = logging.getLogger("migration_001")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def upgrade(sql_mgr: "SQLManager", db_name: str) -> None:
    """
    Apply initial schema to the database.

    Args:
        sql_mgr (SQLManager): SQLManager instance.
        db_name (str): Target database name.
    """
    statements = [
        # Users table
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            email VARCHAR(255) NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        # Roles table
        """
        CREATE TABLE IF NOT EXISTS roles (
            role_id SERIAL PRIMARY KEY,
            role_name VARCHAR(50) NOT NULL UNIQUE
        )
        """,
        # User-Roles mapping
        """
        CREATE TABLE IF NOT EXISTS user_roles (
            user_id INT NOT NULL,
            role_id INT NOT NULL,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, role_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (role_id) REFERENCES roles(role_id)
        )
        """,
        # Index for quick lookup on user_roles
        "CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON user_roles(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_user_roles_role_id ON user_roles(role_id)"
    ]

    for stmt in statements:
        try:
            sql_mgr.execute(db_name, stmt, commit=True)
            logger.info(f"Executed successfully:\n{stmt.strip()}")
        except Exception as e:
            logger.error(f"Failed to execute statement:\n{stmt.strip()}\nError: {e}")
            raise


def downgrade(sql_mgr: "SQLManager", db_name: str) -> None:
    """
    Rollback initial schema from the database.

    Args:
        sql_mgr (SQLManager): SQLManager instance.
        db_name (str): Target database name.
    """
    statements = [
        "DROP TABLE IF EXISTS user_roles",
        "DROP TABLE IF EXISTS roles",
        "DROP TABLE IF EXISTS users"
    ]

    for stmt in statements:
        try:
            sql_mgr.execute(db_name, stmt, commit=True)
            logger.info(f"Dropped successfully:\n{stmt.strip()}")
        except Exception as e:
            logger.error(f"Failed to drop table with statement:\n{stmt.strip()}\nError: {e}")
            raise
