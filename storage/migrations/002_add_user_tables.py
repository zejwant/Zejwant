# storage/migrations/002_add_user_tables.py

"""
Migration 002: Add User Tables
------------------------------
Purpose:
- Add enhanced user management tables: permissions, role-permission mapping.
- Define relationships and constraints for enterprise access control.
- Idempotent and safe for re-execution.

Version: 2.0
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from storage.sql_manager import SQLManager

# Logger setup
logger = logging.getLogger("migration_002")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def upgrade(sql_mgr: "SQLManager", db_name: str) -> None:
    """
    Apply user-related tables to the database.

    Args:
        sql_mgr (SQLManager): SQLManager instance.
        db_name (str): Target database name.
    """
    statements = [
        # Permissions table
        """
        CREATE TABLE IF NOT EXISTS permissions (
            permission_id SERIAL PRIMARY KEY,
            permission_name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT
        )
        """,
        # Role-Permissions mapping
        """
        CREATE TABLE IF NOT EXISTS role_permissions (
            role_id INT NOT NULL,
            permission_id INT NOT NULL,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (role_id, permission_id),
            FOREIGN KEY (role_id) REFERENCES roles(role_id),
            FOREIGN KEY (permission_id) REFERENCES permissions(permission_id)
        )
        """,
        # Indexes for performance
        "CREATE INDEX IF NOT EXISTS idx_role_permissions_role_id ON role_permissions(role_id)",
        "CREATE INDEX IF NOT EXISTS idx_role_permissions_permission_id ON role_permissions(permission_id)"
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
    Rollback user-related tables from the database.

    Args:
        sql_mgr (SQLManager): SQLManager instance.
        db_name (str): Target database name.
    """
    statements = [
        "DROP TABLE IF EXISTS role_permissions",
        "DROP TABLE IF EXISTS permissions"
    ]

    for stmt in statements:
        try:
            sql_mgr.execute(db_name, stmt, commit=True)
            logger.info(f"Dropped successfully:\n{stmt.strip()}")
        except Exception as e:
            logger.error(f"Failed to drop table with statement:\n{stmt.strip()}\nError: {e}")
            raise
          
