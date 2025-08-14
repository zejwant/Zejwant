# storage/migrations/003_add_analytics_tables.py

"""
Migration 003: Add Analytics Tables
-----------------------------------
Purpose:
- Add tables to store application events, logs, and system metrics.
- Include partitioning suggestions for large datasets.
- Idempotent and safe for re-execution.

Version: 3.0
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from storage.sql_manager import SQLManager

# Logger setup
logger = logging.getLogger("migration_003")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def upgrade(sql_mgr: "SQLManager", db_name: str) -> None:
    """
    Apply analytics tables to the database.

    Args:
        sql_mgr (SQLManager): SQLManager instance.
        db_name (str): Target database name.
    """
    statements = [
        # Events table
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id BIGSERIAL PRIMARY KEY,
            user_id INT,
            event_type VARCHAR(100) NOT NULL,
            event_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            event_data JSONB,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """,
        # Logs table
        """
        CREATE TABLE IF NOT EXISTS logs (
            log_id BIGSERIAL PRIMARY KEY,
            log_level VARCHAR(20) NOT NULL,
            log_message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        # Metrics table
        """
        CREATE TABLE IF NOT EXISTS metrics (
            metric_id BIGSERIAL PRIMARY KEY,
            metric_name VARCHAR(100) NOT NULL,
            metric_value DOUBLE PRECISION NOT NULL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]

    # Partitioning recommendations (optional, for PostgreSQL)
    partition_notes = [
        "Consider partitioning 'events' by event_timestamp for large datasets.",
        "Consider partitioning 'logs' by created_at for log archiving.",
        "Consider partitioning 'metrics' by recorded_at for efficient analytics."
    ]

    for note in partition_notes:
        logger.info(f"Partitioning recommendation: {note}")

    for stmt in statements:
        try:
            sql_mgr.execute(db_name, stmt, commit=True)
            logger.info(f"Executed successfully:\n{stmt.strip()}")
        except Exception as e:
            logger.error(f"Failed to execute statement:\n{stmt.strip()}\nError: {e}")
            raise


def downgrade(sql_mgr: "SQLManager", db_name: str) -> None:
    """
    Rollback analytics tables from the database.

    Args:
        sql_mgr (SQLManager): SQLManager instance.
        db_name (str): Target database name.
    """
    statements = [
        "DROP TABLE IF EXISTS metrics",
        "DROP TABLE IF EXISTS logs",
        "DROP TABLE IF EXISTS events"
    ]

    for stmt in statements:
        try:
            sql_mgr.execute(db_name, stmt, commit=True)
            logger.info(f"Dropped successfully:\n{stmt.strip()}")
        except Exception as e:
            logger.error(f"Failed to drop table with statement:\n{stmt.strip()}\nError: {e}")
            raise
  
