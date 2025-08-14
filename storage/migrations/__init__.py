# storage/migrations/__init__.py

"""
Migrations Package
------------------
Manage database schema versioning with dynamic migration scripts.

Features:
- Dynamically discover migration scripts
- Run migrations up (apply) or down (rollback)
- Maintain execution order
- Logging for all migration operations
- Type-hinted and enterprise-ready
"""

import importlib
import logging
import os
from pathlib import Path
from typing import List, Optional

from . import sql_manager

# Logger setup
logger = logging.getLogger("migrations")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


MIGRATIONS_DIR = Path(__file__).parent


def discover_migrations() -> List[str]:
    """
    Discover all migration scripts in the migrations directory.

    Returns:
        List[str]: Sorted list of migration module names (without .py).
    """
    migration_files = [
        f.stem for f in MIGRATIONS_DIR.glob("*.py") if f.name[0:3].isdigit()
    ]
    migration_files.sort()
    return migration_files


def run_migrations_up(db_name: str, sql_mgr: sql_manager.SQLManager, target: Optional[str] = None) -> None:
    """
    Apply migrations in order.

    Args:
        db_name (str): Target database.
        sql_mgr (SQLManager): SQLManager instance for query execution.
        target (str, optional): Apply migrations up to this migration (inclusive).
    """
    migrations = discover_migrations()
    for migration in migrations:
        if target and migration > target:
            break
        try:
            module = importlib.import_module(f".{migration}", package="storage.migrations")
            if hasattr(module, "upgrade"):
                logger.info(f"Applying migration '{migration}'")
                module.upgrade(sql_mgr, db_name)
                logger.info(f"Migration '{migration}' applied successfully")
            else:
                logger.warning(f"No upgrade() found in '{migration}'")
        except Exception as e:
            logger.error(f"Failed to apply migration '{migration}': {e}")
            raise


def run_migrations_down(db_name: str, sql_mgr: sql_manager.SQLManager, target: Optional[str] = None) -> None:
    """
    Rollback migrations in reverse order.

    Args:
        db_name (str): Target database.
        sql_mgr (SQLManager): SQLManager instance for query execution.
        target (str, optional): Rollback migrations down to this migration (inclusive).
    """
    migrations = discover_migrations()
    migrations.reverse()
    for migration in migrations:
        if target and migration < target:
            break
        try:
            module = importlib.import_module(f".{migration}", package="storage.migrations")
            if hasattr(module, "downgrade"):
                logger.info(f"Rolling back migration '{migration}'")
                module.downgrade(sql_mgr, db_name)
                logger.info(f"Migration '{migration}' rolled back successfully")
            else:
                logger.warning(f"No downgrade() found in '{migration}'")
        except Exception as e:
            logger.error(f"Failed to rollback migration '{migration}': {e}")
            raise
