# storage/backups/__init__.py

"""
Backups Package
---------------
Enterprise-grade module for managing database backups and restores.

Features:
- Import BackupManager and RestoreManager
- Expose high-level orchestration functions for backup and restore
- Logging, alerts, and type hints for maintainability
"""

import logging
from typing import Optional
from .backup_manager import BackupManager
from .restore_manager import RestoreManager
from ..sql_manager import SQLManager

# Logger setup
logger = logging.getLogger("backups")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def perform_backup(
    sql_manager: SQLManager,
    db_name: str,
    backup_type: str = "full",
    backup_location: Optional[str] = None
) -> None:
    """
    Orchestrate a database backup.

    Args:
        sql_manager (SQLManager): SQLManager instance.
        db_name (str): Target database name.
        backup_type (str): Type of backup ("full" or "incremental").
        backup_location (Optional[str]): Path to store backup files.
    """
    backup_mgr = BackupManager(sql_manager, db_name)
    try:
        if backup_type == "full":
            backup_mgr.full_backup(backup_location)
        elif backup_type == "incremental":
            backup_mgr.incremental_backup(backup_location)
        else:
            raise ValueError(f"Unsupported backup type: {backup_type}")
        logger.info(f"{backup_type.capitalize()} backup completed successfully for database '{db_name}'")
    except Exception as e:
        logger.error(f"Backup failed for database '{db_name}': {e}")


def perform_restore(
    sql_manager: SQLManager,
    db_name: str,
    backup_file: str
) -> None:
    """
    Orchestrate a database restore from backup.

    Args:
        sql_manager (SQLManager): SQLManager instance.
        db_name (str): Target database name.
        backup_file (str): Path to the backup file to restore.
    """
    restore_mgr = RestoreManager(sql_manager, db_name)
    try:
        restore_mgr.restore_from_backup(backup_file)
        logger.info(f"Restore completed successfully for database '{db_name}' from '{backup_file}'")
    except Exception as e:
        logger.error(f"Restore failed for database '{db_name}' from '{backup_file}': {e}")
      
