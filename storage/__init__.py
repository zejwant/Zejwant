# storage/__init__.py

"""
Storage Package
---------------
Enterprise-grade storage layer for the data platform.

Responsibilities:
- SQL connection management, dynamic table creation, indexing, and partitioning
- Schema metadata tracking and migrations
- Backup, restore, and data archival orchestration
- High-level API for reading/writing data from/to SQL databases
- Logging and monitoring for all storage operations
"""

import logging
from typing import Any, Dict, Optional

# Core storage modules
from . import sql_manager
from . import metadata
from . import table_manager
from . import data_loader
from . import utils

# Optional modules for extended functionality
try:
    from . import partitions
except ImportError:
    partitions = None

try:
    from . import indexing
except ImportError:
    indexing = None

try:
    from . import backups
except ImportError:
    backups = None

try:
    from . import connectors
except ImportError:
    connectors = None

# Initialize logger
logger = logging.getLogger("storage")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# =========================
# High-level Storage APIs
# =========================

def store_data(
    table_name: str,
    data: Any,
    overwrite: bool = False,
    batch_size: int = 1000
) -> None:
    """
    Insert or update data into the storage backend.

    Args:
        table_name (str): Target table name.
        data (Any): DataFrame, list of dicts, or compatible format.
        overwrite (bool): Whether to overwrite existing table/data.
        batch_size (int): Number of rows per batch insert for performance.

    Returns:
        None
    """
    logger.info(f"Storing data into table: {table_name}")
    data_loader.load_data(table_name, data, overwrite=overwrite, batch_size=batch_size)


def read_data(
    table_name: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None
) -> Any:
    """
    Read data from a storage table with optional filtering.

    Args:
        table_name (str): Table to query.
        filters (Dict[str, Any], optional): Column-value filters.
        limit (int, optional): Maximum number of rows to fetch.

    Returns:
        DataFrame: Queried data as a pandas DataFrame.
    """
    logger.info(f"Reading data from table: {table_name}")
    return sql_manager.query_table(table_name, filters=filters, limit=limit)


def manage_schema(
    table_name: str,
    schema_def: Dict[str, str],
    create_if_missing: bool = True
) -> None:
    """
    Manage or update table schema dynamically.

    Args:
        table_name (str): Target table name.
        schema_def (Dict[str, str]): Column name to type mapping.
        create_if_missing (bool): Create table if it doesn't exist.

    Returns:
        None
    """
    logger.info(f"Managing schema for table: {table_name}")
    table_manager.create_or_update_table(table_name, schema_def, create_if_missing=create_if_missing)


def orchestrate_backup(
    target: Optional[str] = None,
    full_backup: bool = True,
    incremental: bool = False
) -> None:
    """
    Trigger backup or restore operations for storage.

    Args:
        target (str, optional): Specific table or database target.
        full_backup (bool): Whether to perform a full backup.
        incremental (bool): Whether to perform incremental backup.

    Returns:
        None
    """
    if backups is None:
        logger.warning("Backup module not installed. Skipping backup.")
        return

    if full_backup:
        logger.info(f"Starting full backup for target: {target or 'all'}")
        backups.full_backup(target)
    elif incremental:
        logger.info(f"Starting incremental backup for target: {target or 'all'}")
        backups.incremental_backup(target)
  
