# storage/__init__.py

"""
Enterprise Storage Package
--------------------------
Unified interface for SQL, metadata, migrations, partitions, indexing,
backups, connectors, and monitoring.
"""

import logging
from typing import Any, Dict, Optional
import pandas as pd

from . import sql_manager
from . import metadata
from . import table_manager
from . import data_loader
from . import utils

# Optional modules
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

# Logger
logger = logging.getLogger("storage")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

# -------------------------
# High-Level Storage API
# -------------------------

def store_data(table_name: str, data: Any, overwrite: bool = False, batch_size: int = 1000) -> None:
    """Insert or update data into storage."""
    try:
        logger.info(f"Storing data into '{table_name}'")
        data_loader.load_data(table_name, data, overwrite=overwrite, batch_size=batch_size)
    except Exception as e:
        logger.error(f"store_data failed: {e}")
        raise

def read_data(table_name: str, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> pd.DataFrame:
    """Read data from storage table."""
    try:
        return sql_manager.query_table(table_name, filters=filters, limit=limit)
    except Exception as e:
        logger.error(f"read_data failed: {e}")
        return pd.DataFrame()

def manage_schema(table_name: str, schema_def: Dict[str, str], create_if_missing: bool = True) -> None:
    """Create or update table schema."""
    try:
        table_manager.create_or_update_table(table_name, schema_def, create_if_missing=create_if_missing)
    except Exception as e:
        logger.error(f"manage_schema failed: {e}")
        raise

def orchestrate_backup(target: Optional[str] = None, full_backup: bool = True, incremental: bool = False) -> None:
    """Trigger backup operations."""
    if backups is None:
        logger.warning("Backup module not installed. Skipping.")
        return
    try:
        if full_backup:
            backups.backup_manager.full_backup(target)
        elif incremental:
            backups.backup_manager.incremental_backup(target)
    except Exception as e:
        logger.error(f"Backup failed: {e}")

def orchestrate_partitioning() -> None:
    """Run partitioning automation across all tables."""
    if partitions is None:
        logger.warning("Partition module not installed. Skipping.")
        return
    try:
        partitions.partition_manager.auto_partition_all()
    except Exception as e:
        logger.error(f"Partitioning failed: {e}")

def optimize_indexes() -> None:
    """Run indexing optimization tasks."""
    if indexing is None:
        logger.warning("Indexing module not installed. Skipping.")
        return
    try:
        indexing.index_manager.optimize_all()
    except Exception as e:
        logger.error(f"Index optimization failed: {e}")

def load_external(connector_name: str, **kwargs) -> None:
    """Load data using external warehouse connectors (BigQuery, Snowflake, Redshift)."""
    if connectors is None:
        logger.warning("Connectors module not installed. Skipping.")
        return
    try:
        connector = connectors.get_connector(connector_name)
        connector.ingest(**kwargs)
    except Exception as e:
        logger.error(f"External load failed for '{connector_name}': {e}")
        
