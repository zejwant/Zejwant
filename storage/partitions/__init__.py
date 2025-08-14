# storage/partitions/__init__.py

"""
Partitions Package
------------------
Enterprise-grade helper for automating table partitioning across multiple tables.

Features:
- Import PartitionManager
- Provide high-level helper to partition multiple tables
- Type hints and docstrings for maintainability
"""

from typing import List, Dict, Any
from .partition_manager import PartitionManager
from ..sql_manager import SQLManager


def auto_partition_tables(
    sql_manager: SQLManager,
    db_name: str,
    tables_config: List[Dict[str, Any]]
) -> None:
    """
    Automate partitioning for multiple tables based on configuration.

    Args:
        sql_manager (SQLManager): SQLManager instance for database execution.
        db_name (str): Target database name.
        tables_config (List[Dict]): List of table configurations. Each config can include:
            - table_name (str): Name of the table.
            - partition_type (str): "date", "hash", or "range".
            - column (str): Column to partition by.
            - start_date (datetime, optional): For date partitioning.
            - end_date (datetime, optional): For date partitioning.
            - interval_days (int, optional): Interval for date partitioning.
            - num_partitions (int, optional): Number of hash partitions.
            - ranges (List[Dict], optional): For range partitioning, list of dicts with 'start' and 'end'.
    """
    partition_mgr = PartitionManager(sql_manager, db_name)

    for config in tables_config:
        table_name = config.get("table_name")
        partition_type = config.get("partition_type")
        column = config.get("column")

        if partition_type == "date":
            partition_mgr.create_date_partition(
                table_name=table_name,
                column=column,
                start_date=config.get("start_date"),
                end_date=config.get("end_date"),
                interval_days=config.get("interval_days", 7)
            )
        elif partition_type == "hash":
            partition_mgr.create_hash_partition(
                table_name=table_name,
                column=column,
                num_partitions=config.get("num_partitions", 4)
            )
        elif partition_type == "range":
            partition_mgr.create_range_partition(
                table_name=table_name,
                column=column,
                ranges=config.get("ranges", [])
            )
        else:
            raise ValueError(f"Unsupported partition type '{partition_type}' for table '{table_name}'")
          
