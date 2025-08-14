# storage/partitions/partition_manager.py

"""
Partition Manager
-----------------
Enterprise-grade module for automating table partitioning.

Features:
- Create and maintain table partitions (date, range, hash)
- Monitor partition sizes and usage
- Alert if partitions grow too large or are unbalanced
- Logging, error handling, and type hints
- Docstrings explaining methods for maintainability
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..sql_manager import SQLManager

# Logger setup
logger = logging.getLogger("partition_manager")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class PartitionManager:
    def __init__(self, sql_manager: SQLManager, db_name: str):
        """
        Initialize PartitionManager.

        Args:
            sql_manager (SQLManager): SQLManager instance for executing queries.
            db_name (str): Database name where partitions exist.
        """
        self.sql_manager = sql_manager
        self.db_name = db_name

    # -------------------------
    # Partition Creation
    # -------------------------
    def create_date_partition(
        self,
        table_name: str,
        column: str,
        start_date: datetime,
        end_date: datetime,
        interval_days: int = 7
    ) -> None:
        """
        Create date-based partitions for a table.

        Args:
            table_name (str): Parent table name.
            column (str): Column to partition by (usually DATE or TIMESTAMP).
            start_date (datetime): Start date for partitioning.
            end_date (datetime): End date for partitioning.
            interval_days (int): Number of days per partition.
        """
        current_start = start_date
        while current_start < end_date:
            current_end = current_start + timedelta(days=interval_days)
            partition_name = f"{table_name}_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}"
            create_stmt = (
                f"CREATE TABLE IF NOT EXISTS {partition_name} "
                f"PARTITION OF {table_name} "
                f"FOR VALUES FROM ('{current_start.date()}') TO ('{current_end.date()}')"
            )
            try:
                self.sql_manager.execute(self.db_name, create_stmt, commit=True)
                logger.info(f"Created partition: {partition_name}")
            except Exception as e:
                logger.error(f"Failed to create partition {partition_name}: {e}")
            current_start = current_end

    def create_hash_partition(
        self,
        table_name: str,
        column: str,
        num_partitions: int
    ) -> None:
        """
        Create hash-based partitions for a table.

        Args:
            table_name (str): Parent table name.
            column (str): Column to hash.
            num_partitions (int): Number of hash partitions.
        """
        for i in range(num_partitions):
            partition_name = f"{table_name}_hash_{i}"
            create_stmt = (
                f"CREATE TABLE IF NOT EXISTS {partition_name} "
                f"PARTITION OF {table_name} "
                f"FOR VALUES WITH (MODULUS {num_partitions}, REMAINDER {i})"
            )
            try:
                self.sql_manager.execute(self.db_name, create_stmt, commit=True)
                logger.info(f"Created hash partition: {partition_name}")
            except Exception as e:
                logger.error(f"Failed to create hash partition {partition_name}: {e}")

    def create_range_partition(
        self,
        table_name: str,
        column: str,
        ranges: List[Dict[str, Any]]
    ) -> None:
        """
        Create range-based partitions for a table.

        Args:
            table_name (str): Parent table name.
            column (str): Column to partition by.
            ranges (List[Dict]): List of dicts with 'start' and 'end' keys.
        """
        for r in ranges:
            partition_name = f"{table_name}_{r['start']}_{r['end']}"
            create_stmt = (
                f"CREATE TABLE IF NOT EXISTS {partition_name} "
                f"PARTITION OF {table_name} "
                f"FOR VALUES FROM ({r['start']}) TO ({r['end']})"
            )
            try:
                self.sql_manager.execute(self.db_name, create_stmt, commit=True)
                logger.info(f"Created range partition: {partition_name}")
            except Exception as e:
                logger.error(f"Failed to create range partition {partition_name}: {e}")

    # -------------------------
    # Partition Monitoring
    # -------------------------
    def monitor_partitions(self, table_name: str) -> None:
        """
        Monitor partition sizes and usage.

        Args:
            table_name (str): Parent table name.
        """
        query = f"""
        SELECT
            relname AS partition_name,
            pg_size_pretty(pg_total_relation_size(relid)) AS total_size
        FROM pg_catalog.pg_statio_user_tables
        WHERE relname LIKE '{table_name}_%';
        """
        try:
            results = self.sql_manager.fetch_all(self.db_name, query)
            for row in results:
                logger.info(f"Partition {row['partition_name']} size: {row['total_size']}")
        except Exception as e:
            logger.error(f"Failed to monitor partitions for {table_name}: {e}")

    def alert_large_partitions(self, table_name: str, size_limit_mb: int = 1024) -> None:
        """
        Alert if any partition exceeds the size limit.

        Args:
            table_name (str): Parent table name.
            size_limit_mb (int): Size threshold in MB.
        """
        query = f"""
        SELECT
            relname AS partition_name,
            pg_total_relation_size(relid)/1024/1024 AS size_mb
        FROM pg_catalog.pg_statio_user_tables
        WHERE relname LIKE '{table_name}_%';
        """
        try:
            results = self.sql_manager.fetch_all(self.db_name, query)
            for row in results:
                if row["size_mb"] > size_limit_mb:
                    logger.warning(f"Partition {row['partition_name']} is large: {row['size_mb']} MB")
        except Exception as e:
            logger.error(f"Failed to alert large partitions for {table_name}: {e}")
          
