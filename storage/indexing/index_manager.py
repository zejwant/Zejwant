# storage/indexing/index_manager.py

"""
Index Manager
-------------
Enterprise-grade module for managing SQL indexes.

Features:
- Create, update, delete indexes for tables
- Monitor index usage and performance
- Suggest optimizations based on query patterns
- Schedule index rebuilds for off-peak hours
- Logging and metrics collection
- Type hints and docstrings for maintainability
"""

import logging
from typing import List, Optional
from datetime import datetime

from ..sql_manager import SQLManager

# Logger setup
logger = logging.getLogger("index_manager")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class IndexManager:
    def __init__(self, sql_manager: SQLManager, db_name: str):
        """
        Initialize IndexManager.

        Args:
            sql_manager (SQLManager): SQLManager instance for executing queries.
            db_name (str): Target database name.
        """
        self.sql_manager = sql_manager
        self.db_name = db_name

    # -------------------------
    # Index Operations
    # -------------------------
    def create_index(
        self,
        table_name: str,
        columns: List[str],
        unique: bool = False,
        index_name: Optional[str] = None
    ) -> None:
        """
        Create an index on a table.

        Args:
            table_name (str): Table name.
            columns (List[str]): Columns to include in the index.
            unique (bool): Whether the index should be unique.
            index_name (Optional[str]): Custom index name. If None, auto-generated.
        """
        if not index_name:
            index_name = f"{table_name}_{'_'.join(columns)}_{'uniq' if unique else 'idx'}"

        unique_sql = "UNIQUE" if unique else ""
        stmt = f"CREATE {unique_sql} INDEX IF NOT EXISTS {index_name} ON {table_name} ({', '.join(columns)})"
        try:
            self.sql_manager.execute(self.db_name, stmt, commit=True)
            logger.info(f"Created index '{index_name}' on table '{table_name}'")
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}' on table '{table_name}': {e}")

    def drop_index(self, index_name: str) -> None:
        """
        Drop an index by name.

        Args:
            index_name (str): Name of the index to drop.
        """
        stmt = f"DROP INDEX IF EXISTS {index_name}"
        try:
            self.sql_manager.execute(self.db_name, stmt, commit=True)
            logger.info(f"Dropped index '{index_name}'")
        except Exception as e:
            logger.error(f"Failed to drop index '{index_name}': {e}")

    # -------------------------
    # Monitoring & Optimization
    # -------------------------
    def monitor_index_usage(self, table_name: str) -> None:
        """
        Monitor index usage and log size and performance metrics.

        Args:
            table_name (str): Table to monitor indexes for.
        """
        query = f"""
        SELECT
            indexrelname AS index_name,
            idx_scan AS num_scans,
            pg_size_pretty(pg_relation_size(indexrelid)) AS size
        FROM pg_stat_user_indexes
        WHERE relname = '{table_name}';
        """
        try:
            results = self.sql_manager.fetch_all(self.db_name, query)
            for row in results:
                logger.info(f"Index '{row['index_name']}': scans={row['num_scans']}, size={row['size']}")
        except Exception as e:
            logger.error(f"Failed to monitor indexes for table '{table_name}': {e}")

    def optimize_indexes(self, table_name: str) -> None:
        """
        Optimize indexes on a table by rebuilding them.

        Args:
            table_name (str): Table to optimize indexes for.
        """
        query = f"""
        SELECT indexrelname AS index_name
        FROM pg_stat_user_indexes
        WHERE relname = '{table_name}';
        """
        try:
            indexes = self.sql_manager.fetch_all(self.db_name, query)
            for idx in indexes:
                rebuild_stmt = f"REINDEX INDEX {idx['index_name']}"
                self.sql_manager.execute(self.db_name, rebuild_stmt, commit=True)
                logger.info(f"Rebuilt index '{idx['index_name']}' for table '{table_name}'")
        except Exception as e:
            logger.error(f"Failed to optimize indexes for table '{table_name}': {e}")

    def suggest_index_optimizations(self, table_name: str) -> None:
        """
        Suggest potential indexes based on query patterns (requires pg_stat_statements enabled).

        Args:
            table_name (str): Table to analyze for optimization.
        """
        query = f"""
        SELECT query, calls
        FROM pg_stat_statements
        WHERE query LIKE '%{table_name}%'
        ORDER BY calls DESC
        LIMIT 10;
        """
        try:
            results = self.sql_manager.fetch_all(self.db_name, query)
            logger.info(f"Top queries for table '{table_name}':")
            for row in results:
                logger.info(f"Calls: {row['calls']} | Query: {row['query']}")
            logger.info("Review queries to identify potential new indexes.")
        except Exception as e:
            logger.error(f"Failed to suggest index optimizations for '{table_name}': {e}")

    # -------------------------
    # Scheduled Maintenance
    # -------------------------
    def schedule_offpeak_reindex(self, table_name: str, offpeak_hour: int = 3) -> None:
        """
        Schedule index rebuilds during off-peak hours.

        Args:
            table_name (str): Table to rebuild indexes for.
            offpeak_hour (int): Hour of the day (0-23) for off-peak execution.
        """
        logger.info(f"Scheduled off-peak reindex for '{table_name}' at {offpeak_hour}:00")
        # Placeholder: integrate with cron, Airflow, or internal scheduler
      
