# storage/data_loader.py

"""
Data Loader
-----------
Enterprise-grade module for loading data into SQL tables.

Features:
- Bulk insert / batch load
- Upsert / conflict handling
- Transactional safety with rollback
- Integration with SQLManager and TableManager
- Logging and performance monitoring
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .sql_manager import SQLManager
from .table_manager import TableManager

# Logger setup
logger = logging.getLogger("data_loader")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class DataLoader:
    def __init__(self, sql_manager: SQLManager, table_manager: TableManager):
        """
        Initialize DataLoader with SQLManager and TableManager.

        Args:
            sql_manager (SQLManager): SQL connection & execution manager.
            table_manager (TableManager): TableManager instance for schema reference.
        """
        self.sql_manager = sql_manager
        self.table_manager = table_manager

    def load_data(
        self,
        table_name: str,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        batch_size: int = 1000,
        overwrite: bool = False,
        upsert: bool = False,
        conflict_columns: Optional[List[str]] = None
    ) -> None:
        """
        Load data into a SQL table with optional batching and upsert.

        Args:
            table_name (str): Target table.
            data (DataFrame or List[Dict]): Data to insert.
            batch_size (int): Number of rows per batch insert.
            overwrite (bool): Drop table before loading if True.
            upsert (bool): Perform upsert (insert or update on conflict) if True.
            conflict_columns (List[str], optional): Columns to resolve conflicts on for upsert.
        """
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        elif isinstance(data, list):
            records = data
        else:
            raise TypeError("Data must be a Pandas DataFrame or list of dicts")

        # Drop table if overwrite is True
        if overwrite:
            self.table_manager.drop_table(table_name)

        # Ensure table exists
        schema = self.table_manager.fetch_table_schema(table_name)
        if not schema:
            if isinstance(data, pd.DataFrame):
                inferred_schema = {col: "TEXT" for col in data.columns}  # default type TEXT
            else:
                inferred_schema = {k: "TEXT" for k in data[0].keys()} if data else {}
            self.table_manager.create_or_update_table(table_name, inferred_schema)
            schema = inferred_schema

        # Split into batches
        total = len(records)
        logger.info(f"Loading {total} rows into '{table_name}' in batches of {batch_size}")
        for i in range(0, total, batch_size):
            batch = records[i:i + batch_size]
            insert_query = self._build_insert_query(table_name, schema, batch, upsert, conflict_columns)
            self.sql_manager.execute(self.table_manager.db_name, insert_query, commit=True, batch=batch)

        logger.info(f"Successfully loaded {total} rows into '{table_name}'")

    # -------------------------
    # Helper Methods
    # -------------------------
    def _build_insert_query(
        self,
        table_name: str,
        schema: Dict[str, str],
        batch: List[Dict[str, Any]],
        upsert: bool = False,
        conflict_columns: Optional[List[str]] = None
    ) -> str:
        """
        Build SQL insert query with optional upsert for batch.

        Args:
            table_name (str): Target table.
            schema (Dict[str, str]): Column schema.
            batch (List[Dict]): Records to insert.
            upsert (bool): Enable upsert.
            conflict_columns (List[str], optional): Columns for conflict resolution.

        Returns:
            str: SQL query string.
        """
        columns = list(schema.keys())
        placeholders = ", ".join([f":{col}" for col in columns])
        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

        # Handle upsert for PostgreSQL
        if upsert and conflict_columns:
            conflict_cols = ", ".join(conflict_columns)
            update_cols = ", ".join([f"{col}=EXCLUDED.{col}" for col in columns if col not in conflict_columns])
            insert_sql += f" ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_cols}"

        return insert_sql
          
