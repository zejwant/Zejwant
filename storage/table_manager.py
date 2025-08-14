# storage/table_manager.py

"""
Table Manager
-------------
Enterprise-grade module to dynamically create, update, and manage database tables.

Features:
- Create tables dynamically based on metadata
- Update schemas and apply constraints
- Support primary/foreign keys and indexes
- Validate schema against MetadataManager
- Integration with SQLManager for execution
- Audit logs for schema changes
"""

import logging
from typing import Any, Dict, List, Optional

from .sql_manager import SQLManager
from .metadata import MetadataManager

# Logger setup
logger = logging.getLogger("table_manager")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class TableManager:
    def __init__(self, sql_manager: SQLManager, metadata_manager: MetadataManager, db_name: str):
        """
        Initialize TableManager with SQLManager and MetadataManager.

        Args:
            sql_manager (SQLManager): SQL connection & execution manager.
            metadata_manager (MetadataManager): Tracks table schemas and lineage.
            db_name (str): Target database name.
        """
        self.sql_manager = sql_manager
        self.metadata_manager = metadata_manager
        self.db_name = db_name

    # -------------------------
    # Table Creation / Updates
    # -------------------------
    def create_or_update_table(
        self,
        table_name: str,
        schema: Dict[str, str],
        primary_keys: Optional[List[str]] = None,
        foreign_keys: Optional[Dict[str, str]] = None,
        indexes: Optional[List[str]] = None,
        overwrite: bool = False
    ) -> None:
        """
        Create a new table or update an existing table schema.

        Args:
            table_name (str): Target table name.
            schema (Dict[str, str]): Column name -> SQL type mapping.
            primary_keys (List[str], optional): List of columns for primary key.
            foreign_keys (Dict[str, str], optional): {column: referenced_table(column)}
            indexes (List[str], optional): Columns to create indexes on.
            overwrite (bool): If True, drop existing table before creating.
        """
        current_schema = self.metadata_manager.get_schema(table_name)
        if current_schema:
            diff = self.metadata_manager.detect_schema_anomalies(table_name, schema)
            if diff and not overwrite:
                logger.info(f"Updating schema for table '{table_name}' with changes: {diff}")
        else:
            logger.info(f"Creating new table '{table_name}'")

        # Drop table if overwrite is True
        if overwrite:
            drop_query = f"DROP TABLE IF EXISTS {table_name}"
            self.sql_manager.execute(self.db_name, drop_query, commit=True)
            logger.info(f"Dropped table '{table_name}' before re-creation")

        # Construct CREATE TABLE statement
        column_defs = []
        for col, col_type in schema.items():
            column_defs.append(f"{col} {col_type}")

        # Primary key
        if primary_keys:
            pk = ", ".join(primary_keys)
            column_defs.append(f"PRIMARY KEY ({pk})")

        # Foreign keys
        if foreign_keys:
            for col, ref in foreign_keys.items():
                column_defs.append(f"FOREIGN KEY ({col}) REFERENCES {ref}")

        create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_defs)})"
        self.sql_manager.execute(self.db_name, create_query, commit=True)
        logger.info(f"Created/Updated table '{table_name}' successfully")

        # Create indexes
        if indexes:
            for idx_col in indexes:
                index_query = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{idx_col} ON {table_name}({idx_col})"
                self.sql_manager.execute(self.db_name, index_query, commit=True)
            logger.info(f"Indexes created on table '{table_name}': {indexes}")

        # Register schema in MetadataManager
        self.metadata_manager.register_schema(table_name, schema)
        logger.info(f"Schema registered for table '{table_name}' in metadata")

    # -------------------------
    # Utility Methods
    # -------------------------
    def drop_table(self, table_name: str) -> None:
        """
        Drop a table from the database.

        Args:
            table_name (str): Table name.
        """
        drop_query = f"DROP TABLE IF EXISTS {table_name}"
        self.sql_manager.execute(self.db_name, drop_query, commit=True)
        logger.info(f"Dropped table '{table_name}'")

    def fetch_table_schema(self, table_name: str) -> Optional[Dict[str, str]]:
        """
        Fetch table schema from metadata.

        Args:
            table_name (str): Table name.

        Returns:
            Dict[str, str] | None
        """
        schema = self.metadata_manager.get_schema(table_name)
        if schema:
            return schema.get("columns")
        logger.warning(f"Schema for table '{table_name}' not found in metadata")
        return None
        
