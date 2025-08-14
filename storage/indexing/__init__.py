# storage/indexing/__init__.py

"""
Indexing Package
----------------
Enterprise-grade module for managing SQL indexes.

Features:
- Import IndexManager
- Provide high-level helper functions for index creation and optimization
- Type hints and docstrings for maintainability
"""

from typing import List, Dict, Any
from .index_manager import IndexManager
from ..sql_manager import SQLManager


def create_indexes(
    sql_manager: SQLManager,
    db_name: str,
    indexes_config: List[Dict[str, Any]]
) -> None:
    """
    Create indexes on multiple tables based on configuration.

    Args:
        sql_manager (SQLManager): SQLManager instance for database execution.
        db_name (str): Target database name.
        indexes_config (List[Dict]): List of index configurations. Each config can include:
            - table_name (str): Table to index.
            - columns (List[str]): Columns to include in the index.
            - unique (bool, optional): Whether the index should be unique.
            - index_name (str, optional): Custom index name.
    """
    index_mgr = IndexManager(sql_manager, db_name)

    for config in indexes_config:
        table_name = config["table_name"]
        columns = config["columns"]
        unique = config.get("unique", False)
        index_name = config.get("index_name")

        index_mgr.create_index(table_name=table_name, columns=columns, unique=unique, index_name=index_name)


def optimize_indexes(
    sql_manager: SQLManager,
    db_name: str,
    tables: List[str]
) -> None:
    """
    Optimize indexes for given tables.

    Args:
        sql_manager (SQLManager): SQLManager instance for database execution.
        db_name (str): Target database name.
        tables (List[str]): List of table names to optimize indexes for.
    """
    index_mgr = IndexManager(sql_manager, db_name)

    for table in tables:
        index_mgr.optimize_indexes(table)
      
