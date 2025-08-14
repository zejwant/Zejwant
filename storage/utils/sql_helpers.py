# storage/utils/sql_helpers.py

"""
SQL Helpers
-----------
Reusable SQL utility functions for query generation, dynamic table creation,
batch inserts, and performance tracking.

Features:
- Generate dynamic queries
- Create tables programmatically
- Execute batch inserts
- Logging, error handling, and performance hooks
- Type hints and docstrings for enterprise maintainability
"""

import logging
from typing import Any, List, Dict, Optional
import time

# Logger setup
logger = logging.getLogger("sql_helpers")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# -------------------------
# SQL Execution
# -------------------------
def execute_sql(query: str, connection: Any, fetch: bool = False) -> Optional[List[Dict[str, Any]]]:
    """
    Execute a SQL query with optional fetch results and performance logging.

    Args:
        query (str): SQL query string.
        connection (Any): Database connection with cursor() method.
        fetch (bool): Whether to fetch results (for SELECT queries).

    Returns:
        Optional[List[Dict[str, Any]]]: Query results as list of dicts if fetch=True.
    """
    start_time = time.time()
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        results = None
        if fetch:
            cols = [desc[0] for desc in cursor.description]
            results = [dict(zip(cols, row)) for row in cursor.fetchall()]
        connection.commit()
        logger.info(f"Executed query successfully in {time.time() - start_time:.2f}s")
        return results
    except Exception as e:
        logger.error(f"Failed to execute query: {e}\nQuery: {query}")
        connection.rollback()
        raise


# -------------------------
# Dynamic Table Creation
# -------------------------
def build_dynamic_create_table(table_name: str, columns: Dict[str, str], primary_key: Optional[str] = None) -> str:
    """
    Generate a CREATE TABLE SQL statement dynamically.

    Args:
        table_name (str): Table name.
        columns (Dict[str, str]): Column names and types, e.g., {"id": "INT", "name": "VARCHAR(255)"}
        primary_key (Optional[str]): Column to set as primary key.

    Returns:
        str: CREATE TABLE SQL string.
    """
    col_defs = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
    pk_def = f", PRIMARY KEY({primary_key})" if primary_key else ""
    create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs}{pk_def});"
    logger.debug(f"Generated CREATE TABLE statement: {create_stmt}")
    return create_stmt


# -------------------------
# Batch Inserts
# -------------------------
def build_insert_statement(table_name: str, columns: List[str], batch_size: int = 1000) -> str:
    """
    Generate an INSERT INTO statement with placeholders for batch execution.

    Args:
        table_name (str): Target table name.
        columns (List[str]): List of column names.
        batch_size (int): Number of rows per batch (used for performance hooks).

    Returns:
        str: Parameterized INSERT statement.
    """
    cols = ", ".join(columns)
    placeholders = ", ".join([f"%({col})s" for col in columns])
    insert_stmt = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders});"
    logger.debug(f"Generated INSERT statement for table '{table_name}': {insert_stmt}")
    return insert_stmt


def batch_insert(connection: Any, table_name: str, rows: List[Dict[str, Any]], batch_size: int = 500) -> None:
    """
    Execute batch inserts into a table.

    Args:
        connection (Any): Database connection.
        table_name (str): Target table.
        rows (List[Dict[str, Any]]): List of row dictionaries.
        batch_size (int): Number of rows per batch.
    """
    if not rows:
        logger.warning(f"No rows provided for batch insert into '{table_name}'")
        return

    columns = list(rows[0].keys())
    insert_sql = build_insert_statement(table_name, columns)
    cursor = connection.cursor()

    start_time = time.time()
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            cursor.executemany(insert_sql, batch)
        connection.commit()
        logger.info(f"Inserted {len(rows)} rows into '{table_name}' in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Batch insert failed: {e}")
        connection.rollback()
        raise


# -------------------------
# Utility
# -------------------------
def fetch_table_schema(connection: Any, table_name: str) -> List[Dict[str, str]]:
    """
    Fetch column names and types for a given table.

    Args:
        connection (Any): Database connection.
        table_name (str): Table name.

    Returns:
        List[Dict[str, str]]: List of column metadata: {"column_name": name, "data_type": type}
    """
    query = f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = '{table_name}';
    """
    results = execute_sql(query, connection, fetch=True)
    return results or []
  
