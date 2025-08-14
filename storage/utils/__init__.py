# storage/utils/__init__.py

"""
Utils Package
-------------
Enterprise-grade utilities for the storage layer.

Features:
- Import core SQL helper functions, data validation, and performance monitoring
- Provide high-level helper functions for SQL execution and monitoring
- Type hints and docstrings for maintainability
"""

from typing import Any, Dict, Optional
from .sql_helpers import execute_sql, fetch_table_schema, build_dynamic_query
from .data_validation_sql import validate_foreign_keys, check_nulls, validate_constraints
from .performance_monitor import log_query_performance, monitor_storage_health


def run_query_with_monitoring(query: str, connection: Any, log_metrics: bool = True) -> Optional[Any]:
    """
    Execute a SQL query and optionally log performance metrics.

    Args:
        query (str): SQL query string.
        connection (Any): Database connection object compatible with execute_sql.
        log_metrics (bool): Whether to log query performance metrics.

    Returns:
        Optional[Any]: Query results, if any.
    """
    result = execute_sql(query, connection)
    if log_metrics:
        log_query_performance(query, connection)
    return result


def validate_table_data(table_name: str, connection: Any) -> Dict[str, Any]:
    """
    Run validation checks on a table's data, including nulls, foreign keys, and constraints.

    Args:
        table_name (str): Name of the table to validate.
        connection (Any): Database connection object.

    Returns:
        Dict[str, Any]: Validation report with success/failure metrics.
    """
    report = {
        "null_check": check_nulls(table_name, connection),
        "foreign_key_check": validate_foreign_keys(table_name, connection),
        "constraints_check": validate_constraints(table_name, connection),
    }
    return report


def monitor_storage_system(connection: Any) -> None:
    """
    Run storage performance and health monitoring.

    Args:
        connection (Any): Database connection object.
    """
    monitor_storage_health(connection)
  
