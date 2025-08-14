# storage/utils/performance_monitor.py

"""
Performance Monitor
-------------------
Enterprise-grade utilities to track database query performance, table growth,
and overall storage health.

Features:
- Track query execution time
- Alert on slow queries and table growth
- Provide structured logs and optional dashboards
- Logging, error handling, and type hints
"""

import logging
import time
from typing import Any, Dict, Optional

# Logger setup
logger = logging.getLogger("performance_monitor")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# -------------------------
# Query Performance Tracking
# -------------------------
def log_query_performance(query: str, connection: Any, threshold: float = 2.0) -> None:
    """
    Execute a query and log performance metrics.
    Alerts if execution exceeds threshold (seconds).

    Args:
        query (str): SQL query to execute.
        connection (Any): Database connection object.
        threshold (float): Threshold in seconds for slow query alerts.
    """
    start_time = time.time()
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        duration = time.time() - start_time
        if duration > threshold:
            logger.warning(f"Slow query detected ({duration:.2f}s): {query[:100]}...")
        else:
            logger.info(f"Query executed in {duration:.2f}s")
    except Exception as e:
        logger.error(f"Failed to execute query for performance monitoring: {e}")
        raise


# -------------------------
# Table Growth Monitoring
# -------------------------
def monitor_table_growth(table_name: str, connection: Any, max_rows: Optional[int] = None) -> None:
    """
    Monitor table row count and alert if exceeding max_rows.

    Args:
        table_name (str): Table to monitor.
        connection (Any): Database connection.
        max_rows (Optional[int]): Threshold row count to trigger alert.
    """
    try:
        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        if max_rows and row_count > max_rows:
            logger.warning(f"Table '{table_name}' exceeded row threshold: {row_count} rows")
        else:
            logger.info(f"Table '{table_name}' row count: {row_count}")
    except Exception as e:
        logger.error(f"Failed to monitor table growth for '{table_name}': {e}")
        raise


# -------------------------
# Storage Health Monitoring
# -------------------------
def monitor_storage_health(connection: Any) -> Dict[str, Any]:
    """
    Monitor storage health metrics: table counts, disk usage, and anomalies.

    Args:
        connection (Any): Database connection object.

    Returns:
        Dict[str, Any]: Summary of storage health metrics.
    """
    metrics = {}
    try:
        cursor = connection.cursor()
        # Count total tables
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'")
        metrics["total_tables"] = cursor.fetchone()[0]

        # Example disk usage monitoring (Postgres/Redshift)
        cursor.execute("""
        SELECT
            table_name,
            pg_total_relation_size(table_name) AS total_size
        FROM information_schema.tables
        WHERE table_schema='public';
        """)
        metrics["table_sizes"] = {row[0]: row[1] for row in cursor.fetchall()}

        # Log alerts for tables > 1GB as example
        for table, size in metrics["table_sizes"].items():
            if size > 1_000_000_000:  # 1GB threshold
                logger.warning(f"Table '{table}' size exceeds 1GB: {size} bytes")

        logger.info(f"Storage health metrics collected: {metrics}")
    except Exception as e:
        logger.error(f"Failed to monitor storage health: {e}")
        raise

    return metrics
          
