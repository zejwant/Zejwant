# storage/utils/data_validation_sql.py

"""
Data Validation SQL
------------------
Enterprise-grade utilities for validating relational database integrity.

Features:
- Cross-table integrity checks
- Referential integrity validation (foreign keys)
- Duplicate detection
- Anomaly detection (e.g., unexpected nulls or outliers)
- Logging and error handling
- Type hints and docstrings
"""

import logging
from typing import Any, Dict, List, Optional

# Logger setup
logger = logging.getLogger("data_validation_sql")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# -------------------------
# Null Checks
# -------------------------
def check_nulls(table_name: str, connection: Any, columns: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Check for null values in the specified table and columns.

    Args:
        table_name (str): Name of the table.
        connection (Any): Database connection object.
        columns (Optional[List[str]]): Columns to check. If None, check all columns.

    Returns:
        Dict[str, int]: Mapping of column name to number of null values.
    """
    null_counts = {}
    try:
        cursor = connection.cursor()
        if not columns:
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'")
            columns = [row[0] for row in cursor.fetchall()]

        for col in columns:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL")
            count = cursor.fetchone()[0]
            null_counts[col] = count

        logger.info(f"Null check completed for table '{table_name}': {null_counts}")
    except Exception as e:
        logger.error(f"Failed to check nulls in '{table_name}': {e}")
        raise
    return null_counts


# -------------------------
# Referential Integrity
# -------------------------
def validate_foreign_keys(table_name: str, connection: Any, fk_mappings: Dict[str, str]) -> Dict[str, int]:
    """
    Validate referential integrity for foreign keys.

    Args:
        table_name (str): Name of the child table.
        connection (Any): Database connection object.
        fk_mappings (Dict[str, str]): Mapping of child column -> parent table.column

    Returns:
        Dict[str, int]: Number of violations for each foreign key.
    """
    violations = {}
    try:
        cursor = connection.cursor()
        for child_col, parent_ref in fk_mappings.items():
            parent_table, parent_col = parent_ref.split(".")
            query = f"""
            SELECT COUNT(*) 
            FROM {table_name} c
            LEFT JOIN {parent_table} p ON c.{child_col} = p.{parent_col}
            WHERE p.{parent_col} IS NULL AND c.{child_col} IS NOT NULL
            """
            cursor.execute(query)
            count = cursor.fetchone()[0]
            violations[child_col] = count

        logger.info(f"Foreign key validation for table '{table_name}': {violations}")
    except Exception as e:
        logger.error(f"Failed to validate foreign keys for '{table_name}': {e}")
        raise
    return violations


# -------------------------
# Duplicate Detection
# -------------------------
def detect_duplicates(table_name: str, connection: Any, subset_columns: Optional[List[str]] = None) -> int:
    """
    Detect duplicate rows in a table based on a subset of columns.

    Args:
        table_name (str): Table to check.
        connection (Any): Database connection.
        subset_columns (Optional[List[str]]): Columns to check duplicates on. Defaults to all columns.

    Returns:
        int: Number of duplicate rows found.
    """
    try:
        cursor = connection.cursor()
        if subset_columns:
            cols = ", ".join(subset_columns)
        else:
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'")
            cols = ", ".join([row[0] for row in cursor.fetchall()])

        query = f"""
        SELECT COUNT(*) - COUNT(DISTINCT {cols}) FROM {table_name}
        """
        cursor.execute(query)
        dup_count = cursor.fetchone()[0]
        logger.info(f"Duplicate check for table '{table_name}': {dup_count} duplicates")
        return dup_count
    except Exception as e:
        logger.error(f"Failed to detect duplicates in '{table_name}': {e}")
        raise


# -------------------------
# Basic Anomaly Detection
# -------------------------
def detect_anomalies(table_name: str, connection: Any, column: str, threshold: float = 3.0) -> int:
    """
    Detect anomalies in a numeric column using z-score method.

    Args:
        table_name (str): Table name.
        connection (Any): Database connection.
        column (str): Column to check for anomalies.
        threshold (float): Z-score threshold to classify as anomaly.

    Returns:
        int: Number of anomalous rows.
    """
    try:
        cursor = connection.cursor()
        # Calculate mean and std
        cursor.execute(f"SELECT AVG({column}), STDDEV({column}) FROM {table_name}")
        mean, stddev = cursor.fetchone()
        if stddev == 0 or stddev is None:
            logger.warning(f"No variance in column '{column}' for anomaly detection")
            return 0

        # Count anomalies
        query = f"""
        SELECT COUNT(*) 
        FROM {table_name} 
        WHERE ABS(({column} - {mean}) / {stddev}) > {threshold}
        """
        cursor.execute(query)
        count = cursor.fetchone()[0]
        logger.info(f"Anomaly detection for '{column}' in '{table_name}': {count} anomalies")
        return count
    except Exception as e:
        logger.error(f"Failed to detect anomalies in '{table_name}.{column}': {e}")
        raise
