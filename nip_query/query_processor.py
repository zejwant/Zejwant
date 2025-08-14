"""
query_processor.py
------------------

Executes SQL queries securely against PostgreSQL.

Features:
    - Accepts SQL + parameters for parameterized execution (avoids injection).
    - Connects via psycopg2 or SQLAlchemy (default: SQLAlchemy for pooling).
    - Uses read-only role for SELECT queries (enforced by DB config).
    - Catches & logs errors with stack trace.
    - Returns results as pandas DataFrame.
    - Supports pagination for large result sets.

Author: Varun-engineer mode (20+ years experience)
"""

from __future__ import annotations
import logging
import traceback
from typing import Dict, Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
DB_READONLY_URI = "postgresql+psycopg2://readonly_user:readonly_pass@localhost:5432/mydb"

# Connection pooling for performance
engine = create_engine(
    DB_READONLY_URI,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800
)

def execute_query(
    sql: str,
    params: Optional[Dict[str, Any]] = None,
    page: int = 1,
    page_size: int = 1000
) -> pd.DataFrame:
    """Execute a SQL query safely with parameter binding and pagination."""
    logger.debug("Preparing to execute query", extra={"sql": sql, "params": params})

    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed in read-only mode.")

    offset = (page - 1) * page_size
    paginated_sql = f"{sql} LIMIT :limit OFFSET :offset"
    params = {**(params or {}), "limit": page_size, "offset": offset}

    try:
        with engine.connect() as conn:
            logger.info(f"Executing SQL (page={page}, page_size={page_size})")
            result_df = pd.read_sql(text(paginated_sql), conn, params=params)
            logger.debug("Query execution successful", extra={"rows_returned": len(result_df)})
            return result_df
    except (SQLAlchemyError, Exception) as e:
        logger.error(f"Error during query execution: {e}")
        logger.error(traceback.format_exc())
        raise

# ---------- Unit Test Stubs ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    try:
        df = execute_query("SELECT * FROM sales WHERE region = :region", {"region": "North"}, page=1, page_size=5)
        print(df.head())
    except Exception as e:
        print(f"Execution failed: {e}")
      
