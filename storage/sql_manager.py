# storage/sql_manager.py

"""
SQL Manager
-----------
Enterprise-grade SQL connection manager and query executor.

Features:
- Connection pooling for PostgreSQL, MySQL, and MSSQL
- Synchronous and asynchronous query execution
- Prepared statements for efficiency
- Batch operations and transactional support
- Query performance logging and retry logic for transient failures
- Scalable, type-hinted, and maintainable design
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.pool import QueuePool

# Async support
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncConnection
except ImportError:
    AsyncEngine = None
    AsyncConnection = None

# Logger setup
logger = logging.getLogger("sql_manager")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# =========================
# Retry Decorator
# =========================
def retry(
    retries: int = 3, delay: float = 2.0, exceptions: tuple = (OperationalError,)
) -> Callable:
    """
    Retry decorator for transient failures.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Query failed with {e}, retrying ({attempt + 1}/{retries})...")
                    time.sleep(delay)
            raise
        return wrapper
    return decorator


# =========================
# SQLManager Class
# =========================
class SQLManager:
    def __init__(self, db_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize SQLManager with multiple database connections.

        Args:
            db_configs (Dict[str, Dict[str, Any]]): Dictionary with database connection info.
                Example:
                {
                    "postgres": {
                        "driver": "postgresql+psycopg2",
                        "host": "localhost",
                        "port": 5432,
                        "username": "user",
                        "password": "pass",
                        "database": "db",
                        "pool_size": 10,
                        "max_overflow": 5
                    },
                    "mysql": { ... }
                }
        """
        self.engines: Dict[str, Engine] = {}
        for name, cfg in db_configs.items():
            driver = cfg.get("driver")
            host = cfg.get("host")
            port = cfg.get("port")
            username = cfg.get("username")
            password = cfg.get("password")
            database = cfg.get("database")
            pool_size = cfg.get("pool_size", 5)
            max_overflow = cfg.get("max_overflow", 10)
            async_flag = cfg.get("async", False)

            connection_str = f"{driver}://{username}:{password}@{host}:{port}/{database}"

            if async_flag and AsyncEngine is not None:
                self.engines[name] = create_async_engine(
                    connection_str, pool_size=pool_size, max_overflow=max_overflow
                )
            else:
                self.engines[name] = create_engine(
                    connection_str,
                    poolclass=QueuePool,
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                    future=True
                )
            logger.info(f"Initialized engine for database '{name}'")

    @retry()
    def execute(
        self,
        db_name: str,
        query: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
        commit: bool = False,
        fetch: bool = False,
        batch: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a query with optional parameters, batch, and transaction support.

        Args:
            db_name (str): Target database name configured in db_configs.
            query (str): SQL query string.
            params (dict or list, optional): Parameters for prepared statements.
            commit (bool): Whether to commit transaction.
            fetch (bool): Whether to fetch results.
            batch (List[Dict], optional): Batch of parameter dicts for multiple inserts.

        Returns:
            List[Dict[str, Any]]: Query results if fetch=True.
        """
        engine: Engine = self.engines[db_name]
        results = None
        start_time = time.time()
        with engine.begin() as conn:
            try:
                if batch:
                    for b in batch:
                        conn.execute(text(query), b)
                else:
                    conn.execute(text(query), params or {})
                if commit:
                    conn.commit()
                if fetch:
                    results = [dict(row) for row in conn.execute(text(query))]
            except SQLAlchemyError as e:
                conn.rollback()
                logger.error(f"Query failed: {e}")
                raise
        duration = time.time() - start_time
        logger.info(f"Executed query on {db_name} in {duration:.4f}s")
        return results

    async def execute_async(
        self,
        db_name: str,
        query: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
        commit: bool = False,
        fetch: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Async query execution.

        Requires AsyncEngine support.

        Args:
            db_name (str): Target database.
            query (str): SQL query string.
            params (dict or list, optional): Query parameters.
            commit (bool): Commit transaction.
            fetch (bool): Fetch results.

        Returns:
            List[Dict[str, Any]]: Query results if fetch=True.
        """
        if AsyncEngine is None:
            raise RuntimeError("Async support not installed")
        engine: AsyncEngine = self.engines[db_name]  # type: ignore
        results = None
        start_time = time.time()
        async with engine.begin() as conn:  # type: ignore
            try:
                await conn.execute(text(query), params or {})
                if commit:
                    await conn.commit()
                if fetch:
                    res = await conn.execute(text(query))
                    results = [dict(row) for row in res]
            except SQLAlchemyError as e:
                await conn.rollback()
                logger.error(f"Async query failed: {e}")
                raise
        duration = time.time() - start_time
        logger.info(f"Executed async query on {db_name} in {duration:.4f}s")
        return results
          
