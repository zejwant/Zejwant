# db_connector.py
"""
Enterprise-grade database connector module.

Supports SQL (PostgreSQL, MySQL) and NoSQL (MongoDB, Cassandra) databases.

Features:
- Connection pooling and retry logic
- Custom query execution
- Batch fetching
- Logging and error handling
- Returns results as Pandas DataFrame
- Async-ready for MongoDB and Cassandra
"""

from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd
import logging
import asyncio
import time
import json

# SQL
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

# MongoDB
from motor.motor_asyncio import AsyncIOMotorClient

# Cassandra
from cassandra.cluster import Cluster, Session
from cassandra.query import SimpleStatement, dict_factory
from cassandra.auth import PlainTextAuthProvider

# Logging
logger = logging.getLogger("db_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class DBConnector:
    def __init__(
        self,
        db_type: str,
        uri: str,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        **kwargs
    ):
        """
        Initialize database connector.

        Args:
            db_type (str): 'postgresql', 'mysql', 'mongodb', 'cassandra'
            uri (str): Database connection URI
            max_retries (int): Retry attempts for failed queries
            retry_backoff (float): Exponential backoff factor
            kwargs: Extra parameters (user, password, keyspace, etc.)
        """
        self.db_type = db_type.lower()
        self.uri = uri
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.kwargs = kwargs

        self.engine: Optional[Engine] = None
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.cassandra_session: Optional[Session] = None

        if self.db_type in ("postgresql", "mysql"):
            self._init_sql()
        elif self.db_type == "mongodb":
            self._init_mongo()
        elif self.db_type == "cassandra":
            self._init_cassandra()
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")

    # -------------------- SQL Init --------------------
    def _init_sql(self):
        try:
            self.engine = sqlalchemy.create_engine(
                self.uri,
                poolclass=QueuePool,
                pool_size=self.kwargs.get("pool_size", 5),
                max_overflow=self.kwargs.get("max_overflow", 10),
            )
            logger.info(f"SQL engine initialized for {self.db_type}")
        except Exception as e:
            logger.error(f"Failed to initialize SQL engine: {e}")
            raise

    # -------------------- MongoDB Init --------------------
    def _init_mongo(self):
        try:
            self.mongo_client = AsyncIOMotorClient(self.uri, **self.kwargs)
            logger.info("MongoDB client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB client: {e}")
            raise

    # -------------------- Cassandra Init --------------------
    def _init_cassandra(self):
        try:
            auth_provider = None
            if "username" in self.kwargs and "password" in self.kwargs:
                auth_provider = PlainTextAuthProvider(
                    username=self.kwargs["username"], password=self.kwargs["password"]
                )
            cluster = Cluster(self.kwargs.get("hosts", ["127.0.0.1"]), auth_provider=auth_provider)
            self.cassandra_session = cluster.connect(self.kwargs.get("keyspace"))
            self.cassandra_session.row_factory = dict_factory
            logger.info("Cassandra session initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Cassandra session: {e}")
            raise

    # -------------------- SQL Query --------------------
    def execute_sql(self, query: str, batch_size: int = 1000) -> pd.DataFrame:
        """
        Execute SQL query with retries and batch fetching.

        Args:
            query (str): SQL query string
            batch_size (int): Number of rows per batch

        Returns:
            pd.DataFrame: Query results
        """
        if not self.engine:
            raise ValueError("SQL engine is not initialized")

        retries = 0
        results = []

        while retries <= self.max_retries:
            try:
                with self.engine.connect() as conn:
                    cursor = conn.execution_options(stream_results=True).execute(text(query))
                    batch = cursor.fetchmany(batch_size)
                    while batch:
                        results.extend([dict(row) for row in batch])
                        batch = cursor.fetchmany(batch_size)
                break
            except Exception as e:
                wait = self.retry_backoff * (2 ** retries)
                logger.warning(f"SQL query failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                retries += 1
                if retries > self.max_retries:
                    logger.error("Max retries reached for SQL query")
                    raise e

        return pd.DataFrame(results)

    # -------------------- MongoDB Query --------------------
    async def query_mongo(
        self, db_name: str, collection: str, filter_query: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Query MongoDB collection asynchronously.

        Args:
            db_name (str): Database name
            collection (str): Collection name
            filter_query (dict, optional): MongoDB filter

        Returns:
            pd.DataFrame: Query results
        """
        if not self.mongo_client:
            raise ValueError("MongoDB client not initialized")

        filter_query = filter_query or {}
        retries = 0
        results = []

        while retries <= self.max_retries:
            try:
                cursor = self.mongo_client[db_name][collection].find(filter_query)
                async for doc in cursor:
                    results.append(doc)
                break
            except Exception as e:
                wait = self.retry_backoff * (2 ** retries)
                logger.warning(f"MongoDB query failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
                retries += 1
                if retries > self.max_retries:
                    logger.error("Max retries reached for MongoDB query")
                    raise e

        return pd.DataFrame(results)

    # -------------------- Cassandra Query --------------------
    def query_cassandra(self, query: str, batch_size: int = 1000) -> pd.DataFrame:
        """
        Execute Cassandra query with batching.

        Args:
            query (str): CQL query
            batch_size (int): Rows per batch

        Returns:
            pd.DataFrame: Query results
        """
        if not self.cassandra_session:
            raise ValueError("Cassandra session not initialized")

        retries = 0
        results = []

        while retries <= self.max_retries:
            try:
                statement = SimpleStatement(query, fetch_size=batch_size)
                rows = self.cassandra_session.execute(statement)
                for row in rows:
                    results.append(row)
                break
            except Exception as e:
                wait = self.retry_backoff * (2 ** retries)
                logger.warning(f"Cassandra query failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                retries += 1
                if retries > self.max_retries:
                    logger.error("Max retries reached for Cassandra query")
                    raise e

        return pd.DataFrame(results)
          
