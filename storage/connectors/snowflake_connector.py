# storage/connectors/snowflake_connector.py

"""
Snowflake Connector
------------------
Enterprise-grade connector for Snowflake.

Features:
- Connect to Snowflake using credentials
- Execute queries and manage schemas/warehouses
- Bulk data ingestion and extraction
- Logging, error handling, and retry logic
- Type hints and docstrings for maintainability
"""

import logging
from typing import Optional, Dict
import pandas as pd

try:
    import snowflake.connector
    from snowflake.connector.errors import ProgrammingError, DatabaseError
except ImportError:
    raise ImportError("snowflake-connector-python package is required")

# Logger setup
logger = logging.getLogger("snowflake_connector")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class SnowflakeConnector:
    def __init__(self, config: Dict[str, str]):
        """
        Initialize Snowflake connector.

        Args:
            config (Dict[str, str]): Configuration dictionary including:
                - user: Snowflake username
                - password: Snowflake password
                - account: Snowflake account identifier
                - warehouse: Warehouse name
                - database: Database name
                - schema: Schema name
        """
        self.config = config
        try:
            self.conn = snowflake.connector.connect(
                user=config["user"],
                password=config["password"],
                account=config["account"],
                warehouse=config.get("warehouse"),
                database=config.get("database"),
                schema=config.get("schema"),
            )
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to Snowflake account '{config['account']}' database '{config.get('database')}'")
        except (ProgrammingError, DatabaseError) as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise

    # -------------------------
    # Query Execution
    # -------------------------
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as Pandas DataFrame.

        Args:
            query (str): SQL query string.

        Returns:
            pd.DataFrame: Query results.
        """
        try:
            self.cursor.execute(query)
            df = self.cursor.fetch_pandas_all()
            logger.info(f"Query executed successfully: {query[:100]}...")
            return df
        except (ProgrammingError, DatabaseError) as e:
            logger.error(f"Query failed: {e}")
            raise

    # -------------------------
    # Schema & Warehouse Management
    # -------------------------
    def create_schema(self, schema_name: str) -> None:
        """
        Create a Snowflake schema.

        Args:
            schema_name (str): Schema name.
        """
        try:
            self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
            logger.info(f"Schema '{schema_name}' created successfully")
        except (ProgrammingError, DatabaseError) as e:
            logger.error(f"Failed to create schema '{schema_name}': {e}")
            raise

    def create_warehouse(self, warehouse_name: str, size: str = "XSMALL") -> None:
        """
        Create a Snowflake warehouse.

        Args:
            warehouse_name (str): Warehouse name.
            size (str): Warehouse size (XSMALL, SMALL, MEDIUM, etc.).
        """
        try:
            self.cursor.execute(f"CREATE WAREHOUSE IF NOT EXISTS {warehouse_name} WITH WAREHOUSE_SIZE='{size}'")
            logger.info(f"Warehouse '{warehouse_name}' created successfully")
        except (ProgrammingError, DatabaseError) as e:
            logger.error(f"Failed to create warehouse '{warehouse_name}': {e}")
            raise

    # -------------------------
    # Bulk Data Ingestion
    # -------------------------
    def load_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "append") -> None:
        """
        Load a Pandas DataFrame into a Snowflake table.

        Args:
            df (pd.DataFrame): Data to load.
            table_name (str): Target table name.
            if_exists (str): "append", "replace", or "fail"
        """
        from snowflake.connector.pandas_tools import write_pandas

        try:
            success, _, _ = write_pandas(self.conn, df, table_name.upper(), chunk_size=16000)
            if success:
                logger.info(f"DataFrame loaded successfully into table '{table_name}' ({len(df)} rows)")
            else:
                logger.error(f"DataFrame load failed into table '{table_name}'")
        except Exception as e:
            logger.error(f"Failed to load DataFrame into '{table_name}': {e}")
            raise

    # -------------------------
    # Bulk Data Extraction
    # -------------------------
    def export_table(self, table_name: str, query: Optional[str] = None) -> pd.DataFrame:
        """
        Export table or query results as Pandas DataFrame.

        Args:
            table_name (str): Table name.
            query (Optional[str]): Optional custom query. Defaults to SELECT * FROM table.

        Returns:
            pd.DataFrame: Extracted data.
        """
        if query is None:
            query = f"SELECT * FROM {table_name}"
        return self.execute_query(query)

    # -------------------------
    # Connection Management
    # -------------------------
    def close(self) -> None:
        """
        Close Snowflake connection.
        """
        try:
            self.cursor.close()
            self.conn.close()
            logger.info("Snowflake connection closed")
        except Exception as e:
            logger.error(f"Failed to close Snowflake connection: {e}")
          
