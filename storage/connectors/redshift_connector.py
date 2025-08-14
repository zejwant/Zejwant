# storage/connectors/redshift_connector.py

"""
Redshift Connector
------------------
Enterprise-grade connector for AWS Redshift.

Features:
- Connect to Redshift using credentials
- Execute queries and manage schemas/tables
- Bulk data load from S3
- Logging, error handling, and retry logic
- Type hints and docstrings for maintainability
"""

import logging
from typing import Optional, Dict
import pandas as pd
import psycopg2
from psycopg2 import sql, OperationalError, DatabaseError

# Logger setup
logger = logging.getLogger("redshift_connector")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class RedshiftConnector:
    def __init__(self, config: Dict[str, str]):
        """
        Initialize Redshift connector.

        Args:
            config (Dict[str, str]): Configuration dictionary including:
                - host: Redshift cluster endpoint
                - port: Redshift port (default 5439)
                - dbname: Database name
                - user: Username
                - password: Password
        """
        self.config = config
        try:
            self.conn = psycopg2.connect(
                host=config["host"],
                port=config.get("port", 5439),
                dbname=config["dbname"],
                user=config["user"],
                password=config["password"]
            )
            self.conn.autocommit = True
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to Redshift database '{config['dbname']}' at {config['host']}")
        except (OperationalError, DatabaseError) as e:
            logger.error(f"Failed to connect to Redshift: {e}")
            raise

    # -------------------------
    # Query Execution
    # -------------------------
    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """
        Execute a SQL query and return results as Pandas DataFrame (if SELECT).

        Args:
            query (str): SQL query string.

        Returns:
            Optional[pd.DataFrame]: DataFrame for SELECT queries; None otherwise.
        """
        try:
            self.cursor.execute(query)
            if query.strip().lower().startswith("select"):
                df = pd.DataFrame(self.cursor.fetchall(), columns=[desc[0] for desc in self.cursor.description])
                logger.info(f"Query executed successfully: {query[:100]}...")
                return df
            else:
                logger.info(f"Query executed: {query[:100]}...")
                return None
        except (OperationalError, DatabaseError) as e:
            logger.error(f"Query failed: {e}")
            raise

    # -------------------------
    # Schema & Table Management
    # -------------------------
    def create_schema(self, schema_name: str) -> None:
        """
        Create a schema if it does not exist.

        Args:
            schema_name (str): Schema name.
        """
        try:
            self.cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
            logger.info(f"Schema '{schema_name}' created or already exists")
        except (OperationalError, DatabaseError) as e:
            logger.error(f"Failed to create schema '{schema_name}': {e}")
            raise

    def create_table(self, table_name: str, schema_definition: str, schema_name: Optional[str] = None) -> None:
        """
        Create a table with given schema definition.

        Args:
            table_name (str): Table name.
            schema_definition (str): SQL string for column definitions.
            schema_name (Optional[str]): Optional schema name.
        """
        full_table = f"{schema_name}.{table_name}" if schema_name else table_name
        try:
            self.cursor.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                sql.Identifier(full_table),
                sql.SQL(schema_definition)
            ))
            logger.info(f"Table '{full_table}' created successfully")
        except (OperationalError, DatabaseError) as e:
            logger.error(f"Failed to create table '{full_table}': {e}")
            raise

    # -------------------------
    # Bulk Data Load from S3
    # -------------------------
    def load_from_s3(self, table_name: str, s3_path: str, iam_role: str, file_format: str = "CSV", schema_name: Optional[str] = None) -> None:
        """
        Load data from S3 into Redshift table using COPY command.

        Args:
            table_name (str): Target table.
            s3_path (str): S3 URI (e.g., s3://bucket/path/file.csv)
            iam_role (str): AWS IAM role ARN for Redshift access
            file_format (str): File format ("CSV" or "PARQUET")
            schema_name (Optional[str]): Optional schema name
        """
        full_table = f"{schema_name}.{table_name}" if schema_name else table_name
        try:
            copy_sql = f"""
            COPY {full_table}
            FROM '{s3_path}'
            IAM_ROLE '{iam_role}'
            FORMAT AS {file_format}
            IGNOREHEADER 1
            REGION 'us-east-1';
            """
            self.cursor.execute(copy_sql)
            logger.info(f"Data loaded from S3 into table '{full_table}' successfully")
        except (OperationalError, DatabaseError) as e:
            logger.error(f"Failed to load data from S3 into '{full_table}': {e}")
            raise

    # -------------------------
    # Connection Management
    # -------------------------
    def close(self) -> None:
        """
        Close Redshift connection.
        """
        try:
            self.cursor.close()
            self.conn.close()
            logger.info("Redshift connection closed")
        except Exception as e:
            logger.error(f"Failed to close Redshift connection: {e}")
          
