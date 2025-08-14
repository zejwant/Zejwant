# storage/connectors/bigquery_connector.py

"""
BigQuery Connector
------------------
Enterprise-grade connector for Google BigQuery.

Features:
- Connect to BigQuery using service account or credentials
- Execute queries and manage tables/datasets
- Bulk data load from Pandas, CSV, or Parquet
- Export data to local or cloud storage
- Logging, error handling, and retry logic
- Type hints and docstrings for maintainability
"""

import logging
from typing import Optional, Union
import pandas as pd

try:
    from google.cloud import bigquery
    from google.api_core.exceptions import GoogleAPIError
except ImportError:
    raise ImportError("google-cloud-bigquery package is required")

# Logger setup
logger = logging.getLogger("bigquery_connector")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class BigQueryConnector:
    def __init__(self, config: dict):
        """
        Initialize BigQuery connector.

        Args:
            config (dict): Configuration dictionary with keys:
                - 'project' (str): GCP project ID
                - 'credentials_path' (Optional[str]): Path to service account JSON
        """
        self.project = config.get("project")
        credentials_path = config.get("credentials_path")
        try:
            if credentials_path:
                self.client = bigquery.Client.from_service_account_json(credentials_path, project=self.project)
            else:
                self.client = bigquery.Client(project=self.project)
            logger.info(f"Connected to BigQuery project '{self.project}'")
        except GoogleAPIError as e:
            logger.error(f"Failed to connect to BigQuery: {e}")
            raise

    # -------------------------
    # Query Execution
    # -------------------------
    def execute_query(self, query: str, job_config: Optional[bigquery.QueryJobConfig] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as Pandas DataFrame.

        Args:
            query (str): SQL query string.
            job_config (Optional[bigquery.QueryJobConfig]): Optional BigQuery job configuration.

        Returns:
            pd.DataFrame: Query results.
        """
        try:
            query_job = self.client.query(query, job_config=job_config)
            result = query_job.result().to_dataframe()
            logger.info(f"Query executed successfully: {query[:100]}...")
            return result
        except GoogleAPIError as e:
            logger.error(f"Query failed: {e}")
            raise

    # -------------------------
    # Table & Dataset Management
    # -------------------------
    def create_dataset(self, dataset_id: str, location: str = "US") -> None:
        """
        Create a BigQuery dataset.

        Args:
            dataset_id (str): Dataset ID to create.
            location (str): Dataset location, e.g., "US".
        """
        from google.cloud.bigquery import Dataset
        dataset_ref = self.client.dataset(dataset_id)
        dataset = Dataset(dataset_ref)
        dataset.location = location
        try:
            self.client.create_dataset(dataset, exists_ok=True)
            logger.info(f"Dataset '{dataset_id}' created in location '{location}'")
        except GoogleAPIError as e:
            logger.error(f"Failed to create dataset '{dataset_id}': {e}")
            raise

    def create_table(self, table_id: str, schema: list) -> None:
        """
        Create a BigQuery table.

        Args:
            table_id (str): Fully-qualified table ID: project.dataset.table
            schema (list): List of bigquery.SchemaField objects
        """
        from google.cloud.bigquery import Table
        table = Table(table_id, schema=schema)
        try:
            self.client.create_table(table, exists_ok=True)
            logger.info(f"Table '{table_id}' created successfully")
        except GoogleAPIError as e:
            logger.error(f"Failed to create table '{table_id}': {e}")
            raise

    # -------------------------
    # Bulk Data Load
    # -------------------------
    def load_dataframe(self, df: pd.DataFrame, table_id: str, write_disposition: str = "WRITE_APPEND") -> None:
        """
        Load a Pandas DataFrame into a BigQuery table.

        Args:
            df (pd.DataFrame): Data to load.
            table_id (str): Fully-qualified table ID.
            write_disposition (str): "WRITE_APPEND", "WRITE_TRUNCATE", or "WRITE_EMPTY"
        """
        job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
        try:
            load_job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
            load_job.result()
            logger.info(f"DataFrame loaded into table '{table_id}' ({len(df)} rows)")
        except GoogleAPIError as e:
            logger.error(f"Failed to load DataFrame into '{table_id}': {e}")
            raise

    # -------------------------
    # Export Data
    # -------------------------
    def export_table(self, table_id: str, destination_uri: str, format_: str = "CSV") -> None:
        """
        Export a table to cloud storage or local path.

        Args:
            table_id (str): Fully-qualified table ID.
            destination_uri (str): Destination URI, e.g., gs://bucket/file.csv
            format_ (str): Export format ("CSV", "PARQUET", "JSON")
        """
        from google.cloud.bigquery import ExtractJobConfig

        format_ = format_.upper()
        if format_ not in ["CSV", "PARQUET", "JSON"]:
            raise ValueError("Unsupported export format. Use CSV, PARQUET, or JSON.")

        job_config = ExtractJobConfig(destination_format=format_)
        try:
            extract_job = self.client.extract_table(table_id, destination_uri, job_config=job_config)
            extract_job.result()
            logger.info(f"Table '{table_id}' exported successfully to '{destination_uri}' as {format_}")
        except GoogleAPIError as e:
            logger.error(f"Failed to export table '{table_id}': {e}")
            raise
          
