# parquet_connector.py
"""
Parquet Connector for enterprise-grade ingestion pipelines.

Features:
- Read Parquet files efficiently
- Handle partitioned datasets
- Schema validation
- Logging and error handling
- Returns Pandas DataFrame
"""

from typing import Any, Dict, Optional
import pandas as pd
import logging
import os
import pyarrow.parquet as pq

# Logging setup
logger = logging.getLogger("parquet_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class ParquetConnector:
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize Parquet Connector.

        Args:
            schema (Dict[str, Any], optional): Expected column types for validation
        """
        self.schema = schema

    def read(self, path: str) -> pd.DataFrame:
        """
        Read a Parquet file or partitioned dataset into a Pandas DataFrame.

        Args:
            path (str): Path to Parquet file or directory containing partitioned Parquet files

        Returns:
            pd.DataFrame: Loaded and validated DataFrame

        Raises:
            FileNotFoundError: If path does not exist
            ValueError: If schema validation fails
        """
        if not os.path.exists(path):
            logger.error(f"Parquet path does not exist: {path}")
            raise FileNotFoundError(f"Path not found: {path}")

        try:
            logger.info(f"Reading Parquet data from: {path}")
            df = pd.read_parquet(path, engine="pyarrow")

            df = self._validate_schema(df)

            logger.info(f"Parquet data loaded successfully: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to read Parquet data from '{path}': {e}")
            raise

    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate DataFrame schema against expected schema.

        Args:
            df (pd.DataFrame): DataFrame to validate

        Returns:
            pd.DataFrame: Validated DataFrame

        Raises:
            ValueError: If schema validation fails
        """
        if not self.schema:
            return df

        for col, dtype in self.schema.items():
            if col not in df.columns:
                raise ValueError(f"Missing expected column: {col}")
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                raise ValueError(f"Failed to cast column '{col}' to {dtype}: {e}")
        return df
      
