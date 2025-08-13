# avro_connector.py
"""
Avro Connector for enterprise-grade ingestion pipelines.

Features:
- Read Avro files efficiently
- Handle Avro schemas
- Logging and error handling
- Returns Pandas DataFrame
"""

from typing import Any, Dict, Optional
import pandas as pd
import logging
import os
import fastavro

# Logging setup
logger = logging.getLogger("avro_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class AvroConnector:
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize Avro Connector.

        Args:
            schema (Dict[str, Any], optional): Expected column types for validation
        """
        self.schema = schema

    def read(self, file_path: str) -> pd.DataFrame:
        """
        Read an Avro file and return a Pandas DataFrame.

        Args:
            file_path (str): Path to the Avro file

        Returns:
            pd.DataFrame: Loaded and validated DataFrame

        Raises:
            FileNotFoundError: If the Avro file does not exist
            ValueError: If schema validation fails
        """
        if not os.path.exists(file_path):
            logger.error(f"Avro file not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            logger.info(f"Reading Avro file: {file_path}")
            with open(file_path, "rb") as f:
                reader = fastavro.reader(f)
                records = [r for r in reader]

            df = pd.DataFrame(records)
            df = self._validate_schema(df)

            logger.info(f"Avro file loaded successfully: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to read Avro file '{file_path}': {e}")
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
      
