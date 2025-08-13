# csv_connector.py
"""
CSV Connector for enterprise-grade ingestion pipelines.

Features:
- Read from local and remote sources (HTTP, S3, local)
- Chunked reading for large files
- Schema validation
- Data cleaning (drop duplicates, handle missing values)
- Logging and error handling
- Returns Pandas DataFrame
"""

from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd
import logging
import os
import io
import requests

# Logging setup
logger = logging.getLogger("csv_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class CSVConnector:
    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        chunk_size: int = 100000,
        clean_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize CSV Connector.

        Args:
            schema (Dict[str, Any], optional): Expected column types for validation
            chunk_size (int): Number of rows per chunk for large files
            clean_options (Dict[str, Any], optional): Cleaning options, e.g.,
                {"drop_duplicates": True, "fillna": {"column": value}}
        """
        self.schema = schema
        self.chunk_size = chunk_size
        self.clean_options = clean_options or {}

    def read(
        self,
        source: str,
        remote: bool = False,
        encoding: str = "utf-8",
        delimiter: str = ",",
    ) -> pd.DataFrame:
        """
        Read CSV file from local or remote source.

        Args:
            source (str): File path or URL
            remote (bool): If True, read from HTTP/HTTPS URL
            encoding (str): File encoding
            delimiter (str): CSV delimiter

        Returns:
            pd.DataFrame: Loaded and validated data
        """
        df_list: List[pd.DataFrame] = []

        try:
            if remote:
                logger.info(f"Downloading CSV from remote source: {source}")
                resp = requests.get(source, stream=True)
                resp.raise_for_status()
                buffer = io.StringIO(resp.text)
                csv_iter = pd.read_csv(
                    buffer,
                    chunksize=self.chunk_size,
                    delimiter=delimiter,
                    encoding=encoding,
                )
            else:
                logger.info(f"Reading local CSV file: {source}")
                if not os.path.exists(source):
                    raise FileNotFoundError(f"File not found: {source}")
                csv_iter = pd.read_csv(
                    source,
                    chunksize=self.chunk_size,
                    delimiter=delimiter,
                    encoding=encoding,
                )

            for chunk in csv_iter:
                chunk = self._validate_schema(chunk)
                chunk = self._clean_data(chunk)
                df_list.append(chunk)

            df = pd.concat(df_list, ignore_index=True)
            logger.info(f"CSV loaded successfully: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to read CSV '{source}': {e}")
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

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data cleaning options.

        Args:
            df (pd.DataFrame): DataFrame to clean

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if self.clean_options.get("drop_duplicates", True):
            df = df.drop_duplicates()

        fillna_dict = self.clean_options.get("fillna", {})
        if fillna_dict:
            df = df.fillna(fillna_dict)

        # Additional cleaning options can be added here
        return df
      
