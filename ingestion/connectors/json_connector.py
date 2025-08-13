# json_connector.py
"""
JSON Connector for enterprise-grade ingestion pipelines.

Features:
- Ingest local JSON files or remote JSON APIs
- Automatically flatten nested structures
- Schema validation
- Logging and error handling
- Returns Pandas DataFrame
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import json
import logging
import os
import requests
from pandas import json_normalize

# Logging setup
logger = logging.getLogger("json_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class JSONConnector:
    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        flatten_sep: str = ".",
    ):
        """
        Initialize JSON Connector.

        Args:
            schema (Dict[str, Any], optional): Expected column types for validation
            flatten_sep (str): Separator for nested JSON flattening
        """
        self.schema = schema
        self.flatten_sep = flatten_sep

    def read(
        self,
        source: str,
        remote: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Read JSON data from local file or remote API.

        Args:
            source (str): File path or URL
            remote (bool): If True, source is an HTTP/HTTPS URL
            headers (Dict[str,str], optional): HTTP headers for remote request

        Returns:
            pd.DataFrame: Loaded and validated DataFrame

        Raises:
            FileNotFoundError: If local file does not exist
            ValueError: If schema validation fails
        """
        try:
            if remote:
                logger.info(f"Fetching JSON from remote source: {source}")
                resp = requests.get(source, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            else:
                if not os.path.exists(source):
                    raise FileNotFoundError(f"File not found: {source}")
                logger.info(f"Reading local JSON file: {source}")
                with open(source, "r", encoding="utf-8") as f:
                    data = json.load(f)

            df = self._normalize_json(data)
            df = self._validate_schema(df)

            logger.info(f"JSON loaded successfully: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to read JSON from '{source}': {e}")
            raise

    def _normalize_json(self, data: Union[List, Dict]) -> pd.DataFrame:
        """
        Flatten nested JSON structures into a Pandas DataFrame.

        Args:
            data (List or Dict): JSON data

        Returns:
            pd.DataFrame: Flattened DataFrame
        """
        if isinstance(data, dict):
            data = [data]

        df = json_normalize(data, sep=self.flatten_sep)
        return df

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
              
