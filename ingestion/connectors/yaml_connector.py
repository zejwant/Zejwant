# yaml_connector.py
"""
YAML Connector for enterprise-grade ingestion pipelines.

Features:
- Read YAML files
- Validate schema
- Logging and error handling
- Return Pandas DataFrame or dict
"""

from typing import Any, Dict, Optional, Union
import pandas as pd
import yaml
import logging
import os

# Logging setup
logger = logging.getLogger("yaml_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class YAMLConnector:
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize YAML Connector.

        Args:
            schema (Dict[str, Any], optional): Expected structure for validation
        """
        self.schema = schema

    def read(self, file_path: str, as_dataframe: bool = True) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Read a YAML file and optionally convert it to a Pandas DataFrame.

        Args:
            file_path (str): Path to the YAML file
            as_dataframe (bool): If True, return a DataFrame; otherwise, return dict

        Returns:
            Union[pd.DataFrame, Dict[str, Any]]: Parsed YAML data

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If schema validation fails
        """
        if not os.path.exists(file_path):
            logger.error(f"YAML file not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            logger.info(f"Reading YAML file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            self._validate_schema(data)

            if as_dataframe:
                # Convert nested YAML to flat DataFrame
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                else:
                    df = pd.json_normalize([data])
                logger.info(f"YAML file converted to DataFrame: {len(df)} rows")
                return df
            else:
                logger.info("YAML file loaded as dict")
                return data

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file '{file_path}': {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read YAML file '{file_path}': {e}")
            raise

    def _validate_schema(self, data: Any):
        """
        Validate YAML data against expected schema.

        Args:
            data (Any): Parsed YAML data

        Raises:
            ValueError: If schema validation fails
        """
        if not self.schema:
            return

        # Simple schema validation: check top-level keys
        if isinstance(data, dict):
            missing_keys = [k for k in self.schema.keys() if k not in data]
            if missing_keys:
                raise ValueError(f"Missing expected keys in YAML: {missing_keys}")
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                missing_keys = [k for k in self.schema.keys() if k not in item]
                if missing_keys:
                    raise ValueError(f"Missing expected keys in YAML item {idx}: {missing_keys}")
                  
