# sensor_iot_connector.py
"""
Sensor IoT Connector for enterprise-grade ingestion pipelines.

Features:
- Ingest IoT sensor data from MQTT, Kafka, or files (CSV/JSON)
- Handle time series data
- Data validation and cleaning
- Logging and error handling
- Returns Pandas DataFrame
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import logging
import os
import json
import re

# Logging setup
logger = logging.getLogger("sensor_iot_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class SensorIoTConnector:
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize Sensor IoT Connector.

        Args:
            schema (Dict[str, Any], optional): Expected schema for validation
        """
        self.schema = schema

    def read_file(self, file_path: str) -> pd.DataFrame:
        """
        Read sensor data from CSV or JSON file.

        Args:
            file_path (str): Path to the file

        Returns:
            pd.DataFrame: Structured sensor data
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Reading sensor file: {file_path}")
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext == ".json":
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            df = self._validate_and_clean(df)
            logger.info(f"Sensor data loaded from file: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to read sensor file '{file_path}': {e}")
            raise

    def process_stream(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process sensor data from streaming sources (MQTT/Kafka).

        Args:
            messages (List[Dict[str, Any]]): List of sensor messages

        Returns:
            pd.DataFrame: Structured sensor data
        """
        try:
            df = pd.json_normalize(messages)
            df = self._validate_and_clean(df)
            logger.info(f"Sensor data processed from stream: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to process sensor stream data: {e}")
            raise

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean sensor data.

        Args:
            df (pd.DataFrame): Raw sensor data

        Returns:
            pd.DataFrame: Cleaned sensor data
        """
        # Example cleaning: drop duplicates, fill missing timestamps
        df = df.drop_duplicates()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp")

        # Schema validation if schema provided
        if self.schema:
            missing_cols = [col for col in self.schema.keys() if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing expected columns: {missing_cols}")

        return df
      
