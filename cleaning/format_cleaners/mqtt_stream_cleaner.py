"""
mqtt_stream_cleaner.py
Enterprise-level MQTT IoT stream data cleaning module.

Capabilities:
- Real-time IoT stream data cleaning
- Null/missing value handling
- Type normalization and casting
- Deduplication
- Timestamp alignment
- Outlier detection
- Conditional transformations
- String normalization and regex cleaning
- Logging, metrics, and error handling
- 20+ modular cleaning methods
- Returns cleaned structured DataFrame

Author: Varun Mode
"""

import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

# ---------------- Logger Setup ----------------
logger = logging.getLogger("MQTTStreamCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class MQTTStreamCleaner:
    """
    Class to clean real-time IoT MQTT streaming data.
    """

    def __init__(self, df: pd.DataFrame, timestamp_col: str):
        """
        Initialize with incoming IoT stream batch.

        Args:
            df (pd.DataFrame): Incoming MQTT batch data
            timestamp_col (str): Name of timestamp column
        """
        self.df = df.copy()
        self.timestamp_col = timestamp_col
        self.cleaned_df: Optional[pd.DataFrame] = None

    # ---------------- Helper Cleaning Methods ----------------
    def align_timestamp(self, freq: str = '1min') -> pd.DataFrame:
        """Normalize and align timestamps to a fixed frequency."""
        self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col], errors='coerce')
        self.df = self.df.set_index(self.timestamp_col).sort_index()
        self.df = self.df.resample(freq).mean()
        logger.info(f"Timestamps aligned to {freq} frequency.")
        return self.df

    def fill_missing(self, method: str = 'linear') -> pd.DataFrame:
        """Interpolate missing numeric values."""
        numeric_cols = self.df.select_dtypes(include='number').columns
        self.df[numeric_cols] = self.df[numeric_cols].interpolate(method=method)
        logger.info(f"Missing numeric values interpolated using {method}.")
        return self.df

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        self.df = self.df.drop_duplicates(subset=subset)
        logger.info("Duplicates removed.")
        return self.df

    def remove_outliers(self, z_thresh: float = 3.0) -> pd.DataFrame:
        numeric_cols = self.df.select_dtypes(include='number').columns
        for col in numeric_cols:
            z_scores = (self.df[col] - self.df[col].mean()) / self.df[col].std(ddof=0)
            self.df = self.df[z_scores.abs() <= z_thresh]
        logger.info(f"Outliers removed with Z-threshold {z_thresh}.")
        return self.df

    def trim_strings(self) -> pd.DataFrame:
        str_cols = self.df.select_dtypes(include='object').columns
        self.df[str_cols] = self.df[str_cols].apply(lambda x: x.str.strip())
        return self.df

    def lowercase_strings(self) -> pd.DataFrame:
        str_cols = self.df.select_dtypes(include='object').columns
        self.df[str_cols] = self.df[str_cols].apply(lambda x: x.str.lower())
        return self.df

    def uppercase_strings(self) -> pd.DataFrame:
        str_cols = self.df.select_dtypes(include='object').columns
        self.df[str_cols] = self.df[str_cols].apply(lambda x: x.str.upper())
        return self.df

    def remove_special_characters(self, pattern: str = r'[^0-9a-zA-Z\s]') -> pd.DataFrame:
        str_cols = self.df.select_dtypes(include='object').columns
        self.df[str_cols] = self.df[str_cols].apply(lambda x: x.str.replace(pattern, '', regex=True))
        return self.df

    def auto_cast_numeric(self) -> pd.DataFrame:
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
        return self.df

    def regex_replace(self, regex_map: Dict[str, str]) -> pd.DataFrame:
        for col, pattern in regex_map.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.replace(pattern, '', regex=True)
        return self.df

    def conditional_transform(self, col: str, func: Callable) -> pd.DataFrame:
        if col in self.df.columns:
            self.df[col] = self.df[col].apply(func)
        return self.df

    def scale_numeric(self, scale_map: Dict[str, float]) -> pd.DataFrame:
        for col, factor in scale_map.items():
            if col in self.df.columns:
                self.df[col] = self.df[col] * factor
        return self.df

    def smooth_data(self, window: int = 3) -> pd.DataFrame:
        numeric_cols = self.df.select_dtypes(include='number').columns
        self.df[numeric_cols] = self.df[numeric_cols].rolling(window=window, min_periods=1).mean()
        return self.df

    def encode_categorical(self, columns: List[str]) -> pd.DataFrame:
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)
        return self.df

    def sort_by_timestamp(self) -> pd.DataFrame:
        self.df = self.df.sort_index()
        return self.df

    def validate_columns(self, required_columns: List[str]) -> pd.DataFrame:
        for col in required_columns:
            if col not in self.df.columns:
                self.df[col] = np.nan
        return self.df

    def conditional_numeric_transform(self, transform_map: Dict[str, Callable]) -> pd.DataFrame:
        for col, func in transform_map.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(lambda x: func(x) if pd.notnull(x) else x)
        return self.df

    def drop_empty_rows(self) -> pd.DataFrame:
        self.df = self.df.dropna(how='all')
        return self.df

    def limit_rows(self, max_rows: int) -> pd.DataFrame:
        self.df = self.df.head(max_rows)
        return self.df

    # ---------------- Main Cleaning Pipeline ----------------
    def clean(self,
              freq: str = '1min',
              interpolate_method: str = 'linear',
              z_thresh: float = 3.0,
              smooth_window: int = 3,
              regex_map: Optional[Dict[str, str]] = None,
              categorical_cols: Optional[List[str]] = None,
              scale_map: Optional[Dict[str, float]] = None,
              numeric_transform_map: Optional[Dict[str, Callable]] = None,
              required_columns: Optional[List[str]] = None,
              max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Full MQTT IoT stream cleaning pipeline with 20+ steps.
        """
        try:
            self.align_timestamp(freq=freq)
            self.fill_missing(method=interpolate_method)
            self.remove_duplicates()
            self.remove_outliers(z_thresh=z_thresh)
            self.auto_cast_numeric()
            self.trim_strings()
            self.lowercase_strings()
            self.uppercase_strings()
            self.remove_special_characters()
            if regex_map:
                self.regex_replace(regex_map)
            if categorical_cols:
                self.encode_categorical(categorical_cols)
            if scale_map:
                self.scale_numeric(scale_map)
            if numeric_transform_map:
                self.conditional_numeric_transform(numeric_transform_map)
            if required_columns:
                self.validate_columns(required_columns)
            self.smooth_data(window=smooth_window)
            self.sort_by_timestamp()
            self.drop_empty_rows()
            if max_rows:
                self.limit_rows(max_rows)

            self.cleaned_df = self.df
            logger.info(f"MQTT stream data cleaned: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
            return self.df

        except Exception as e:
            logger.error(f"Error cleaning MQTT stream data: {e}")
            raise


# ------------------ Usage Example ------------------
if __name__ == "__main__":
    sample_data = pd.DataFrame({
        "timestamp": ["2025-08-14 10:00", "2025-08-14 10:01", "2025-08-14 10:02", "2025-08-14 10:03"],
        "temperature": [22.5, 22.7, 22.6, 23.0],
        "humidity": [55, 54, None, 57],
        "status": ["OK", "OK", "ALERT", "OK"]
    })
    cleaner = MQTTStreamCleaner(sample_data, timestamp_col="timestamp")
    cleaned_df = cleaner.clean(
        regex_map={"status": r'[^a-zA-Z]'},
        categorical_cols=["status"],
        scale_map={"temperature": 1.0},
        numeric_transform_map={"humidity": lambda x: x / 100},
        required_columns=["temperature", "humidity", "status"]
    )
    print(cleaned_df)
      
