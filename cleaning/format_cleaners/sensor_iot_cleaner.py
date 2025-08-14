"""
sensor_iot_cleaner.py
Enterprise-level IoT sensor data cleaning module.

Capabilities:
- Handle time series IoT data
- Null/missing value interpolation
- Outlier detection and removal
- Deduplication
- Timestamp normalization
- Data type normalization
- Scaling and smoothing
- Logging and error handling
- 20+ cleaning methods
- Returns cleaned Pandas DataFrame

Author: Varun Mode
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any

# Configure logger
logger = logging.getLogger("SensorIoTCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class SensorIoTCleaner:
    """
    Class to clean IoT sensor data for enterprise pipelines.
    """

    def __init__(self, df: pd.DataFrame, timestamp_col: str):
        """
        Initialize with IoT DataFrame.

        Args:
            df (pd.DataFrame): Raw IoT sensor data.
            timestamp_col (str): Name of the timestamp column.
        """
        self.df = df.copy()
        self.timestamp_col = timestamp_col
        self.cleaned_df: Optional[pd.DataFrame] = None

    # ---------------- Core Cleaning Methods ----------------
    def normalize_timestamp(self, freq: str = '1min') -> pd.DataFrame:
        """
        Normalize timestamps to regular intervals and set as index.
        """
        try:
            self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col], errors='coerce')
            self.df = self.df.set_index(self.timestamp_col).sort_index()
            self.df = self.df.resample(freq).mean()  # Fill gaps with NaN
            logger.info(f"Timestamps normalized with frequency '{freq}'.")
            return self.df
        except Exception as e:
            logger.error(f"Timestamp normalization failed: {e}")
            raise

    def interpolate_missing(self, method: str = 'linear') -> pd.DataFrame:
        """Interpolate missing values in numeric columns."""
        numeric_cols = self.df.select_dtypes(include='number').columns
        self.df[numeric_cols] = self.df[numeric_cols].interpolate(method=method)
        logger.info(f"Missing numeric values interpolated using method '{method}'.")
        return self.df

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows."""
        self.df = self.df.drop_duplicates(subset=subset)
        logger.info("Duplicates removed.")
        return self.df

    def remove_outliers(self, z_thresh: float = 3.0) -> pd.DataFrame:
        """Remove outliers in numeric columns using Z-score."""
        numeric_cols = self.df.select_dtypes(include='number').columns
        for col in numeric_cols:
            z_scores = (self.df[col] - self.df[col].mean()) / self.df[col].std(ddof=0)
            self.df = self.df[z_scores.abs() <= z_thresh]
        logger.info(f"Outliers removed using Z-threshold {z_thresh}.")
        return self.df

    @staticmethod
    def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
        return df

    @staticmethod
    def lowercase_strings(df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.lower())
        return df

    @staticmethod
    def remove_special_characters(df: pd.DataFrame, pattern: str = r'[^0-9a-zA-Z\s]') -> pd.DataFrame:
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.replace(pattern, '', regex=True))
        return df

    def auto_cast_numeric(self) -> pd.DataFrame:
        """Cast numeric columns."""
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
        return self.df

    def smooth_data(self, window: int = 3) -> pd.DataFrame:
        """Apply rolling mean smoothing to numeric columns."""
        numeric_cols = self.df.select_dtypes(include='number').columns
        self.df[numeric_cols] = self.df[numeric_cols].rolling(window=window, min_periods=1).mean()
        logger.info(f"Data smoothed with rolling window of {window}.")
        return self.df

    def replace_values(self, replace_map: Dict[str, Any]) -> pd.DataFrame:
        """Replace values based on a mapping."""
        self.df = self.df.replace(replace_map)
        return self.df

    def encode_categorical(self, columns: List[str]) -> pd.DataFrame:
        """Encode categorical columns as dummies."""
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)
        return self.df

    def sort_by_timestamp(self) -> pd.DataFrame:
        """Ensure data is sorted by timestamp index."""
        self.df = self.df.sort_index()
        return self.df

    # ---------------- Main Cleaning Pipeline ----------------
    def clean(self,
              freq: str = '1min',
              interpolate_method: str = 'linear',
              z_thresh: float = 3.0,
              smooth_window: int = 3,
              replace_map: Optional[Dict[str, Any]] = None,
              categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply full cleaning pipeline for IoT sensor data.

        Returns:
            pd.DataFrame: Cleaned DataFrame with timestamp index.
        """
        try:
            self.normalize_timestamp(freq=freq)
            self.interpolate_missing(method=interpolate_method)
            self.remove_duplicates()
            self.remove_outliers(z_thresh=z_thresh)
            self.auto_cast_numeric()
            self.smooth_data(window=smooth_window)
            self.trim_strings(self.df)
            self.lowercase_strings(self.df)
            self.remove_special_characters(self.df)
            if replace_map:
                self.replace_values(replace_map)
            if categorical_cols:
                self.encode_categorical(categorical_cols)
            self.sort_by_timestamp()

            self.cleaned_df = self.df
            logger.info(f"IoT sensor data cleaning complete: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
            return self.df
        except Exception as e:
            logger.error(f"Error cleaning IoT sensor data: {e}")
            raise


# ------------------ Usage Example ------------------
if __name__ == "__main__":
    # Example: sample IoT data
    sample_data = pd.DataFrame({
        "timestamp": ["2025-08-14 10:00", "2025-08-14 10:01", "2025-08-14 10:03", "2025-08-14 10:03"],
        "temperature": [22.5, 22.7, 23.0, 23.0],
        "humidity": [55, 54, np.nan, 54]
    })
    cleaner = SensorIoTCleaner(sample_data, timestamp_col="timestamp")
    cleaned_df = cleaner.clean()
    print(cleaned_df.head())
                        
