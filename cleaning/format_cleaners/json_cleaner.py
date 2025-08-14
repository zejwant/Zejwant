"""
json_cleaner.py
Enterprise-level JSON data cleaning module.

Capabilities:
- Flatten nested JSON structures
- Type validation and casting
- Missing value handling
- Regex cleaning for strings
- Date/time normalization
- Deduplication and normalization
- Conditional transformations
- Logging and error handling
- 20+ cleaning methods
- Returns cleaned Pandas DataFrame

Author: Varun Mode
"""

import pandas as pd
import numpy as np
import logging
import json
import re
from typing import Any, Dict, List, Optional, Union
from pandas import DataFrame

# Configure logger
logger = logging.getLogger("JSONCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class JSONCleaner:
    """
    Class to handle enterprise-level JSON cleaning.
    """

    def __init__(self, json_data: Union[str, Dict, List[Dict]]):
        """
        Initialize JSONCleaner with a JSON string, dict, or list of dicts.

        Args:
            json_data (Union[str, Dict, List[Dict]]): JSON input to clean.
        """
        self.raw_data = json_data
        self.df: Optional[DataFrame] = None
        self.cleaned_df: Optional[DataFrame] = None
        self._load_json()

    def _load_json(self) -> None:
        """Load JSON data into a flattened DataFrame."""
        try:
            if isinstance(self.raw_data, str):
                self.raw_data = json.loads(self.raw_data)
            # Flatten nested JSON into dataframe
            self.df = pd.json_normalize(self.raw_data, sep='_')
            logger.info(f"Loaded JSON with {len(self.df)} records and {len(self.df.columns)} columns.")
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            raise

    # ---------------- Cleaning Methods ----------------

    @staticmethod
    def fill_missing(df: DataFrame, fill_value: Union[str, int, float] = "") -> DataFrame:
        """Fill missing values with specified value."""
        return df.fillna(fill_value)

    @staticmethod
    def drop_missing(df: DataFrame, axis: int = 0, thresh: Optional[int] = None) -> DataFrame:
        """Drop rows or columns with missing values below threshold."""
        return df.dropna(axis=axis, thresh=thresh)

    @staticmethod
    def strip_whitespace(df: DataFrame) -> DataFrame:
        """Trim whitespace from string columns."""
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
        return df

    @staticmethod
    def lowercase_strings(df: DataFrame) -> DataFrame:
        """Lowercase string columns."""
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.lower())
        return df

    @staticmethod
    def remove_special_characters(df: DataFrame, pattern: str = r'[^0-9a-zA-Z\s]') -> DataFrame:
        """Remove special characters from string columns."""
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.replace(pattern, '', regex=True))
        return df

    @staticmethod
    def auto_cast_numeric(df: DataFrame) -> DataFrame:
        """Cast numeric-looking columns to numeric types."""
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return df

    @staticmethod
    def parse_dates(df: DataFrame, columns: List[str]) -> DataFrame:
        """Parse specified columns as datetime."""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    @staticmethod
    def remove_duplicates(df: DataFrame, subset: Optional[List[str]] = None) -> DataFrame:
        """Remove duplicate rows."""
        return df.drop_duplicates(subset=subset)

    @staticmethod
    def normalize_column(df: DataFrame, column: str) -> DataFrame:
        """Normalize a numeric column between 0 and 1."""
        if column in df.columns:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        return df

    @staticmethod
    def standardize_column(df: DataFrame, column: str) -> DataFrame:
        """Standardize a numeric column (z-score)."""
        if column in df.columns:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
        return df

    @staticmethod
    def remove_outliers(df: DataFrame, z_thresh: float = 3.0) -> DataFrame:
        """Remove outliers based on z-score for numeric columns."""
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            z_scores = (df[col] - df[col].mean()) / df[col].std(ddof=0)
            df = df[z_scores.abs() <= z_thresh]
        return df

    @staticmethod
    def regex_replace(df: DataFrame, column: str, pattern: str, replacement: str = "") -> DataFrame:
        """Apply regex replacement to a column."""
        if column in df.columns:
            df[column] = df[column].astype(str).str.replace(pattern, replacement, regex=True)
        return df

    @staticmethod
    def rename_columns(df: DataFrame, columns_map: Dict[str, str]) -> DataFrame:
        """Rename columns using a dictionary."""
        return df.rename(columns=columns_map)

    @staticmethod
    def conditional_transform(df: DataFrame) -> DataFrame:
        """Example conditional transformation."""
        if 'status' in df.columns:
            df['status'] = df['status'].str.upper()
        return df

    @staticmethod
    def encode_categorical(df: DataFrame, columns: List[str]) -> DataFrame:
        """One-hot encode categorical columns."""
        return pd.get_dummies(df, columns=columns, drop_first=True)

    @staticmethod
    def replace_values(df: DataFrame, replace_map: Dict[str, Any]) -> DataFrame:
        """Replace values based on a dictionary."""
        return df.replace(replace_map)

    @staticmethod
    def sort_by_column(df: DataFrame, column: str, ascending: bool = True) -> DataFrame:
        """Sort dataframe by a column."""
        if column in df.columns:
            df = df.sort_values(by=column, ascending=ascending)
        return df

    @staticmethod
    def flatten_nested(df: DataFrame) -> DataFrame:
        """Ensure nested dicts/lists are flattened (already normalized)."""
        # Placeholder: json_normalize already flattens nested structures
        return df

    # ---------------- Main Cleaning Pipeline ----------------

    def clean(self,
              date_columns: Optional[List[str]] = None,
              categorical_columns: Optional[List[str]] = None,
              regex_columns: Optional[Dict[str, str]] = None,
              replace_map: Optional[Dict[str, Any]] = None,
              columns_map: Optional[Dict[str, str]] = None) -> DataFrame:
        """
        Apply all cleaning steps to the JSON dataframe.

        Args:
            date_columns (List[str], optional): Columns to parse as dates.
            categorical_columns (List[str], optional): Columns to one-hot encode.
            regex_columns (Dict[str, str], optional): Columns with regex patterns to clean.
            replace_map (Dict[str, Any], optional): Values to replace.
            columns_map (Dict[str, str], optional): Column renaming map.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        df = self.df.copy()
        try:
            df = self.flatten_nested(df)
            df = self.fill_missing(df)
            df = self.strip_whitespace(df)
            df = self.lowercase_strings(df)
            df = self.remove_special_characters(df)
            df = self.auto_cast_numeric(df)
            df = self.remove_duplicates(df)
            df = self.remove_outliers(df)
            df = self.conditional_transform(df)

            if date_columns:
                df = self.parse_dates(df, date_columns)
            if categorical_columns:
                df = self.encode_categorical(df, categorical_columns)
            if regex_columns:
                for col, pattern in regex_columns.items():
                    df = self.regex_replace(df, col, pattern)
            if replace_map:
                df = self.replace_values(df, replace_map)
            if columns_map:
                df = self.rename_columns(df, columns_map)

            self.cleaned_df = df
            logger.info(f"JSON cleaning complete: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
        except Exception as e:
            logger.error(f"Error cleaning JSON: {e}")
            raise

# ------------------ Usage Example ------------------
if __name__ == "__main__":
    sample_json = [
        {"id": 1, "name": " Alice ", "status": "active", "score": "100"},
        {"id": 2, "name": "Bob", "status": "inactive", "score": None},
        {"id": 3, "name": "Charlie", "status": "Active", "score": "200"}
    ]

    cleaner = JSONCleaner(sample_json)
    cleaned_df = cleaner.clean(date_columns=[], categorical_columns=['status'])
    print(cleaned_df)
      
