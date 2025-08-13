"""
cleaning/cleaner_core.py

Enterprise-grade core data cleaning utilities for all formats.

Features:
- Text normalization: trimming, case normalization, whitespace, encoding, special characters
- Null/missing handling and type conversions
- Standardization, scaling, normalization
- Outlier detection
- Data enrichment stubs
- Logging and error handling
"""

import logging
import re
from typing import Any, List, Dict, Union, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Initialize module logger
logger = logging.getLogger("cleaner_core")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class CleanerCore:
    """Core reusable cleaning utilities for data pipelines."""

    @staticmethod
    def trim_strings(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Trim leading/trailing whitespace from string columns."""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        logger.info(f"Trimmed strings in columns: {columns}")
        return df

    @staticmethod
    def lowercase_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Convert string columns to lowercase."""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()
        logger.info(f"Lowercased columns: {columns}")
        return df

    @staticmethod
    def remove_extra_whitespace(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Replace multiple spaces with a single space."""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
        logger.info(f"Removed extra whitespace in columns: {columns}")
        return df

    @staticmethod
    def fix_encoding(df: pd.DataFrame, columns: List[str], encoding='utf-8') -> pd.DataFrame:
        """Fix text encoding issues."""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: x.encode('utf-8', errors='ignore').decode(encoding, errors='ignore'))
        logger.info(f"Fixed encoding in columns: {columns}")
        return df

    @staticmethod
    def remove_special_characters(df: pd.DataFrame, columns: List[str], pattern: str = r'[^a-zA-Z0-9\s]') -> pd.DataFrame:
        """Remove special characters from text columns."""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(pattern, '', regex=True)
        logger.info(f"Removed special characters from columns: {columns}")
        return df

    @staticmethod
    def fill_missing(df: pd.DataFrame, columns: List[str], value: Any = 0) -> pd.DataFrame:
        """Fill missing/null values with a specified value."""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna(value)
        logger.info(f"Filled missing values in columns: {columns} with {value}")
        return df

    @staticmethod
    def drop_missing(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Drop rows with missing values in specified columns or entire dataframe."""
        original_len = len(df)
        df = df.dropna(subset=columns) if columns else df.dropna()
        logger.info(f"Dropped {original_len - len(df)} rows with missing values in columns: {columns}")
        return df

    @staticmethod
    def convert_types(df: pd.DataFrame, type_map: Dict[str, Any]) -> pd.DataFrame:
        """Convert column types based on type map."""
        for col, dtype in type_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        logger.info(f"Converted column types: {type_map}")
        return df

    @staticmethod
    def standardize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Standardize numeric column to mean=0, std=1."""
        if column in df.columns:
            scaler = StandardScaler()
            df[column] = scaler.fit_transform(df[[column]])
        logger.info(f"Standardized column: {column}")
        return df

    @staticmethod
    def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Normalize numeric column to [0, 1]."""
        if column in df.columns:
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(df[[column]])
        logger.info(f"Normalized column: {column}")
        return df

    @staticmethod
    def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """Flag rows where values are outliers based on Z-score."""
        if column in df.columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df['outlier_' + column] = z_scores > threshold
        logger.info(f"Detected outliers in column: {column} using Z-score threshold {threshold}")
        return df

    @staticmethod
    def categorical_mapping(df: pd.DataFrame, column: str, mapping: Dict[Any, Any]) -> pd.DataFrame:
        """Map categorical values based on provided mapping dictionary."""
        if column in df.columns:
            df[column] = df[column].map(mapping).fillna(df[column])
        logger.info(f"Applied categorical mapping on column: {column}")
        return df

    @staticmethod
    def parse_dates(df: pd.DataFrame, column: str, date_format: Optional[str] = None) -> pd.DataFrame:
        """Parse string columns into datetime objects."""
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], format=date_format, errors='coerce')
        logger.info(f"Parsed dates in column: {column} with format: {date_format}")
        return df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows."""
        original_len = len(df)
        df = df.drop_duplicates(subset=subset)
        logger.info(f"Removed {original_len - len(df)} duplicate rows in subset: {subset}")
        return df

    @staticmethod
    def round_numeric(df: pd.DataFrame, columns: List[str], decimals: int = 2) -> pd.DataFrame:
        """Round numeric columns to specified decimals."""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].round(decimals)
        logger.info(f"Rounded columns {columns} to {decimals} decimals")
        return df

    @staticmethod
    def strip_prefix_suffix(df: pd.DataFrame, columns: List[str], prefix: str = '', suffix: str = '') -> pd.DataFrame:
        """Remove specified prefix/suffix from string columns."""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.removeprefix(prefix).str.removesuffix(suffix)
        logger.info(f"Stripped prefix '{prefix}' and suffix '{suffix}' from columns: {columns}")
        return df

    @staticmethod
    def replace_values(df: pd.DataFrame, column: str, replacements: Dict[Any, Any]) -> pd.DataFrame:
        """Replace specific values in a column."""
        if column in df.columns:
            df[column] = df[column].replace(replacements)
        logger.info(f"Replaced values in column {column}: {replacements}")
        return df

    @staticmethod
    def apply_custom_function(df: pd.DataFrame, column: str, func) -> pd.DataFrame:
        """Apply a custom function to a column."""
        if column in df.columns:
            df[column] = df[column].apply(func)
        logger.info(f"Applied custom function to column: {column}")
        return df

    @staticmethod
    def cast_to_string(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Ensure specified columns are of type string."""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        logger.info(f"Casted columns to string: {columns}")
        return df

    @staticmethod
    def enforce_unique(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Ensure values in column are unique by appending numeric suffixes."""
        if column in df.columns:
            counts = {}
            new_values = []
            for val in df[column]:
                if val not in counts:
                    counts[val] = 0
                    new_values.append(val)
                else:
                    counts[val] += 1
                    new_values.append(f"{val}_{counts[val]}")
            df[column] = new_values
        logger.info(f"Enforced unique values in column: {column}")
        return df

    @staticmethod
    def remove_zero_rows(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Drop rows where all specified columns are zero (or entire df if None)."""
        original_len = len(df)
        if columns:
            df = df[(df[columns] != 0).any(axis=1)]
        else:
            df = df[(df != 0).any(axis=1)]
        logger.info(f"Removed {original_len - len(df)} rows with all zeros in columns: {columns}")
        return df
      
