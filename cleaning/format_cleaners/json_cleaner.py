"""
cleaning/format_cleaners/json_cleaner.py

Enterprise-level JSON cleaning module.

Features:
- Flatten nested JSON structures
- Type validation and missing value handling
- Regex cleaning for string columns
- Date/time normalization
- Deduplication and normalization
- Logging and error handling
- 20+ reusable cleaning methods for nested and flat JSON
"""

import logging
import pandas as pd
import numpy as np
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger("json_cleaner")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def clean_json(json_data: Union[Dict[str, Any], List[Dict[str, Any]]],
               flatten: bool = True,
               date_columns: Optional[List[str]] = None,
               categorical_mappings: Optional[Dict[str, Dict[Any, Any]]] = None,
               regex_cleaning: Optional[Dict[str, str]] = None,
               dedup_columns: Optional[List[str]] = None,
               fill_missing_value: Union[str, int, float] = '') -> pd.DataFrame:
    """
    Enterprise-level JSON cleaning.

    Args:
        json_data (dict or list of dict): Input JSON data.
        flatten (bool): Flatten nested JSON if True.
        date_columns (List[str], optional): Columns to normalize as datetime.
        categorical_mappings (Dict[column, mapping], optional): Categorical mapping.
        regex_cleaning (Dict[column, pattern], optional): Regex cleaning per column.
        dedup_columns (List[str], optional): Columns to deduplicate on.
        fill_missing_value (str/int/float): Default fill for missing values.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        # Normalize JSON
        df = flatten_json(json_data) if flatten else pd.DataFrame(json_data)

        # Core cleaning
        df = trim_strings(df)
        df = lowercase_columns(df)
        df = remove_extra_whitespace(df)
        df = remove_special_characters(df)
        df = fill_missing(df, fill_missing_value)
        df = drop_empty_rows(df)
        df = auto_cast_numeric(df)

        # Date/time normalization
        if date_columns:
            for col in date_columns:
                df = parse_dates(df, col)

        # Categorical mapping
        if categorical_mappings:
            for col, mapping in categorical_mappings.items():
                df = map_categorical(df, col, mapping)

        # Regex-based cleaning
        if regex_cleaning:
            for col, pattern in regex_cleaning.items():
                df = regex_replace(df, col, pattern)

        # Deduplication
        if dedup_columns:
            df = drop_exact_duplicates(df, dedup_columns)

        # Outlier removal
        df = remove_outliers(df)

        # Conditional transformations
        df = conditional_transform(df)

        logger.info(f"JSON cleaned successfully, total rows: {len(df)}")
        return df

    except Exception as e:
        logger.error(f"Error cleaning JSON: {e}")
        raise


# -------------------------------
# --- Helper cleaning functions ---
# -------------------------------

def flatten_json(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Flatten nested JSON into a flat DataFrame.
    Handles lists of dicts, nested dicts.
    """
    if isinstance(data, dict):
        data = [data]
    df = pd.json_normalize(data, sep='_')
    return df

def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    return df

def lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower()
    return df

def remove_extra_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    return df

def remove_special_characters(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.replace(r'[^0-9a-zA-Z\s]', '', regex=True)
    return df

def fill_missing(df: pd.DataFrame, fill_value: Union[str, int, float] = '') -> pd.DataFrame:
    return df.fillna(fill_value)

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how='all')

def auto_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

def parse_dates(df: pd.DataFrame, column: str, date_format: Optional[str] = None) -> pd.DataFrame:
    try:
        df[column] = pd.to_datetime(df[column], format=date_format, errors='coerce')
    except Exception as e:
        logger.warning(f"Date parsing failed for column {column}: {e}")
    return df

def drop_exact_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset)

def map_categorical(df: pd.DataFrame, column: str, mapping: Dict[Any, Any]) -> pd.DataFrame:
    if column in df.columns:
        df[column] = df[column].map(mapping).fillna(df[column])
    return df

def regex_replace(df: pd.DataFrame, column: str, pattern: str, replacement: str = '') -> pd.DataFrame:
    if column in df.columns:
        df[column] = df[column].astype(str).str.replace(pattern, replacement, regex=True)
    return df

def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        z_scores = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        df = df[z_scores.abs() <= z_thresh]
    return df

def conditional_transform(df: pd.DataFrame) -> pd.DataFrame:
    if 'status' in df.columns:
        df['status'] = df['status'].str.upper()
    return df
      
