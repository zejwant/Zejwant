"""
cleaning/format_cleaners/csv_cleaner.py

Enterprise-level CSV cleaning module.

Features:
- Core string cleaning: trim, case normalization, whitespace removal
- Null/missing value handling, type casting
- Deduplication, outlier detection
- Date parsing, categorical mapping
- Regex-based column cleaning
- Special character removal, encoding fixes
- Column renaming/standardization
- Conditional transformations
- Chunked processing for large CSVs
"""

import logging
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional, Union
from datetime import datetime

logger = logging.getLogger("csv_cleaner")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def clean_csv(file_path: str,
              chunksize: int = 100000,
              encoding: str = 'utf-8',
              date_columns: Optional[List[str]] = None,
              categorical_mappings: Optional[Dict[str, Dict[Any, Any]]] = None,
              column_renames: Optional[Dict[str, str]] = None,
              regex_cleaning: Optional[Dict[str, str]] = None,
              dedup_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Enterprise-level CSV cleaning.

    Args:
        file_path (str): Path to CSV file.
        chunksize (int): Rows per chunk for large CSVs.
        encoding (str): File encoding.
        date_columns (List[str], optional): Columns to parse as dates.
        categorical_mappings (Dict[str, Dict[Any, Any]], optional): Map values for categorical columns.
        column_renames (Dict[str, str], optional): Rename columns.
        regex_cleaning (Dict[str, str], optional): Regex replacement per column.
        dedup_columns (List[str], optional): Columns to deduplicate on.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_chunks = []

    try:
        for chunk in pd.read_csv(file_path, chunksize=chunksize, encoding=encoding):
            df = chunk.copy()

            # --- Core string cleaning ---
            df = trim_strings(df)
            df = lowercase_columns(df)
            df = remove_extra_whitespace(df)
            df = remove_special_characters(df)

            # --- Null/missing handling ---
            df = fill_missing(df)
            df = drop_empty_rows(df)

            # --- Type casting ---
            if date_columns:
                for col in date_columns:
                    df = parse_dates(df, col)
            df = auto_cast_numeric(df)

            # --- Deduplication ---
            if dedup_columns:
                df = drop_exact_duplicates(df, subset=dedup_columns)

            # --- Categorical mapping ---
            if categorical_mappings:
                for col, mapping in categorical_mappings.items():
                    df = map_categorical(df, col, mapping)

            # --- Regex-based cleaning ---
            if regex_cleaning:
                for col, pattern in regex_cleaning.items():
                    df = regex_replace(df, col, pattern)

            # --- Column renaming/standardization ---
            if column_renames:
                df.rename(columns=column_renames, inplace=True)

            # --- Outlier detection / removal ---
            df = remove_outliers(df)

            # --- Conditional transformations (example stub) ---
            df = conditional_transform(df)

            cleaned_chunks.append(df)
            logger.info(f"Processed chunk with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error reading/cleaning CSV: {e}")
        raise

    cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
    logger.info(f"CSV cleaned successfully, total rows: {len(cleaned_df)}")
    return cleaned_df


# -------------------------------
# --- Helper cleaning functions ---
# -------------------------------

def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading/trailing whitespace from string columns."""
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    return df

def lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase all string columns."""
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower()
    return df

def remove_extra_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extra spaces inside strings."""
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    return df

def remove_special_characters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-alphanumeric characters from strings."""
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.replace(r'[^0-9a-zA-Z\s]', '', regex=True)
    return df

def fill_missing(df: pd.DataFrame, fill_value: Union[str, int, float] = '') -> pd.DataFrame:
    """Fill missing values with a default."""
    return df.fillna(fill_value)

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where all columns are empty."""
    return df.dropna(how='all')

def parse_dates(df: pd.DataFrame, column: str, date_format: Optional[str] = None) -> pd.DataFrame:
    """Parse date column into datetime."""
    try:
        df[column] = pd.to_datetime(df[column], format=date_format, errors='coerce')
    except Exception as e:
        logger.warning(f"Date parsing failed for column {column}: {e}")
    return df

def auto_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically convert numeric columns."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

def drop_exact_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Drop exact duplicates in DataFrame."""
    return df.drop_duplicates(subset=subset)

def map_categorical(df: pd.DataFrame, column: str, mapping: Dict[Any, Any]) -> pd.DataFrame:
    """Map categorical column values."""
    if column in df.columns:
        df[column] = df[column].map(mapping).fillna(df[column])
    return df

def regex_replace(df: pd.DataFrame, column: str, pattern: str, replacement: str = '') -> pd.DataFrame:
    """Apply regex replacement on column."""
    if column in df.columns:
        df[column] = df[column].astype(str).str.replace(pattern, replacement, regex=True)
    return df

def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """Remove outliers from numeric columns using z-score."""
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        z_scores = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        df = df[z_scores.abs() <= z_thresh]
    return df

def conditional_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example stub for conditional transformation:
    Apply custom rules depending on column values.
    """
    # Example: uppercase 'status' column if present
    if 'status' in df.columns:
        df['status'] = df['status'].str.upper()
    return df
  
