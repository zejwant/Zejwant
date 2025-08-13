"""
cleaning/format_cleaners/excel_cleaner.py

Enterprise-level Excel cleaning module.

Features:
- Handle multiple sheets, merged cells, and formulas
- Sheet-specific cleaning
- Core string cleaning: trim, case normalization, whitespace removal
- Null/missing value handling, type casting
- Deduplication, outlier detection
- Date parsing, categorical mapping
- Regex-based cleaning
- Column renaming/standardization
- Conditional transformations
- Logging and error handling
"""

import logging
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
import re

logger = logging.getLogger("excel_cleaner")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def clean_excel(file_path: str,
                sheet_names: Optional[List[str]] = None,
                chunksize: int = 100000,
                date_columns: Optional[Dict[str, List[str]]] = None,
                categorical_mappings: Optional[Dict[str, Dict[str, Dict[Union[str, int], Union[str, int]]]]] = None,
                column_renames: Optional[Dict[str, Dict[str, str]]] = None,
                regex_cleaning: Optional[Dict[str, Dict[str, str]]] = None,
                dedup_columns: Optional[Dict[str, List[str]]] = None,
                formulas_to_values: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Enterprise-level Excel cleaning.

    Args:
        file_path (str): Path to Excel file.
        sheet_names (List[str], optional): Specific sheets to clean; default all sheets.
        chunksize (int): Rows per chunk for large sheets (pandas handles large sheets natively).
        date_columns (Dict[sheet_name, List[column_name]], optional): Columns to parse as dates per sheet.
        categorical_mappings (Dict[sheet_name, Dict[column, mapping]], optional): Categorical mapping per sheet.
        column_renames (Dict[sheet_name, Dict[old_name, new_name]], optional): Column renames per sheet.
        regex_cleaning (Dict[sheet_name, Dict[column, pattern]], optional): Regex cleaning per sheet.
        dedup_columns (Dict[sheet_name, List[column]], optional): Columns to deduplicate per sheet.
        formulas_to_values (bool): Convert formulas to values if True.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of cleaned DataFrames keyed by sheet name.
    """
    cleaned_sheets = {}

    try:
        xls = pd.ExcelFile(file_path)
        target_sheets = sheet_names or xls.sheet_names

        for sheet in target_sheets:
            logger.info(f"Processing sheet: {sheet}")
            df = pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl')

            if formulas_to_values:
                df = convert_formulas_to_values(df)

            # --- Core string cleaning ---
            df = trim_strings(df)
            df = lowercase_columns(df)
            df = remove_extra_whitespace(df)
            df = remove_special_characters(df)

            # --- Null/missing handling ---
            df = fill_missing(df)
            df = drop_empty_rows(df)

            # --- Type casting ---
            if date_columns and sheet in date_columns:
                for col in date_columns[sheet]:
                    df = parse_dates(df, col)
            df = auto_cast_numeric(df)

            # --- Deduplication ---
            if dedup_columns and sheet in dedup_columns:
                df = drop_exact_duplicates(df, subset=dedup_columns[sheet])

            # --- Categorical mapping ---
            if categorical_mappings and sheet in categorical_mappings:
                for col, mapping in categorical_mappings[sheet].items():
                    df = map_categorical(df, col, mapping)

            # --- Regex-based cleaning ---
            if regex_cleaning and sheet in regex_cleaning:
                for col, pattern in regex_cleaning[sheet].items():
                    df = regex_replace(df, col, pattern)

            # --- Column renaming/standardization ---
            if column_renames and sheet in column_renames:
                df.rename(columns=column_renames[sheet], inplace=True)

            # --- Outlier detection / removal ---
            df = remove_outliers(df)

            # --- Conditional transformations ---
            df = conditional_transform(df)

            cleaned_sheets[sheet] = df
            logger.info(f"Finished cleaning sheet: {sheet}, rows: {len(df)}")

    except Exception as e:
        logger.error(f"Error cleaning Excel file: {e}")
        raise

    return cleaned_sheets


# -------------------------------
# --- Helper cleaning functions ---
# -------------------------------

def convert_formulas_to_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert formulas to their calculated values.
    For now, assuming read_excel already evaluates formulas using openpyxl.
    Stub included for future formula evaluation logic.
    """
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

def parse_dates(df: pd.DataFrame, column: str, date_format: Optional[str] = None) -> pd.DataFrame:
    try:
        df[column] = pd.to_datetime(df[column], format=date_format, errors='coerce')
    except Exception as e:
        logger.warning(f"Date parsing failed for column {column}: {e}")
    return df

def auto_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
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
          
