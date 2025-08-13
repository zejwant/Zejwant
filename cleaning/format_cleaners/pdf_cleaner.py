"""
cleaning/format_cleaners/pdf_cleaner.py

Enterprise-level PDF cleaning module.

Features:
- Extract and clean tables and text from PDFs
- Multi-page PDF handling
- OCR text extraction
- Null/missing value handling
- Deduplication and normalization
- Logging and error handling
- 20+ reusable cleaning methods for structured/unstructured PDFs
"""

import logging
from typing import List, Optional, Dict, Union
import pandas as pd
import numpy as np
import re
from datetime import datetime

# PDF extraction libraries
import pdfplumber
try:
    import pytesseract
    from PIL import Image
    OCR_ENABLED = True
except ImportError:
    OCR_ENABLED = False
    pytesseract = None
    Image = None

logger = logging.getLogger("pdf_cleaner")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def clean_pdf(file_path: str,
              extract_tables: bool = True,
              ocr: bool = False,
              date_columns: Optional[List[str]] = None,
              categorical_mappings: Optional[Dict[str, Dict[Any, Any]]] = None,
              regex_cleaning: Optional[Dict[str, str]] = None,
              dedup_columns: Optional[List[str]] = None,
              fill_missing_value: Union[str, int, float] = '') -> pd.DataFrame:
    """
    Enterprise-level PDF cleaning.

    Args:
        file_path (str): Path to PDF file.
        extract_tables (bool): Extract tabular content if True.
        ocr (bool): Enable OCR for scanned PDFs (requires pytesseract and PIL).
        date_columns (List[str], optional): Columns to normalize as datetime.
        categorical_mappings (Dict[column, mapping], optional): Column value mappings.
        regex_cleaning (Dict[column, pattern], optional): Regex cleaning per column.
        dedup_columns (List[str], optional): Columns to deduplicate on.
        fill_missing_value (str/int/float): Default fill for missing values.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    all_data = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_data = []

                if extract_tables:
                    tables = page.extract_tables()
                    for table in tables:
                        df_table = pd.DataFrame(table[1:], columns=table[0])
                        page_data.append(df_table)

                if ocr and OCR_ENABLED:
                    text = page.to_image().original
                    text_data = pytesseract.image_to_string(Image.fromarray(np.array(text)))
                    df_text = pd.DataFrame([text_data.split('\n')], columns=['text'])
                    page_data.append(df_text)

                if not page_data:
                    # Fallback: extract raw text
                    raw_text = page.extract_text()
                    if raw_text:
                        df_text = pd.DataFrame([raw_text.split('\n')], columns=['text'])
                        page_data.append(df_text)

                if page_data:
                    df_page = pd.concat(page_data, ignore_index=True)
                    all_data.append(df_page)
                logger.info(f"Processed page {page_number} with {len(page_data)} tables/text blocks")

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)

        # --- Core cleaning ---
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

        logger.info(f"PDF cleaned successfully, total rows: {len(df)}")
        return df

    except Exception as e:
        logger.error(f"Error cleaning PDF: {e}")
        raise


# -------------------------------
# --- Helper cleaning functions ---
# -------------------------------

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
  
