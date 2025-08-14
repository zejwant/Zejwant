"""
pdf_cleaner.py
Enterprise-level PDF data cleaning module.

Capabilities:
- Extract text and tables from PDFs (multi-page)
- OCR for scanned PDFs
- Null/missing value handling
- Deduplication and normalization
- Regex cleaning for strings
- Date/time normalization
- Conditional transformations
- Logging and error handling
- 20+ cleaning methods for structured and unstructured content
- Returns cleaned Pandas DataFrame

Author: Varun Mode
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Optional, List, Dict, Any, Union
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from io import StringIO

# Configure logger
logger = logging.getLogger("PDFCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class PDFCleaner:
    """
    Class to handle enterprise-level PDF cleaning.
    """

    def __init__(self, pdf_file: str, ocr: bool = True):
        """
        Initialize PDFCleaner with file path.

        Args:
            pdf_file (str): Path to PDF file.
            ocr (bool): Enable OCR for scanned PDFs.
        """
        self.pdf_file = pdf_file
        self.ocr = ocr
        self.raw_data: List[pd.DataFrame] = []
        self.cleaned_df: Optional[pd.DataFrame] = None
        self._extract_pdf()

    # ---------------- PDF Extraction ----------------

    def _extract_pdf(self) -> None:
        """Extract tables and text from PDF."""
        try:
            with pdfplumber.open(self.pdf_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    logger.info(f"Processing page {i + 1}")
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        self.raw_data.append(df)
                    # Extract text if OCR enabled
                    if self.ocr and not tables:
                        img = page.to_image(resolution=300).original
                        text = pytesseract.image_to_string(img)
                        df_text = pd.DataFrame([line.split() for line in text.split('\n') if line.strip()])
                        if not df_text.empty:
                            self.raw_data.append(df_text)
            logger.info(f"PDF extraction complete: {len(self.raw_data)} dataframes extracted.")
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise

    # ---------------- Cleaning Methods ----------------

    @staticmethod
    def fill_missing(df: pd.DataFrame, fill_value: Any = "") -> pd.DataFrame:
        return df.fillna(fill_value)

    @staticmethod
    def drop_missing(df: pd.DataFrame, axis: int = 0, thresh: Optional[int] = None) -> pd.DataFrame:
        return df.dropna(axis=axis, thresh=thresh)

    @staticmethod
    def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
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

    @staticmethod
    def auto_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return df

    @staticmethod
    def parse_dates(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        return df.drop_duplicates(subset=subset)

    @staticmethod
    def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column in df.columns:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        return df

    @staticmethod
    def standardize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column in df.columns:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
        return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            z_scores = (df[col] - df[col].mean()) / df[col].std(ddof=0)
            df = df[z_scores.abs() <= z_thresh]
        return df

    @staticmethod
    def regex_replace(df: pd.DataFrame, column: str, pattern: str, replacement: str = "") -> pd.DataFrame:
        if column in df.columns:
            df[column] = df[column].astype(str).str.replace(pattern, replacement, regex=True)
        return df

    @staticmethod
    def rename_columns(df: pd.DataFrame, columns_map: Dict[str, str]) -> pd.DataFrame:
        return df.rename(columns=columns_map)

    @staticmethod
    def conditional_transform(df: pd.DataFrame) -> pd.DataFrame:
        if 'status' in df.columns:
            df['status'] = df['status'].str.upper()
        return df

    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return pd.get_dummies(df, columns=columns, drop_first=True)

    @staticmethod
    def replace_values(df: pd.DataFrame, replace_map: Dict[str, Any]) -> pd.DataFrame:
        return df.replace(replace_map)

    @staticmethod
    def sort_by_column(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
        if column in df.columns:
            df = df.sort_values(by=column, ascending=ascending)
        return df

    # ---------------- Main Cleaning Pipeline ----------------

    def clean(self,
              date_columns: Optional[List[str]] = None,
              categorical_columns: Optional[List[str]] = None,
              regex_columns: Optional[Dict[str, str]] = None,
              replace_map: Optional[Dict[str, Any]] = None,
              columns_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Apply all cleaning steps to extracted PDF data and merge into a single DataFrame.

        Returns:
            pd.DataFrame: Cleaned PDF data.
        """
        try:
            # Merge all extracted DataFrames
            df = pd.concat(self.raw_data, ignore_index=True) if self.raw_data else pd.DataFrame()

            # Apply cleaning methods
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
            logger.info(f"PDF cleaning complete: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df

        except Exception as e:
            logger.error(f"Error cleaning PDF: {e}")
            raise


# ------------------ Usage Example ------------------
if __name__ == "__main__":
    cleaner = PDFCleaner("sample.pdf", ocr=True)
    cleaned_df = cleaner.clean(date_columns=[], categorical_columns=['status'])
    print(cleaned_df)
            
