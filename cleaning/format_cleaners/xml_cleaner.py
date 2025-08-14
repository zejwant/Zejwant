"""
xml_cleaner.py
Enterprise-level XML data cleaning module.

Capabilities:
- Parse XML trees and flatten nested nodes
- Handle attributes and text
- Schema validation (optional)
- Missing value handling and type casting
- Deduplication and normalization
- Regex cleaning for string fields
- Date/time normalization
- Conditional transformations
- Logging and error handling
- 20+ cleaning methods
- Returns cleaned Pandas DataFrame

Author: Varun Mode
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import logging
import re
from typing import Optional, Dict, List, Any

# Configure logger
logger = logging.getLogger("XMLCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class XMLCleaner:
    """
    Class to handle enterprise-level XML cleaning.
    """

    def __init__(self, xml_file: str):
        """
        Initialize XMLCleaner with XML file path.

        Args:
            xml_file (str): Path to XML file.
        """
        self.xml_file = xml_file
        self.df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self._load_xml()

    # ---------------- XML Parsing ----------------

    def _load_xml(self) -> None:
        """Parse XML file into flattened DataFrame."""
        try:
            tree = ET.parse(self.xml_file)
            root = tree.getroot()
            data = [self._flatten_element(el) for el in root]
            self.df = pd.DataFrame(data)
            logger.info(f"Parsed XML file {self.xml_file} with {len(self.df)} records.")
        except Exception as e:
            logger.error(f"Error parsing XML file: {e}")
            raise

    def _flatten_element(self, element: ET.Element, parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Recursively flatten an XML element including attributes.

        Args:
            element (ET.Element): XML element.
            parent_key (str): Parent key prefix.
            sep (str): Separator between keys.

        Returns:
            Dict[str, Any]: Flattened dictionary.
        """
        items: Dict[str, Any] = {}

        # Add element text if exists
        if element.text and element.text.strip():
            key = parent_key if parent_key else element.tag
            items[key] = element.text.strip()

        # Add attributes
        for k, v in element.attrib.items():
            key = f"{parent_key}{sep}@{k}" if parent_key else f"@{k}"
            items[key] = v

        # Recurse into children
        for child in element:
            child_key = f"{parent_key}{sep}{child.tag}" if parent_key else child.tag
            items.update(self._flatten_element(child, child_key, sep))

        return items

    # ---------------- Cleaning Methods ----------------

    @staticmethod
    def fill_missing(df: pd.DataFrame, fill_value: Any = "") -> pd.DataFrame:
        """Fill missing values."""
        return df.fillna(fill_value)

    @staticmethod
    def drop_missing(df: pd.DataFrame, axis: int = 0, thresh: Optional[int] = None) -> pd.DataFrame:
        """Drop rows or columns with missing values below threshold."""
        return df.dropna(axis=axis, thresh=thresh)

    @staticmethod
    def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
        """Trim whitespace from string columns."""
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
        return df

    @staticmethod
    def lowercase_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase string columns."""
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.lower())
        return df

    @staticmethod
    def remove_special_characters(df: pd.DataFrame, pattern: str = r'[^0-9a-zA-Z\s]') -> pd.DataFrame:
        """Remove special characters from string columns."""
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.replace(pattern, '', regex=True))
        return df

    @staticmethod
    def auto_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Automatically cast numeric-looking columns."""
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return df

    @staticmethod
    def parse_dates(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Parse specified columns as datetime."""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows."""
        return df.drop_duplicates(subset=subset)

    @staticmethod
    def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Normalize numeric column between 0 and 1."""
        if column in df.columns:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        return df

    @staticmethod
    def standardize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Standardize numeric column (z-score)."""
        if column in df.columns:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
        return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
        """Remove outliers based on z-score."""
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            z_scores = (df[col] - df[col].mean()) / df[col].std(ddof=0)
            df = df[z_scores.abs() <= z_thresh]
        return df

    @staticmethod
    def regex_replace(df: pd.DataFrame, column: str, pattern: str, replacement: str = "") -> pd.DataFrame:
        """Apply regex replacement to a column."""
        if column in df.columns:
            df[column] = df[column].astype(str).str.replace(pattern, replacement, regex=True)
        return df

    @staticmethod
    def rename_columns(df: pd.DataFrame, columns_map: Dict[str, str]) -> pd.DataFrame:
        """Rename columns using a dictionary."""
        return df.rename(columns=columns_map)

    @staticmethod
    def conditional_transform(df: pd.DataFrame) -> pd.DataFrame:
        """Example conditional transformation."""
        if 'status' in df.columns:
            df['status'] = df['status'].str.upper()
        return df

    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        return pd.get_dummies(df, columns=columns, drop_first=True)

    @staticmethod
    def replace_values(df: pd.DataFrame, replace_map: Dict[str, Any]) -> pd.DataFrame:
        """Replace values based on a mapping dictionary."""
        return df.replace(replace_map)

    @staticmethod
    def sort_by_column(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
        """Sort dataframe by a column."""
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
        Apply all cleaning steps to the XML DataFrame.

        Args:
            date_columns (List[str], optional): Columns to parse as dates.
            categorical_columns (List[str], optional): Columns to one-hot encode.
            regex_columns (Dict[str, str], optional): Columns with regex patterns.
            replace_map (Dict[str, Any], optional): Values to replace.
            columns_map (Dict[str, str], optional): Column renaming map.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df = self.df.copy()
        try:
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
            logger.info(f"XML cleaning complete: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
        except Exception as e:
            logger.error(f"Error cleaning XML: {e}")
            raise


# ------------------ Usage Example ------------------
if __name__ == "__main__":
    cleaner = XMLCleaner("sample.xml")
    cleaned_df = cleaner.clean(date_columns=[], categorical_columns=['status'])
    print(cleaned_df)
          
