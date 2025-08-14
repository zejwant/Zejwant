"""
html_scraper_cleaner.py
Enterprise-level HTML scraper cleaning module.

Capabilities:
- Clean HTML tables and scraped elements
- Remove unwanted tags and characters
- Normalize types, handle missing values
- Deduplication
- Regex-based cleaning and string normalization
- Conditional transformations
- Logging and error handling
- 20+ cleaning methods
- Returns cleaned Pandas DataFrame

Author: Varun Mode
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup

# Configure logger
logger = logging.getLogger("HTMLScraperCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class HTMLScraperCleaner:
    """
    Class to clean HTML scraped data into structured Pandas DataFrame.
    """

    def __init__(self, html_content: str):
        """
        Initialize with raw HTML content.

        Args:
            html_content (str): HTML content as string.
        """
        self.html_content = html_content
        self.df: pd.DataFrame = self._parse_html()
        self.cleaned_df: Optional[pd.DataFrame] = None

    # ---------------- HTML Parsing ----------------
    def _parse_html(self) -> pd.DataFrame:
        """Parse HTML tables into a DataFrame."""
        try:
            tables = pd.read_html(self.html_content)
            if len(tables) == 0:
                logger.warning("No tables found in HTML content.")
                return pd.DataFrame()
            # Combine all tables into a single DataFrame
            df = pd.concat(tables, ignore_index=True)
            logger.info(f"Parsed HTML tables: {len(tables)}, total rows: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            raise

    # ---------------- Core Cleaning Methods ----------------
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
    def remove_html_tags(df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.apply(lambda v: BeautifulSoup(v, "html.parser").get_text() if isinstance(v, str) else v))
        return df

    @staticmethod
    def remove_special_characters(df: pd.DataFrame, pattern: str = r'[^0-9a-zA-Z\s]') -> pd.DataFrame:
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.replace(pattern, '', regex=True))
        return df

    @staticmethod
    def fill_missing(df: pd.DataFrame, fill_value: Any = "") -> pd.DataFrame:
        return df.fillna(fill_value)

    @staticmethod
    def drop_missing(df: pd.DataFrame, axis: int = 0, thresh: Optional[int] = None) -> pd.DataFrame:
        return df.dropna(axis=axis, thresh=thresh)

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
    def replace_values(df: pd.DataFrame, replace_map: Dict[str, Any]) -> pd.DataFrame:
        return df.replace(replace_map)

    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return pd.get_dummies(df, columns=columns, drop_first=True)

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
        Apply all cleaning steps to the scraped HTML DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        try:
            df = self.df.copy()

            df = self.fill_missing(df)
            df = self.remove_html_tags(df)
            df = self.trim_strings(df)
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
            logger.info(f"HTML scraping cleaning complete: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df

        except Exception as e:
            logger.error(f"Error cleaning HTML scraped data: {e}")
            raise


# ------------------ Usage Example ------------------
if __name__ == "__main__":
    sample_html = "<table><tr><th>Name</th><th>Age</th></tr><tr><td> Alice </td><td>30</td></tr></table>"
    cleaner = HTMLScraperCleaner(sample_html)
    cleaned_df = cleaner.clean()
    print(cleaned_df)
      
