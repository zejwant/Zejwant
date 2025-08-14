"""
orc_cleaner.py
Enterprise-level ORC data cleaning module.

Capabilities:
- Load single or partitioned ORC files
- Column trimming, type normalization
- Null/missing value handling
- Deduplication
- Outlier detection
- Scaling/normalization for numeric columns
- Regex-based cleaning for string columns
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pyarrow.orc as orc
import pyarrow as pa

# Configure logger
logger = logging.getLogger("ORCCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class ORCCleaner:
    """
    Class to handle enterprise-level ORC data cleaning.
    """

    def __init__(self, file_path: str, partitioned: bool = False):
        """
        Initialize ORCCleaner.

        Args:
            file_path (str): Path to ORC file or directory.
            partitioned (bool): True if dataset is partitioned.
        """
        self.file_path = file_path
        self.partitioned = partitioned
        self.df: pd.DataFrame = self._load_orc()
        self.cleaned_df: Optional[pd.DataFrame] = None

    # ---------------- Data Loading ----------------

    def _load_orc(self) -> pd.DataFrame:
        """Load ORC file(s) into a Pandas DataFrame."""
        try:
            if self.partitioned:
                # For partitioned ORC datasets, read multiple ORC files
                import os, glob
                files = glob.glob(f"{self.file_path}/**/*.orc", recursive=True)
                dfs = []
                for file in files:
                    table = orc.ORCFile(file).read()
                    dfs.append(table.to_pandas())
                df = pd.concat(dfs, ignore_index=True)
            else:
                table = orc.ORCFile(self.file_path).read()
                df = table.to_pandas()

            logger.info(f"Loaded ORC dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            logger.error(f"Error loading ORC file: {e}")
            raise

    # ---------------- Cleaning Methods ----------------

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
    def minmax_scale(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        cols = columns or df.select_dtypes(include='number').columns.tolist()
        scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
        return df

    @staticmethod
    def standardize(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        cols = columns or df.select_dtypes(include='number').columns.tolist()
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
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
              columns_map: Optional[Dict[str, str]] = None,
              scale_method: Optional[str] = None) -> pd.DataFrame:
        """
        Apply all cleaning steps to ORC dataset.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        try:
            df = self.df.copy()

            df = self.trim_strings(df)
            df = self.lowercase_strings(df)
            df = self.remove_special_characters(df)
            df = self.fill_missing(df)
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

            if scale_method:
                if scale_method.lower() == "minmax":
                    df = self.minmax_scale(df)
                elif scale_method.lower() == "standard":
                    df = self.standardize(df)

            self.cleaned_df = df
            logger.info(f"ORC cleaning complete: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df

        except Exception as e:
            logger.error(f"Error cleaning ORC dataset: {e}")
            raise


# ------------------ Usage Example ------------------
if __name__ == "__main__":
    cleaner = ORCCleaner("sample.orc", partitioned=False)
    cleaned_df = cleaner.clean(date_columns=["created_at"],
                               categorical_columns=["status"],
                               scale_method="minmax")
    print(cleaned_df)
                
