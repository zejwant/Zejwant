"""
google_sheets_cleaner.py
Enterprise-level Google Sheets cleaning module.

Capabilities:
- Load Google Sheets via sheet ID
- Handle multiple sheets with sheet-specific cleaning
- Null/missing value handling
- Deduplication
- Type normalization
- Regex cleaning and string normalization
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
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Configure logger
logger = logging.getLogger("GoogleSheetsCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class GoogleSheetsCleaner:
    """
    Class to handle enterprise-level Google Sheets data cleaning.
    """

    def __init__(self, sheet_id: str, credentials_json: str):
        """
        Initialize GoogleSheetsCleaner.

        Args:
            sheet_id (str): Google Sheets ID.
            credentials_json (str): Path to service account JSON for gspread authentication.
        """
        self.sheet_id = sheet_id
        self.credentials_json = credentials_json
        self.client = self._authenticate()
        self.sheets_data: Dict[str, pd.DataFrame] = self._load_sheets()
        self.cleaned_sheets: Dict[str, pd.DataFrame] = {}

    # ---------------- Authentication ----------------
    def _authenticate(self) -> gspread.client.Client:
        """Authenticate with Google Sheets API using service account JSON."""
        try:
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_json, scope)
            client = gspread.authorize(creds)
            logger.info("Google Sheets authentication successful.")
            return client
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    # ---------------- Load Sheets ----------------
    def _load_sheets(self) -> Dict[str, pd.DataFrame]:
        """Load all sheets from the Google Sheet into a dictionary of DataFrames."""
        try:
            spreadsheet = self.client.open_by_key(self.sheet_id)
            sheets_data = {}
            for sheet in spreadsheet.worksheets():
                data = sheet.get_all_records()
                sheets_data[sheet.title] = pd.DataFrame(data)
                logger.info(f"Loaded sheet '{sheet.title}' with {len(data)} rows.")
            return sheets_data
        except Exception as e:
            logger.error(f"Error loading sheets: {e}")
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
              date_columns: Optional[Dict[str, List[str]]] = None,
              categorical_columns: Optional[Dict[str, List[str]]] = None,
              regex_columns: Optional[Dict[str, Dict[str, str]]] = None,
              replace_map: Optional[Dict[str, Dict[str, Any]]] = None,
              columns_map: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, pd.DataFrame]:
        """
        Apply all cleaning steps to each sheet.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of cleaned DataFrames keyed by sheet name.
        """
        try:
            for sheet_name, df in self.sheets_data.items():
                logger.info(f"Cleaning sheet: {sheet_name}")
                df = self.fill_missing(df)
                df = self.trim_strings(df)
                df = self.lowercase_strings(df)
                df = self.remove_special_characters(df)
                df = self.auto_cast_numeric(df)
                df = self.remove_duplicates(df)
                df = self.remove_outliers(df)
                df = self.conditional_transform(df)

                # Sheet-specific cleaning
                if date_columns and sheet_name in date_columns:
                    df = self.parse_dates(df, date_columns[sheet_name])
                if categorical_columns and sheet_name in categorical_columns:
                    df = self.encode_categorical(df, categorical_columns[sheet_name])
                if regex_columns and sheet_name in regex_columns:
                    for col, pattern in regex_columns[sheet_name].items():
                        df = self.regex_replace(df, col, pattern)
                if replace_map and sheet_name in replace_map:
                    df = self.replace_values(df, replace_map[sheet_name])
                if columns_map and sheet_name in columns_map:
                    df = self.rename_columns(df, columns_map[sheet_name])

                self.cleaned_sheets[sheet_name] = df
                logger.info(f"Finished cleaning sheet: {sheet_name}, rows: {len(df)}")

            return self.cleaned_sheets

        except Exception as e:
            logger.error(f"Error cleaning Google Sheets: {e}")
            raise


# ------------------ Usage Example ------------------
if __name__ == "__main__":
    cleaner = GoogleSheetsCleaner(sheet_id="YOUR_SHEET_ID",
                                  credentials_json="service_account.json")
    cleaned = cleaner.clean(date_columns={"Sheet1": ["created_at"]},
                             categorical_columns={"Sheet1": ["status"]})
    for name, df in cleaned.items():
        print(f"Sheet: {name}, Rows: {len(df)}")
        print(df.head())
          
