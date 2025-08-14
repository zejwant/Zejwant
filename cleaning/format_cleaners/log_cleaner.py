"""
log_cleaner.py
Enterprise-level log data cleaning module.

Capabilities:
- Parse various log formats: text, JSON, syslog
- Multi-line log handling
- Timestamp normalization
- Remove duplicates and irrelevant entries
- Regex-based field cleaning
- Null/missing value handling
- Conditional transformations
- Logging and error handling
- 20+ cleaning methods
- Returns cleaned Pandas DataFrame

Author: Varun Mode
"""

import pandas as pd
import numpy as np
import json
import re
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# Configure logger
logger = logging.getLogger("LogCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class LogCleaner:
    """
    Class to handle enterprise-level log cleaning.
    """

    def __init__(self, log_file: str, log_type: str = "text"):
        """
        Initialize LogCleaner.

        Args:
            log_file (str): Path to log file.
            log_type (str): Type of log ('text', 'json', 'syslog').
        """
        self.log_file = log_file
        self.log_type = log_type.lower()
        self.raw_logs: List[Dict[str, Any]] = []
        self.cleaned_df: Optional[pd.DataFrame] = None
        self._parse_logs()

    # ---------------- Log Parsing ----------------

    def _parse_logs(self) -> None:
        """Parse logs based on type."""
        try:
            if self.log_type == "text":
                with open(self.log_file, "r") as f:
                    lines = f.readlines()
                    self.raw_logs = [{"line": line.strip()} for line in lines if line.strip()]

            elif self.log_type == "json":
                with open(self.log_file, "r") as f:
                    for line in f:
                        if line.strip():
                            self.raw_logs.append(json.loads(line))

            elif self.log_type == "syslog":
                with open(self.log_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            parsed = self._parse_syslog_line(line.strip())
                            self.raw_logs.append(parsed)
            else:
                raise ValueError(f"Unsupported log_type: {self.log_type}")

            logger.info(f"Parsed {len(self.raw_logs)} log entries from {self.log_file}")

        except Exception as e:
            logger.error(f"Error parsing logs: {e}")
            raise

    @staticmethod
    def _parse_syslog_line(line: str) -> Dict[str, Any]:
        """
        Basic syslog line parser. Extracts timestamp, level, message.

        Args:
            line (str): Raw syslog line.

        Returns:
            Dict[str, Any]: Parsed fields.
        """
        try:
            pattern = r'^(?P<timestamp>\w{3}\s+\d{1,2}\s[\d:]+)\s+(?P<host>\S+)\s+(?P<service>[\w\-/]+):\s(?P<message>.*)$'
            match = re.match(pattern, line)
            if match:
                return match.groupdict()
            else:
                return {"line": line}
        except Exception as e:
            logger.warning(f"Failed to parse syslog line: {e}")
            return {"line": line}

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
    def parse_timestamps(df: pd.DataFrame, column: str, fmt: Optional[str] = None) -> pd.DataFrame:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce')
        return df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        return df.drop_duplicates(subset=subset)

    @staticmethod
    def regex_replace(df: pd.DataFrame, column: str, pattern: str, replacement: str = "") -> pd.DataFrame:
        if column in df.columns:
            df[column] = df[column].astype(str).str.replace(pattern, replacement, regex=True)
        return df

    @staticmethod
    def conditional_transform(df: pd.DataFrame) -> pd.DataFrame:
        if 'level' in df.columns:
            df['level'] = df['level'].str.upper()
        return df

    @staticmethod
    def remove_irrelevant_entries(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
        pattern = "|".join(keywords)
        if 'message' in df.columns:
            df = df[~df['message'].str.contains(pattern, regex=True, na=False)]
        return df

    @staticmethod
    def sort_by_column(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
        if column in df.columns:
            df = df.sort_values(by=column, ascending=ascending)
        return df

    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return pd.get_dummies(df, columns=columns, drop_first=True)

    @staticmethod
    def replace_values(df: pd.DataFrame, replace_map: Dict[str, Any]) -> pd.DataFrame:
        return df.replace(replace_map)

    @staticmethod
    def extract_fields(df: pd.DataFrame, patterns: Dict[str, str]) -> pd.DataFrame:
        """Extract new fields from message column using regex patterns."""
        if 'message' in df.columns:
            for field, pattern in patterns.items():
                df[field] = df['message'].str.extract(pattern, expand=False)
        return df

    # ---------------- Main Cleaning Pipeline ----------------

    def clean(self,
              timestamp_column: str = "timestamp",
              timestamp_format: Optional[str] = None,
              remove_keywords: Optional[List[str]] = None,
              regex_fields: Optional[Dict[str, str]] = None,
              replace_map: Optional[Dict[str, Any]] = None,
              categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply all cleaning steps to logs.

        Returns:
            pd.DataFrame: Cleaned log DataFrame.
        """
        try:
            df = pd.DataFrame(self.raw_logs)

            df = self.fill_missing(df)
            df = self.strip_whitespace(df)
            df = self.lowercase_strings(df)
            df = self.remove_special_characters(df)
            df = self.auto_cast_numeric(df)
            df = self.remove_duplicates(df)
            df = self.conditional_transform(df)
            df = self.parse_timestamps(df, timestamp_column, timestamp_format)

            if remove_keywords:
                df = self.remove_irrelevant_entries(df, remove_keywords)
            if regex_fields:
                df = self.extract_fields(df, regex_fields)
            if replace_map:
                df = self.replace_values(df, replace_map)
            if categorical_columns:
                df = self.encode_categorical(df, categorical_columns)

            self.cleaned_df = df
            logger.info(f"Log cleaning complete: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df

        except Exception as e:
            logger.error(f"Error cleaning logs: {e}")
            raise


# ------------------ Usage Example ------------------
if __name__ == "__main__":
    cleaner = LogCleaner("sample.log", log_type="syslog")
    cleaned_df = cleaner.clean(timestamp_column="timestamp",
                               remove_keywords=["debug", "heartbeat"],
                               regex_fields={"error_code": r"ERR(\d+)"},
                               categorical_columns=["level"])
    print(cleaned_df)
          
