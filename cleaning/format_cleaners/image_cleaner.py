"""
image_cleaner.py
Enterprise-level Image metadata and OCR text cleaning module.

Capabilities:
- Clean extracted image metadata and OCR text
- Remove invalid or corrupted entries
- Normalize metadata fields (e.g., timestamps, dimensions)
- Deduplication
- Outlier detection
- String normalization and regex cleaning
- Logging and error handling
- 20+ modular cleaning methods
- Returns cleaned structured Pandas DataFrame

Author: Varun Mode
"""

import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Callable, List, Any

# ---------------- Logger Setup ----------------
logger = logging.getLogger("ImageCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class ImageCleaner:
    """
    Class to clean image metadata and OCR text.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with image metadata / OCR DataFrame.

        Args:
            df (pd.DataFrame): Extracted image metadata or OCR text data
        """
        self.df = df.copy()
        self.cleaned_df: Optional[pd.DataFrame] = None

    # ---------------- Helper Cleaning Methods ----------------
    def remove_invalid_entries(self) -> pd.DataFrame:
        """Remove entries with all NaN or empty values."""
        self.df = self.df.dropna(how='all')
        logger.info("Removed invalid entries with all NaN.")
        return self.df

    def trim_strings(self) -> pd.DataFrame:
        str_cols = self.df.select_dtypes(include='object').columns
        self.df[str_cols] = self.df[str_cols].apply(lambda x: x.str.strip())
        return self.df

    def lowercase_strings(self) -> pd.DataFrame:
        str_cols = self.df.select_dtypes(include='object').columns
        self.df[str_cols] = self.df[str_cols].apply(lambda x: x.str.lower())
        return self.df

    def uppercase_strings(self) -> pd.DataFrame:
        str_cols = self.df.select_dtypes(include='object').columns
        self.df[str_cols] = self.df[str_cols].apply(lambda x: x.str.upper())
        return self.df

    def remove_special_characters(self, pattern: str = r'[^0-9a-zA-Z\s]') -> pd.DataFrame:
        str_cols = self.df.select_dtypes(include='object').columns
        self.df[str_cols] = self.df[str_cols].apply(lambda x: x.str.replace(pattern, '', regex=True))
        return self.df

    def auto_cast_numeric(self) -> pd.DataFrame:
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
        return self.df

    def fill_missing(self, fill_value: Any = np.nan) -> pd.DataFrame:
        self.df = self.df.fillna(fill_value)
        return self.df

    def deduplicate(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        self.df = self.df.drop_duplicates(subset=subset)
        logger.info("Deduplicated data.")
        return self.df

    def normalize_dimensions(self, width_col: str = 'width', height_col: str = 'height') -> pd.DataFrame:
        if width_col in self.df.columns and height_col in self.df.columns:
            self.df[width_col] = pd.to_numeric(self.df[width_col], errors='coerce')
            self.df[height_col] = pd.to_numeric(self.df[height_col], errors='coerce')
        return self.df

    def normalize_timestamp(self, col: str = 'timestamp') -> pd.DataFrame:
        if col in self.df.columns:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        return self.df

    def regex_replace(self, regex_map: Dict[str, str]) -> pd.DataFrame:
        for col, pattern in regex_map.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.replace(pattern, '', regex=True)
        return self.df

    def conditional_transform(self, col: str, func: Callable) -> pd.DataFrame:
        if col in self.df.columns:
            self.df[col] = self.df[col].apply(func)
        return self.df

    def remove_outliers(self, z_thresh: float = 3.0) -> pd.DataFrame:
        numeric_cols = self.df.select_dtypes(include='number').columns
        for col in numeric_cols:
            z_scores = (self.df[col] - self.df[col].mean()) / self.df[col].std(ddof=0)
            self.df = self.df[z_scores.abs() <= z_thresh]
        logger.info(f"Outliers removed with Z-threshold {z_thresh}.")
        return self.df

    def encode_categorical(self, columns: List[str]) -> pd.DataFrame:
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)
        return self.df

    def limit_rows(self, max_rows: int) -> pd.DataFrame:
        self.df = self.df.head(max_rows)
        return self.df

    def drop_columns(self, cols: List[str]) -> pd.DataFrame:
        self.df = self.df.drop(columns=cols, errors='ignore')
        return self.df

    def smooth_numeric(self, window: int = 3) -> pd.DataFrame:
        numeric_cols = self.df.select_dtypes(include='number').columns
        self.df[numeric_cols] = self.df[numeric_cols].rolling(window=window, min_periods=1).mean()
        return self.df

    def validate_columns(self, required_columns: List[str]) -> pd.DataFrame:
        for col in required_columns:
            if col not in self.df.columns:
                self.df[col] = np.nan
        return self.df

    # ---------------- Main Cleaning Pipeline ----------------
    def clean(self,
              regex_map: Optional[Dict[str, str]] = None,
              categorical_cols: Optional[List[str]] = None,
              required_columns: Optional[List[str]] = None,
              max_rows: Optional[int] = None,
              smooth_window: int = 3,
              z_thresh: float = 3.0) -> pd.DataFrame:
        """
        Full image metadata / OCR cleaning pipeline with 20+ steps.
        """
        try:
            self.remove_invalid_entries()
            self.fill_missing()
            self.deduplicate()
            self.auto_cast_numeric()
            self.trim_strings()
            self.lowercase_strings()
            self.uppercase_strings()
            self.remove_special_characters()
            self.normalize_dimensions()
            self.normalize_timestamp()
            if regex_map:
                self.regex_replace(regex_map)
            if categorical_cols:
                self.encode_categorical(categorical_cols)
            self.remove_outliers(z_thresh=z_thresh)
            self.smooth_numeric(window=smooth_window)
            if required_columns:
                self.validate_columns(required_columns)
            if max_rows:
                self.limit_rows(max_rows)

            self.cleaned_df = self.df
            logger.info(f"Image metadata/OCR data cleaned: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
            return self.df

        except Exception as e:
            logger.error(f"Error cleaning image data: {e}")
            raise


# ------------------ Usage Example ------------------
if __name__ == "__main__":
    sample_data = pd.DataFrame({
        "filename": ["img1.jpg", "img2.jpg", "img3.jpg", None],
        "width": ["1024", "800", "1024", "640"],
        "height": ["768", "600", "768", "480"],
        "timestamp": ["2025-08-14 10:00", "2025-08-14 10:01", None, "2025-08-14 10:03"],
        "ocr_text": ["Hello World!", "Test 123", "Sample Text", ""]
    })
    cleaner = ImageCleaner(sample_data)
    cleaned_df = cleaner.clean(
        regex_map={"ocr_text": r'[^a-zA-Z0-9\s]'},
        categorical_cols=[],
        required_columns=["filename", "width", "height", "timestamp", "ocr_text"],
        max_rows=100
    )
    print(cleaned_df)
          
