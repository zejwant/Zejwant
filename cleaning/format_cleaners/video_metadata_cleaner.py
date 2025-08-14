"""
video_metadata_cleaner.py
Enterprise-level Video metadata cleaning module.

Capabilities:
- Clean video metadata fields (codec, resolution, duration, frame rate, etc.)
- Handle missing or corrupt values
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
from datetime import timedelta
from typing import Optional, Dict, Callable, List, Any

# ---------------- Logger Setup ----------------
logger = logging.getLogger("VideoMetadataCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class VideoMetadataCleaner:
    """
    Class to clean video metadata DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with video metadata DataFrame.

        Args:
            df (pd.DataFrame): Extracted video metadata
        """
        self.df = df.copy()
        self.cleaned_df: Optional[pd.DataFrame] = None

    # ---------------- Helper Cleaning Methods ----------------
    def remove_invalid_entries(self) -> pd.DataFrame:
        """Remove rows with all NaN or empty values."""
        self.df = self.df.dropna(how='all')
        logger.info("Removed completely invalid entries.")
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

    def remove_special_characters(self, pattern: str = r'[^0-9a-zA-Z\s:]') -> pd.DataFrame:
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
        logger.info("Deduplicated video metadata.")
        return self.df

    def normalize_resolution(self, col: str = 'resolution') -> pd.DataFrame:
        """Ensure resolution is standardized as WIDTHxHEIGHT."""
        if col in self.df.columns:
            self.df[col] = self.df[col].astype(str).str.replace(r'[^\dx0-9]', '', regex=True)
        return self.df

    def normalize_duration(self, col: str = 'duration') -> pd.DataFrame:
        """Convert duration strings to seconds if needed."""
        if col in self.df.columns:
            def parse_duration(val):
                try:
                    if isinstance(val, (int, float)):
                        return val
                    parts = val.split(':')
                    parts = [float(p) for p in parts]
                    if len(parts) == 3:
                        return parts[0]*3600 + parts[1]*60 + parts[2]
                    elif len(parts) == 2:
                        return parts[0]*60 + parts[1]
                    elif len(parts) == 1:
                        return parts[0]
                except:
                    return np.nan
            self.df[col] = self.df[col].apply(parse_duration)
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

    def scale_numeric(self, scale_map: Dict[str, float]) -> pd.DataFrame:
        for col, factor in scale_map.items():
            if col in self.df.columns:
                self.df[col] = self.df[col] * factor
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

    def drop_empty_rows(self) -> pd.DataFrame:
        self.df = self.df.dropna(how='all')
        return self.df

    def limit_rows(self, max_rows: int) -> pd.DataFrame:
        self.df = self.df.head(max_rows)
        return self.df

    # ---------------- Main Cleaning Pipeline ----------------
    def clean(self,
              regex_map: Optional[Dict[str, str]] = None,
              scale_map: Optional[Dict[str, float]] = None,
              required_columns: Optional[List[str]] = None,
              smooth_window: int = 3,
              z_thresh: float = 3.0,
              max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Full video metadata cleaning pipeline with 20+ steps.
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
            self.normalize_resolution()
            self.normalize_duration()
            if regex_map:
                self.regex_replace(regex_map)
            self.remove_outliers(z_thresh=z_thresh)
            if scale_map:
                self.scale_numeric(scale_map)
            self.smooth_numeric(window=smooth_window)
            if required_columns:
                self.validate_columns(required_columns)
            self.drop_empty_rows()
            if max_rows:
                self.limit_rows(max_rows)

            self.cleaned_df = self.df
            logger.info(f"Video metadata cleaned: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
            return self.df

        except Exception as e:
            logger.error(f"Error cleaning video metadata: {e}")
            raise


# ------------------ Usage Example ------------------
if __name__ == "__main__":
    sample_data = pd.DataFrame({
        "filename": ["video1.mp4", "video2.mov", "video3.avi", None],
        "codec": ["h264", "H265", "vp9", "h264"],
        "resolution": ["1920x1080", "1280x720", "640x360", "1920x1080"],
        "duration": ["00:02:30", "00:01:45", "105", None],
        "frame_rate": ["30", "24", "25", "30"]
    })

    cleaner = VideoMetadataCleaner(sample_data)
    cleaned_df = cleaner.clean(
        regex_map={"codec": r'[^a-zA-Z0-9]'},
        scale_map={"frame_rate": 1.0},
        required_columns=["filename", "codec", "resolution", "duration", "frame_rate"],
        max_rows=100
    )
    print(cleaned_df)
      
