"""
yaml_cleaner.py
Enterprise-level YAML cleaning module with 20+ methods.

Capabilities:
- Flatten nested YAML structures
- Handle nested dicts and lists
- Null/missing value handling
- Type normalization and casting
- Deduplication
- Regex-based string cleaning
- Date/time normalization
- Conditional transformations
- Key renaming/standardization
- Outlier detection for numeric fields
- Logging and error handling
- Returns cleaned dict or Pandas DataFrame

Author: Varun Mode
"""

import yaml
import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, Union, List

# Configure logger
logger = logging.getLogger("YAMLCleaner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class YAMLCleaner:
    """
    Class to clean YAML files into structured dict or DataFrame.
    """

    def __init__(self, yaml_content: Union[str, Dict[str, Any]]):
        """
        Initialize with YAML content.

        Args:
            yaml_content (str | dict): YAML file path or dict content.
        """
        if isinstance(yaml_content, str):
            try:
                with open(yaml_content, 'r') as f:
                    self.data = yaml.safe_load(f)
                logger.info(f"YAML file loaded: {yaml_content}")
            except Exception as e:
                logger.error(f"Failed to load YAML file: {e}")
                raise
        elif isinstance(yaml_content, dict):
            self.data = yaml_content
        else:
            raise ValueError("yaml_content must be a file path or a dict.")

        self.cleaned_dict: Dict[str, Any] = {}
        self.cleaned_df: Optional[pd.DataFrame] = None

    # ---------------- Helper Cleaning Methods ----------------
    def flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        """Flatten nested dictionary including lists."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self.flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def trim_strings(d: dict) -> dict:
        return {k: v.strip() if isinstance(v, str) else v for k, v in d.items()}

    @staticmethod
    def lowercase_strings(d: dict) -> dict:
        return {k: v.lower() if isinstance(v, str) else v for k, v in d.items()}

    @staticmethod
    def uppercase_strings(d: dict) -> dict:
        return {k: v.upper() if isinstance(v, str) else v for k, v in d.items()}

    @staticmethod
    def remove_special_characters(d: dict, pattern: str = r'[^0-9a-zA-Z\s]') -> dict:
        return {k: re.sub(pattern, '', v) if isinstance(v, str) else v for k, v in d.items()}

    @staticmethod
    def fill_missing(d: dict, fill_value: Any = "") -> dict:
        return {k: (fill_value if v is None else v) for k, v in d.items()}

    @staticmethod
    def auto_cast_numeric(d: dict) -> dict:
        result = {}
        for k, v in d.items():
            try:
                if isinstance(v, str):
                    if v.isdigit():
                        result[k] = int(v)
                    else:
                        result[k] = float(v)
                else:
                    result[k] = v
            except:
                result[k] = v
        return result

    @staticmethod
    def parse_dates(d: dict, date_keys: List[str], date_format: Optional[str] = None) -> dict:
        for key in date_keys:
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = pd.to_datetime(d[key], format=date_format, errors='coerce')
                except:
                    d[key] = None
        return d

    @staticmethod
    def remove_duplicates_list(d: dict) -> dict:
        return {k: list(dict.fromkeys(v)) if isinstance(v, list) else v for k, v in d.items()}

    @staticmethod
    def conditional_transform(d: dict) -> dict:
        if 'status' in d:
            d['status'] = str(d['status']).upper()
        return d

    @staticmethod
    def regex_replace(d: dict, pattern_map: Dict[str, str]) -> dict:
        for k, pattern in pattern_map.items():
            if k in d and isinstance(d[k], str):
                d[k] = re.sub(pattern, '', d[k])
        return d

    @staticmethod
    def rename_keys(d: dict, rename_map: Dict[str, str]) -> dict:
        return {rename_map.get(k, k): v for k, v in d.items()}

    @staticmethod
    def remove_outliers(d: dict, z_thresh: float = 3.0) -> dict:
        numeric_values = {k: v for k, v in d.items() if isinstance(v, (int, float))}
        if numeric_values:
            mean = np.mean(list(numeric_values.values()))
            std = np.std(list(numeric_values.values()))
            for k, v in numeric_values.items():
                if std > 0 and abs(v - mean) / std > z_thresh:
                    d[k] = None
        return d

    @staticmethod
    def scale_numeric(d: dict, scale_map: Dict[str, float]) -> dict:
        for k, factor in scale_map.items():
            if k in d and isinstance(d[k], (int, float)):
                d[k] *= factor
        return d

    @staticmethod
    def encode_categorical(d: dict, categorical_keys: List[str]) -> dict:
        for key in categorical_keys:
            if key in d and isinstance(d[key], str):
                d[key] = hash(d[key]) % 10000  # simple encoding
        return d

    @staticmethod
    def conditional_numeric_transform(d: dict, transform_map: Dict[str, Any]) -> dict:
        for k, func in transform_map.items():
            if k in d and isinstance(d[k], (int, float)):
                d[k] = func(d[k])
        return d

    @staticmethod
    def validate_keys(d: dict, required_keys: List[str]) -> dict:
        for k in required_keys:
            if k not in d:
                d[k] = None
        return d

    @staticmethod
    def cast_types(d: dict, type_map: Dict[str, Any]) -> dict:
        for k, t in type_map.items():
            if k in d:
                try:
                    d[k] = t(d[k])
                except:
                    d[k] = None
        return d

    # ---------------- Main Cleaning Pipeline ----------------
    def clean(self,
              to_dataframe: bool = True,
              regex_map: Optional[Dict[str, str]] = None,
              rename_map: Optional[Dict[str, str]] = None,
              fill_value: Any = "",
              date_keys: Optional[List[str]] = None,
              scale_map: Optional[Dict[str, float]] = None,
              categorical_keys: Optional[List[str]] = None,
              numeric_transform_map: Optional[Dict[str, Any]] = None,
              required_keys: Optional[List[str]] = None,
              type_map: Optional[Dict[str, Any]] = None,
              flatten: bool = True) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Full YAML cleaning pipeline with 20+ steps.
        """
        try:
            data = self.data
            if flatten:
                data = self.flatten_dict(data)
                logger.info("YAML data flattened.")

            data = self.fill_missing(data, fill_value=fill_value)
            data = self.trim_strings(data)
            data = self.lowercase_strings(data)
            data = self.uppercase_strings(data)
            data = self.remove_special_characters(data)
            data = self.auto_cast_numeric(data)
            if date_keys:
                data = self.parse_dates(data, date_keys)
            data = self.remove_duplicates_list(data)
            data = self.conditional_transform(data)
            if regex_map:
                data = self.regex_replace(data, regex_map)
            if rename_map:
                data = self.rename_keys(data, rename_map)
            data = self.remove_outliers(data)
            if scale_map:
                data = self.scale_numeric(data, scale_map)
            if categorical_keys:
                data = self.encode_categorical(data, categorical_keys)
            if numeric_transform_map:
                data = self.conditional_numeric_transform(data, numeric_transform_map)
            if required_keys:
                data = self.validate_keys(data, required_keys)
            if type_map:
                data = self.cast_types(data, type_map)

            self.cleaned_dict = data
            if to_dataframe:
                self.cleaned_df = pd.DataFrame([data])
                return self.cleaned_df
            return data

        except Exception as e:
            logger.error(f"Error cleaning YAML data: {e}")
            raise


# ------------------ Usage Example ------------------
if __name__ == "__main__":
    sample_yaml = {
        "sensor": {"id": "001", "status": "active"},
        "metrics": {"temperature": "22.5", "humidity": None},
        "tags": ["IoT", "Sensor", "IoT"]
    }
    cleaner = YAMLCleaner(sample_yaml)
    cleaned_df = cleaner.clean(
        date_keys=[],
        scale_map={"metrics.temperature": 1.0},
        categorical_keys=["sensor.status"],
        required_keys=["sensor.id", "metrics.temperature"]
    )
    print(cleaned_df)
                           
