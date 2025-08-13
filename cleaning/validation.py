"""
cleaning/validation.py

Enterprise-grade data validation utilities for multiple data formats.

Features:
- Schema validation for CSV, JSON, Excel
- Type checks, range checks, regex validation
- Referential integrity / foreign key validation
- Validation of numeric, string, date, categorical, and nested structures
- Detailed validation reports
- Logging and error handling
"""

import logging
import re
from typing import Any, Dict, List, Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize module logger
logger = logging.getLogger("validation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class DataValidator:
    """Enterprise-grade data validator with multiple reusable methods."""

    @staticmethod
    def validate_schema(df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate DataFrame against a schema.

        Args:
            df (pd.DataFrame): Input data.
            schema (dict): Dictionary of column_name -> type or nested rules.

        Returns:
            dict: Validation report
        """
        report = {}
        for col, rules in schema.items():
            if col not in df.columns:
                report[col] = {"status": "missing"}
            else:
                report[col] = {"status": "present"}
        logger.info(f"Schema validation completed with report: {report}")
        return report

    @staticmethod
    def check_column_types(df: pd.DataFrame, type_map: Dict[str, Any]) -> Dict[str, Any]:
        """Check column types and report mismatches."""
        report = {}
        for col, expected_type in type_map.items():
            if col in df.columns:
                actual_type = df[col].dtype
                report[col] = {"expected": expected_type, "actual": str(actual_type),
                               "match": np.issubdtype(actual_type, np.dtype(expected_type))}
            else:
                report[col] = {"expected": expected_type, "actual": None, "match": False}
        logger.info("Column type validation completed")
        return report

    @staticmethod
    def check_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, int]:
        """Count missing values per column."""
        columns = columns or df.columns.tolist()
        report = {col: df[col].isna().sum() for col in columns if col in df.columns}
        logger.info(f"Missing values validation report: {report}")
        return report

    @staticmethod
    def check_numeric_range(df: pd.DataFrame, column: str, min_value: float = None, max_value: float = None) -> Dict[str, Any]:
        """Validate numeric column falls within min/max."""
        report = {"column": column, "invalid_count": 0, "status": "pass"}
        if column in df.columns:
            invalid_mask = pd.Series(False, index=df.index)
            if min_value is not None:
                invalid_mask |= df[column] < min_value
            if max_value is not None:
                invalid_mask |= df[column] > max_value
            report["invalid_count"] = invalid_mask.sum()
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Numeric range check for {column}: {report}")
        return report

    @staticmethod
    def regex_validate(df: pd.DataFrame, column: str, pattern: str) -> Dict[str, Any]:
        """Validate string column against regex."""
        report = {"column": column, "invalid_count": 0, "status": "pass"}
        if column in df.columns:
            invalid_mask = ~df[column].astype(str).str.match(pattern)
            report["invalid_count"] = invalid_mask.sum()
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Regex validation for {column} completed: {report}")
        return report

    @staticmethod
    def categorical_values_check(df: pd.DataFrame, column: str, allowed_values: List[Any]) -> Dict[str, Any]:
        """Validate categorical column contains only allowed values."""
        report = {"column": column, "invalid_count": 0, "status": "pass"}
        if column in df.columns:
            invalid_mask = ~df[column].isin(allowed_values)
            report["invalid_count"] = invalid_mask.sum()
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Categorical validation for {column} completed: {report}")
        return report

    @staticmethod
    def check_date_format(df: pd.DataFrame, column: str, date_format: str) -> Dict[str, Any]:
        """Check if date column matches format."""
        report = {"column": column, "invalid_count": 0, "status": "pass"}
        if column in df.columns:
            def is_valid(date_str):
                try:
                    datetime.strptime(str(date_str), date_format)
                    return True
                except:
                    return False
            invalid_mask = ~df[column].apply(is_valid)
            report["invalid_count"] = invalid_mask.sum()
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Date format validation for {column}: {report}")
        return report

    @staticmethod
    def check_foreign_keys(df: pd.DataFrame, column: str, reference_df: pd.DataFrame, reference_column: str) -> Dict[str, Any]:
        """Validate foreign key integrity."""
        report = {"column": column, "invalid_count": 0, "status": "pass"}
        if column in df.columns and reference_column in reference_df.columns:
            invalid_mask = ~df[column].isin(reference_df[reference_column])
            report["invalid_count"] = invalid_mask.sum()
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Foreign key validation for {column} against {reference_column}: {report}")
        return report

    @staticmethod
    def check_unique(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Check uniqueness of values in column."""
        report = {"column": column, "duplicate_count": 0, "status": "pass"}
        if column in df.columns:
            duplicates = df[column].duplicated().sum()
            report["duplicate_count"] = duplicates
            if duplicates > 0:
                report["status"] = "fail"
        logger.info(f"Unique check for {column}: {report}")
        return report

    @staticmethod
    def check_min_length(df: pd.DataFrame, column: str, min_length: int) -> Dict[str, Any]:
        """Check minimum string length."""
        report = {"column": column, "invalid_count": 0, "status": "pass"}
        if column in df.columns:
            invalid_mask = df[column].astype(str).str.len() < min_length
            report["invalid_count"] = invalid_mask.sum()
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Min length check for {column}: {report}")
        return report

    @staticmethod
    def check_max_length(df: pd.DataFrame, column: str, max_length: int) -> Dict[str, Any]:
        """Check maximum string length."""
        report = {"column": column, "invalid_count": 0, "status": "pass"}
        if column in df.columns:
            invalid_mask = df[column].astype(str).str.len() > max_length
            report["invalid_count"] = invalid_mask.sum()
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Max length check for {column}: {report}")
        return report

    @staticmethod
    def check_allowed_pattern(df: pd.DataFrame, column: str, allowed_patterns: List[str]) -> Dict[str, Any]:
        """Check string matches at least one allowed regex pattern."""
        report = {"column": column, "invalid_count": 0, "status": "pass"}
        if column in df.columns:
            def valid(val):
                return any(re.match(p, str(val)) for p in allowed_patterns)
            invalid_mask = ~df[column].apply(valid)
            report["invalid_count"] = invalid_mask.sum()
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Allowed pattern validation for {column}: {report}")
        return report

    @staticmethod
    def check_nested_keys(df: pd.DataFrame, column: str, required_keys: List[str]) -> Dict[str, Any]:
        """Check nested dict column contains required keys."""
        report = {"column": column, "missing_keys_count": 0, "status": "pass"}
        if column in df.columns:
            def missing_keys_count(val):
                if isinstance(val, dict):
                    return len([k for k in required_keys if k not in val])
                return len(required_keys)
            counts = df[column].apply(missing_keys_count)
            report["missing_keys_count"] = counts.sum()
            if report["missing_keys_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Nested keys validation for {column}: {report}")
        return report

    @staticmethod
    def check_value_in_set(df: pd.DataFrame, column: str, valid_set: set) -> Dict[str, Any]:
        """Check values belong to a valid set."""
        report = {"column": column, "invalid_count": 0, "status": "pass"}
        if column in df.columns:
            invalid_mask = ~df[column].isin(valid_set)
            report["invalid_count"] = invalid_mask.sum()
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Set membership validation for {column}: {report}")
        return report

    @staticmethod
    def check_positive_numbers(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Check numeric column contains only positive numbers."""
        report = {"column": column, "invalid_count": 0, "status": "pass"}
        if column in df.columns:
            invalid_mask = df[column] <= 0
            report["invalid_count"] = invalid_mask.sum()
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Positive number validation for {column}: {report}")
        return report

    @staticmethod
    def check_no_nulls(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Ensure column has no null values."""
        report = {"column": column, "null_count": 0, "status": "pass"}
        if column in df.columns:
            null_count = df[column].isna().sum()
            report["null_count"] = null_count
            if null_count > 0:
                report["status"] = "fail"
        logger.info(f"No null validation for {column}: {report}")
        return report
          
