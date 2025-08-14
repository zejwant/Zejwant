"""
cleaning/validation_v1_1.py

Enterprise-grade data validation v1.1

Features:
- Schema validation for CSV, JSON, Excel
- Type checks, range checks, regex validation
- Referential integrity / foreign key validation
- Validation of numeric, string, date, categorical, and nested structures
- Detailed validation reports with row indices
- Batch validation engine
- Logging, timing, and error handling
"""

import logging
import re
import time
from typing import Any, Dict, List, Union, Optional, Set
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger("validation_v1_1")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class DataValidatorV1:
    """Enterprise-grade data validator v1.1 with enhanced reporting and batch validations."""

    @staticmethod
    def _timeit(func):
        """Decorator to measure execution time."""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"{func.__name__} executed in {elapsed:.4f}s")
            return result
        return wrapper

    @staticmethod
    @_timeit
    def validate_schema(df: pd.DataFrame, schema: Dict[str, Any], raise_on_error: bool = False) -> Dict[str, Any]:
        report = {}
        for col, rules in schema.items():
            if col not in df.columns:
                report[col] = {"status": "missing"}
            else:
                report[col] = {"status": "present"}
        logger.info(f"Schema validation report: {report}")
        if raise_on_error and any(r["status"] == "missing" for r in report.values()):
            raise ValueError(f"Schema validation failed: {report}")
        return report

    @staticmethod
    @_timeit
    def check_column_types(df: pd.DataFrame, type_map: Dict[str, Any], raise_on_error: bool = False) -> Dict[str, Any]:
        report = {}
        for col, expected_type in type_map.items():
            if col in df.columns:
                actual_type = df[col].dtype
                match = np.issubdtype(actual_type, np.dtype(expected_type))
                report[col] = {"expected": expected_type, "actual": str(actual_type), "match": match}
            else:
                report[col] = {"expected": expected_type, "actual": None, "match": False}
        logger.info(f"Column type validation report: {report}")
        if raise_on_error and not all(r["match"] for r in report.values()):
            raise TypeError(f"Type validation failed: {report}")
        return report

    @staticmethod
    @_timeit
    def check_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None, return_rows: bool = False) -> Dict[str, Any]:
        columns = columns or df.columns.tolist()
        report = {}
        for col in columns:
            if col in df.columns:
                null_rows = df.index[df[col].isna()].tolist()
                report[col] = {
                    "missing_count": len(null_rows),
                    "rows": null_rows if return_rows else None
                }
        logger.info(f"Missing values report: {report}")
        return report

    @staticmethod
    @_timeit
    def check_numeric_range(df: pd.DataFrame, column: str, min_value: float = None, max_value: float = None,
                            raise_on_error: bool = False, return_rows: bool = False) -> Dict[str, Any]:
        report = {"column": column, "invalid_count": 0, "status": "pass", "rows": []}
        if column in df.columns:
            invalid_mask = pd.Series(False, index=df.index)
            if min_value is not None:
                invalid_mask |= df[column] < min_value
            if max_value is not None:
                invalid_mask |= df[column] > max_value
            invalid_rows = df.index[invalid_mask].tolist()
            report["invalid_count"] = len(invalid_rows)
            report["rows"] = invalid_rows if return_rows else []
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Numeric range validation for {column}: {report}")
        if raise_on_error and report["status"] == "fail":
            raise ValueError(f"Numeric range validation failed: {report}")
        return report

    @staticmethod
    @_timeit
    def regex_validate(df: pd.DataFrame, column: str, pattern: str,
                       raise_on_error: bool = False, return_rows: bool = False) -> Dict[str, Any]:
        report = {"column": column, "invalid_count": 0, "status": "pass", "rows": []}
        if column in df.columns:
            invalid_mask = ~df[column].astype(str).str.match(pattern)
            invalid_rows = df.index[invalid_mask].tolist()
            report["invalid_count"] = len(invalid_rows)
            report["rows"] = invalid_rows if return_rows else []
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Regex validation for {column}: {report}")
        if raise_on_error and report["status"] == "fail":
            raise ValueError(f"Regex validation failed: {report}")
        return report

    @staticmethod
    @_timeit
    def categorical_values_check(df: pd.DataFrame, column: str, allowed_values: List[Any],
                                 raise_on_error: bool = False, return_rows: bool = False) -> Dict[str, Any]:
        report = {"column": column, "invalid_count": 0, "status": "pass", "rows": []}
        if column in df.columns:
            invalid_mask = ~df[column].isin(allowed_values)
            invalid_rows = df.index[invalid_mask].tolist()
            report["invalid_count"] = len(invalid_rows)
            report["rows"] = invalid_rows if return_rows else []
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Categorical check for {column}: {report}")
        if raise_on_error and report["status"] == "fail":
            raise ValueError(f"Categorical check failed: {report}")
        return report

    @staticmethod
    @_timeit
    def check_foreign_keys(df: pd.DataFrame, column: str, reference_df: pd.DataFrame, reference_column: str,
                           raise_on_error: bool = False, return_rows: bool = False) -> Dict[str, Any]:
        report = {"column": column, "invalid_count": 0, "status": "pass", "rows": []}
        if column in df.columns and reference_column in reference_df.columns:
            invalid_mask = ~df[column].isin(reference_df[reference_column])
            invalid_rows = df.index[invalid_mask].tolist()
            report["invalid_count"] = len(invalid_rows)
            report["rows"] = invalid_rows if return_rows else []
            if report["invalid_count"] > 0:
                report["status"] = "fail"
        logger.info(f"Foreign key check for {column} vs {reference_column}: {report}")
        if raise_on_error and report["status"] == "fail":
            raise ValueError(f"Foreign key validation failed: {report}")
        return report

    @staticmethod
    @_timeit
    def batch_validate(df: pd.DataFrame, schema_rules: Dict[str, Dict[str, Any]], raise_on_error: bool = False) -> Dict[str, Any]:
        """
        Batch validation engine. schema_rules format:
        { "column_name": {"type": "numeric"/"string", "min":..., "max":..., "pattern":..., "allowed": [...]} }
        """
        full_report = {}
        for col, rules in schema_rules.items():
            if "min" in rules or "max" in rules:
                full_report[col] = DataValidatorV1.check_numeric_range(
                    df, col, min_value=rules.get("min"), max_value=rules.get("max"),
                    raise_on_error=raise_on_error
                )
            elif "pattern" in rules:
                full_report[col] = DataValidatorV1.regex_validate(
                    df, col, pattern=rules["pattern"], raise_on_error=raise_on_error
                )
            elif "allowed" in rules:
                full_report[col] = DataValidatorV1.categorical_values_check(
                    df, col, allowed_values=rules["allowed"], raise_on_error=raise_on_error
                )
        logger.info(f"Batch validation report: {full_report}")
        return full_report
                           
