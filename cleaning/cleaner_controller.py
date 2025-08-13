"""
cleaning/cleaner_controller.py

Enterprise-grade cleaning workflow orchestrator.

Features:
- Automatic format detection (CSV, Excel, JSON, XML, PDF, Log)
- Routing to format-specific cleaners
- Core cleaning, validation, deduplication integration
- Logging, error handling, metrics, retries
- Detailed cleaning report generation
"""

import logging
import time
from typing import Any, Dict, Optional, List, Union
import pandas as pd

from .cleaner_core import CleanerCore
from .validation import DataValidator
from .deduplication import Deduplicator
from .format_cleaners.csv_cleaner import clean_csv
from .format_cleaners.excel_cleaner import clean_excel
from .format_cleaners.json_cleaner import clean_json
from .format_cleaners.xml_cleaner import clean_xml
from .format_cleaners.pdf_cleaner import clean_pdf
from .format_cleaners.log_cleaner import clean_log

# Logger setup
logger = logging.getLogger("cleaner_controller")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class CleanerController:
    """
    Orchestrates the full cleaning workflow for enterprise-grade data pipelines.
    """

    def __init__(self, retry_attempts: int = 3):
        self.retry_attempts = retry_attempts
        self.metrics = {
            "core_cleaning_steps": 0,
            "validation_steps": 0,
            "deduplication_steps": 0,
            "total_cleaned": 0,
        }

    def detect_format(self, data: Any, file_name: Optional[str] = None) -> str:
        """
        Auto-detect data format.
        Args:
            data: Data content or DataFrame
            file_name: Optional file name to infer format
        Returns:
            format string: 'csv', 'excel', 'json', 'xml', 'pdf', 'log'
        """
        if isinstance(data, pd.DataFrame):
            logger.info("Detected format: DataFrame")
            return "dataframe"
        if file_name:
            ext = file_name.split('.')[-1].lower()
            mapping = {
                "csv": "csv",
                "xlsx": "excel",
                "xls": "excel",
                "json": "json",
                "xml": "xml",
                "pdf": "pdf",
                "log": "log",
            }
            if ext in mapping:
                logger.info(f"Detected format from file extension: {ext}")
                return mapping[ext]
        logger.warning("Unable to detect format, defaulting to raw")
        return "raw"

    def route_to_cleaner(self, data: Any, data_format: str, **kwargs) -> Any:
        """
        Route data to the corresponding format-specific cleaner.
        """
        cleaners = {
            "csv": clean_csv,
            "excel": clean_excel,
            "json": clean_json,
            "xml": clean_xml,
            "pdf": clean_pdf,
            "log": clean_log,
            "dataframe": lambda x, **kw: x,  # already loaded
            "raw": lambda x, **kw: x,
        }

        if data_format not in cleaners:
            logger.error(f"No cleaner available for format: {data_format}")
            raise ValueError(f"Unsupported data format: {data_format}")

        attempts = 0
        while attempts < self.retry_attempts:
            try:
                logger.info(f"Running format-specific cleaner for: {data_format}")
                cleaned_data = cleaners[data_format](data, **kwargs)
                return cleaned_data
            except Exception as e:
                attempts += 1
                logger.error(f"Cleaner failed on attempt {attempts} for {data_format}: {e}")
                if attempts >= self.retry_attempts:
                    raise

    def run_core_cleaning(self, df: pd.DataFrame, cleaning_steps: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply core cleaning functions sequentially.
        """
        cleaning_steps = cleaning_steps or [
            "trim_strings", "lowercase_columns", "remove_extra_whitespace",
            "remove_special_characters", "fill_missing"
        ]
        for step in cleaning_steps:
            func = getattr(CleanerCore, step, None)
            if callable(func):
                df = func(df, df.columns.tolist())
                self.metrics["core_cleaning_steps"] += 1
                logger.info(f"Applied core cleaning step: {step}")
        return df

    def run_validation(self, df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run validation checks and return report.
        """
        report = {}
        if schema:
            report["schema"] = DataValidator.validate_schema(df, schema)
            self.metrics["validation_steps"] += 1
        # Add other generic validations if needed
        self.metrics["validation_steps"] += 1
        return report

    def run_deduplication(self, df: pd.DataFrame, dedup_strategy: str = "drop_exact_duplicates",
                          columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run deduplication using specified strategy.
        """
        func = getattr(Deduplicator, dedup_strategy, None)
        if callable(func):
            result = func(df, subset=columns)
            self.metrics["deduplication_steps"] += 1
            logger.info(f"Deduplication applied: {dedup_strategy}")
            return result
        else:
            logger.warning(f"Deduplication strategy not found: {dedup_strategy}")
            return {"data": df, "removed": 0}

    def clean_data(self, data: Any, file_name: Optional[str] = None,
                   schema: Optional[Dict[str, Any]] = None,
                   dedup_strategy: str = "drop_exact_duplicates",
                   dedup_columns: Optional[List[str]] = None,
                   core_steps: Optional[List[str]] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Full cleaning pipeline:
        1. Format detection
        2. Format-specific cleaning
        3. Core cleaning
        4. Validation
        5. Deduplication
        Returns a detailed cleaning report.
        """
        start_time = time.time()
        report = {"steps": {}, "metrics": {}, "errors": []}

        try:
            data_format = self.detect_format(data, file_name)
            df = self.route_to_cleaner(data, data_format, **kwargs)
            report["steps"]["format_cleaning"] = f"Format cleaned: {data_format}"
        except Exception as e:
            report["errors"].append({"step": "format_cleaning", "error": str(e)})
            logger.error(f"Format cleaning failed: {e}")
            return report

        try:
            if isinstance(df, pd.DataFrame):
                df = self.run_core_cleaning(df, core_steps)
                report["steps"]["core_cleaning"] = "Core cleaning applied"
        except Exception as e:
            report["errors"].append({"step": "core_cleaning", "error": str(e)})
            logger.error(f"Core cleaning failed: {e}")

        try:
            if isinstance(df, pd.DataFrame) and schema:
                validation_report = self.run_validation(df, schema)
                report["steps"]["validation"] = validation_report
        except Exception as e:
            report["errors"].append({"step": "validation", "error": str(e)})
            logger.error(f"Validation failed: {e}")

        try:
            if isinstance(df, pd.DataFrame) and dedup_strategy:
                dedup_result = self.run_deduplication(df, dedup_strategy, dedup_columns)
                df = dedup_result["data"]
                report["steps"]["deduplication"] = dedup_result
        except Exception as e:
            report["errors"].append({"step": "deduplication", "error": str(e)})
            logger.error(f"Deduplication failed: {e}")

        report["metrics"] = self.metrics.copy()
        report["metrics"]["elapsed_time_sec"] = time.time() - start_time
        report["cleaned_data"] = df

        self.metrics["total_cleaned"] += 1
        logger.info("Cleaning pipeline completed successfully")

        return report
                     
