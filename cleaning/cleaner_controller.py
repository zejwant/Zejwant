"""
cleaner_controller.py
Unified Enterprise-grade Cleaning Orchestrator for 30+ formats.

Features:
- Auto-detect file, stream, IoT, media formats
- Route to individual cleaners
- Core cleaning, validation, deduplication integration
- Logging, metrics, retries
- Detailed cleaning report
- Ready for ETL pipelines
"""

import logging
import time
from typing import Any, Dict, Optional, List
import pandas as pd

# --- Core / Validation / Dedup modules ---
from .cleaner_core import CleanerCore
from .validation import DataValidator
from .deduplication import Deduplicator

# --- Format-specific cleaners ---
from .format_cleaners.csv_cleaner import clean_csv
from .format_cleaners.excel_cleaner import clean_excel
from .format_cleaners.json_cleaner import clean_json
from .format_cleaners.xml_cleaner import clean_xml
from .format_cleaners.pdf_cleaner import clean_pdf
from .format_cleaners.log_cleaner import clean_log
from .format_cleaners.parquet_cleaner import clean_parquet
from .format_cleaners.avro_cleaner import clean_avro
from .format_cleaners.orc_cleaner import clean_orc
from .format_cleaners.yaml_cleaner import clean_yaml
from .format_cleaners.html_scraper_cleaner import clean_html
from .format_cleaners.google_sheets_cleaner import clean_google_sheets
from .format_cleaners.audio_metadata_cleaner import AudioMetadataCleaner
from .format_cleaners.video_metadata_cleaner import VideoMetadataCleaner
from .format_cleaners.image_cleaner import ImageCleaner
from .format_cleaners.sensor_iot_cleaner import SensorIoTCleaner
from .format_cleaners.mqtt_stream_cleaner import MQTTStreamCleaner
from .format_cleaners.kafka_stream_cleaner import KafkaStreamCleaner

# ---------------- Logger ----------------
logger = logging.getLogger("UnifiedCleanerController")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class UnifiedCleanerController:
    """
    Enterprise-grade orchestrator for all file, stream, IoT, and media formats.
    """

    def __init__(self, retry_attempts: int = 3):
        self.retry_attempts = retry_attempts
        self.metrics = {
            "format_cleaning": 0,
            "core_cleaning": 0,
            "validation": 0,
            "deduplication": 0,
            "total_cleaned": 0,
        }

    # ---------------- Format Detection ----------------
    def detect_format(self, data: Any, file_name: Optional[str] = None) -> str:
        """Auto-detect format by type or file extension."""
        if isinstance(data, pd.DataFrame):
            return "dataframe"
        if file_name:
            ext = file_name.split('.')[-1].lower()
            mapping = {
                "csv": "csv", "xlsx": "excel", "xls": "excel", "json": "json",
                "xml": "xml", "pdf": "pdf", "log": "log", "parquet": "parquet",
                "avro": "avro", "orc": "orc", "yaml": "yaml", "yml": "yaml",
                "html": "html", "gsheet": "google_sheets"
            }
            if ext in mapping:
                return mapping[ext]
        if isinstance(data, dict):
            return "yaml"  # fallback for structured dict
        if hasattr(data, "stream_type"):
            return getattr(data, "stream_type")  # IoT or stream data
        return "raw"

    # ---------------- Cleaner Routing ----------------
    def route_to_cleaner(self, data: Any, data_format: str, **kwargs) -> Any:
        """Route data to corresponding cleaner."""
        cleaners = {
            "csv": clean_csv,
            "excel": clean_excel,
            "json": clean_json,
            "xml": clean_xml,
            "pdf": clean_pdf,
            "log": clean_log,
            "parquet": clean_parquet,
            "avro": clean_avro,
            "orc": clean_orc,
            "yaml": clean_yaml,
            "html": clean_html,
            "google_sheets": clean_google_sheets,
            "dataframe": lambda x, **kw: x,
            "raw": lambda x, **kw: x,
            "audio": AudioMetadataCleaner,
            "video": VideoMetadataCleaner,
            "image": ImageCleaner,
            "sensor_iot": SensorIoTCleaner,
            "mqtt_stream": MQTTStreamCleaner,
            "kafka_stream": KafkaStreamCleaner
        }

        if data_format not in cleaners:
            raise ValueError(f"No cleaner for format: {data_format}")

        attempts = 0
        while attempts < self.retry_attempts:
            try:
                cleaner = cleaners[data_format]
                if callable(cleaner):
                    if data_format in ["audio", "video", "image", "sensor_iot", "mqtt_stream", "kafka_stream"]:
                        # Instantiate class-based cleaner
                        return cleaner(data).clean(**kwargs)
                    return cleaner(data, **kwargs)
                else:
                    raise TypeError(f"Cleaner is not callable: {data_format}")
            except Exception as e:
                attempts += 1
                logger.error(f"Cleaner failed (attempt {attempts}) for {data_format}: {e}")
                if attempts >= self.retry_attempts:
                    raise

    # ---------------- Core Cleaning ----------------
    def run_core_cleaning(self, df: pd.DataFrame, steps: Optional[List[str]] = None) -> pd.DataFrame:
        steps = steps or ["trim_strings", "lowercase_columns", "remove_extra_whitespace", "remove_special_characters", "fill_missing"]
        for step in steps:
            func = getattr(CleanerCore, step, None)
            if callable(func):
                df = func(df, df.columns.tolist())
                self.metrics["core_cleaning"] += 1
        return df

    # ---------------- Validation ----------------
    def run_validation(self, df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        report = {}
        if schema:
            report["schema"] = DataValidator.validate_schema(df, schema)
            self.metrics["validation"] += 1
        return report

    # ---------------- Deduplication ----------------
    def run_deduplication(self, df: pd.DataFrame, strategy: str = "drop_exact_duplicates",
                          columns: Optional[List[str]] = None) -> Dict[str, Any]:
        func = getattr(Deduplicator, strategy, None)
        if callable(func):
            result = func(df, subset=columns)
            self.metrics["deduplication"] += 1
            return result
        return {"data": df, "removed": 0}

    # ---------------- Full Cleaning Pipeline ----------------
    def clean_data(self, data: Any, file_name: Optional[str] = None,
                   schema: Optional[Dict[str, Any]] = None,
                   dedup_strategy: str = "drop_exact_duplicates",
                   dedup_columns: Optional[List[str]] = None,
                   core_steps: Optional[List[str]] = None,
                   **kwargs) -> Dict[str, Any]:
        start = time.time()
        report = {"steps": {}, "metrics": {}, "errors": []}

        try:
            data_format = self.detect_format(data, file_name)
            df = self.route_to_cleaner(data, data_format, **kwargs)
            report["steps"]["format_cleaning"] = f"Format cleaned: {data_format}"
            self.metrics["format_cleaning"] += 1
        except Exception as e:
            report["errors"].append({"step": "format_cleaning", "error": str(e)})
            return report

        try:
            if isinstance(df, pd.DataFrame):
                df = self.run_core_cleaning(df, core_steps)
                report["steps"]["core_cleaning"] = "Core cleaning applied"
        except Exception as e:
            report["errors"].append({"step": "core_cleaning", "error": str(e)})

        try:
            if isinstance(df, pd.DataFrame) and schema:
                validation_report = self.run_validation(df, schema)
                report["steps"]["validation"] = validation_report
        except Exception as e:
            report["errors"].append({"step": "validation", "error": str(e)})

        try:
            if isinstance(df, pd.DataFrame) and dedup_strategy:
                dedup_result = self.run_deduplication(df, dedup_strategy, dedup_columns)
                df = dedup_result["data"]
                report["steps"]["deduplication"] = dedup_result
        except Exception as e:
            report["errors"].append({"step": "deduplication", "error": str(e)})

        report["metrics"] = self.metrics.copy()
        report["metrics"]["elapsed_time_sec"] = time.time() - start
        report["cleaned_data"] = df
        self.metrics["total_cleaned"] += 1

        return report
                      
