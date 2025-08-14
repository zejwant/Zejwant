"""
cleaning/__init__.py

Enterprise-grade data cleaning package initializer.

Features:
- Logging and metrics initialization
- Imports core and format-specific cleaning modules
- Provides `clean_data()` as a high-level entry point
"""

import logging
import time
from typing import Any, Dict, Union

# Core cleaning modules
from .validation import DataValidatorV1
from cleaning.validation import validate_data
from .cleaner_core import CleanerCore
validator = DataValidatorV1()
report = validator.batch_validate(df, schema_rules)

from .validation import validate_data
from .deduplication import deduplicate_data
from .cleaner_controller import CleanerController

# Format-specific cleaners
from .format_cleaners.csv_cleaner import clean_csv
from .format_cleaners.excel_cleaner import clean_excel
from .format_cleaners.json_cleaner import clean_json
from .format_cleaners.xml_cleaner import clean_xml
from .format_cleaners.pdf_cleaner import clean_pdf
from .format_cleaners.log_cleaner import clean_log
from .format_cleaners.parquet_cleaner import clean_parquet
from .format_cleaners.avro_cleaner import clean_avro
from .format_cleaners.orc_cleaner import clean_orc
from .format_cleaners.google_sheets_cleaner import clean_google_sheets
from .format_cleaners.html_scraper_cleaner import clean_html_scraper
from .format_cleaners.sensor_iot_cleaner import clean_sensor_iot
from .format_cleaners.yaml_cleaner import clean_yaml
from .format_cleaners.kafka_stream_cleaner import clean_kafka_stream
from .format_cleaners.mqtt_stream_cleaner import clean_mqtt_stream
from .format_cleaners.image_cleaner import clean_image
from .format_cleaners.video_metadata_cleaner import clean_video_metadata
from .format_cleaners.audio_metadata_cleaner import clean_audio_metadata



# Initialize logging
logger = logging.getLogger("cleaning")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Metrics placeholder (can be replaced with Prometheus, Datadog, etc.)
metrics = {
    "total_cleaned": 0,
    "format_counts": {}
}


def clean_data(data: Any, data_format: str, **kwargs) -> Any:
    """
    High-level function to clean data based on its format.

    Args:
        data (Any): Input data to be cleaned. Can be pandas DataFrame, dict, str, or file-like object.
        data_format (str): Data format identifier. Supported: 'csv', 'excel', 'json', 'xml', 'pdf', 'log'.
        **kwargs: Optional arguments passed to format-specific cleaners.

    Returns:
        Any: Cleaned data in the same structure as input.

    Raises:
        ValueError: If data_format is unsupported.
    """
    start_time = time.time()
    data_format = data_format.lower()

    format_cleaners = {
    "csv": clean_csv,
    "excel": clean_excel,
    "json": clean_json,
    "xml": clean_xml,
    "pdf": clean_pdf,
    "log": clean_log,
    "parquet": clean_parquet,
    "avro": clean_avro,
    "orc": clean_orc,
    "google_sheets": clean_google_sheets,
    "html": clean_html_scraper,
    "iot": clean_sensor_iot,
    "yaml": clean_yaml,
    "kafka": clean_kafka_stream,
    "mqtt": clean_mqtt_stream,
    "image": clean_image,
    "video": clean_video_metadata,
    "audio": clean_audio_metadata,
    }
    
    if data_format not in format_cleaners:
        logger.error(f"Unsupported data format: {data_format}")
        raise ValueError(f"Unsupported data format: {data_format}")

    logger.info(f"Cleaning data with format: {data_format}")
    cleaned_data = format_cleaners[data_format](data, **kwargs)

    # Update metrics
    metrics["total_cleaned"] += 1
    metrics["format_counts"][data_format] = metrics["format_counts"].get(data_format, 0) + 1

    elapsed_time = time.time() - start_time
    logger.info(f"Data cleaned in {elapsed_time:.3f} seconds")

    return cleaned_data

   data_format = ext_map.get(data_format.lower(), data_format.lower())

   ext_map = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".xml": "xml",
    ".pdf": "pdf",
    ".log": "log",
    ".parquet": "parquet",
    ".avro": "avro",
    ".orc": "orc",
    ".gsheet": "google_sheets",
    ".html": "html",
    ".iot": "iot",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".kafka": "kafka",
    ".mqtt": "mqtt",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".mp4": "video",
    ".mov": "video",
    ".mp3": "audio",
    ".wav": "audio",
   }
