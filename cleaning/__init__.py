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
from .cleaner_core import CleanerCore
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
  
