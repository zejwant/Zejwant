"""
cleaning/format_cleaners/__init__.py

Format-specific cleaners subpackage initializer.

Features:
- Dynamically imports all format-specific cleaner modules
- Exposes a factory function to retrieve cleaner by file type
"""

import importlib
import pkgutil
from typing import Callable, Dict, Optional

# Dynamically import all modules in this package
__all__ = []
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f".{module_name}", package=__name__)
    __all__.append(module_name)

# Map file types to cleaner function names
_CLEANER_MAP = {
    "csv": "clean_csv",
    "excel": "clean_excel",
    "json": "clean_json",
    "xml": "clean_xml",
    "pdf": "clean_pdf",
    "log": "clean_log",
    "parquet": "clean_parquet",
    "avro": "clean_avro",
    "orc": "clean_orc",
    "google_sheets": "clean_google_sheets",
    "html": "clean_html_scraper",
    "iot": "clean_sensor_iot",
    "yaml": "clean_yaml",
    "kafka": "clean_kafka_stream",
    "mqtt": "clean_mqtt_stream",
    "image": "clean_image",
    "video": "clean_video_metadata",
    "audio": "clean_audio_metadata",
}

# Dictionary to store actual callable cleaners
_CLEANERS: Dict[str, Callable] = {}

for module_name in __all__:
    module = importlib.import_module(f".{module_name}", package=__name__)
    for ft, cleaner_name in _CLEANER_MAP.items():
        if hasattr(module, cleaner_name):
            _CLEANERS[ft] = getattr(module, cleaner_name)

def get_cleaner(file_type: str) -> Optional[Callable]:
    """
    Factory function to get cleaner function by file type.
    """
    return _CLEANERS.get(file_type.lower())

# Optional: detect cleaner by filename
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

def get_cleaner_by_filename(filename: str) -> Optional[Callable]:
    ext = "." + filename.split(".")[-1].lower()
    file_type = ext_map.get(ext)
    if not file_type:
        return None
    return get_cleaner(file_type)
    
