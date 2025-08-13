# connectors/__init__.py
"""
Connectors subpackage.

This module dynamically imports all available connector modules
and exposes a factory function `get_connector` to instantiate
the appropriate connector class based on file type or data source.

Supported types:
csv, excel, json, xml, pdf, parquet, avro, orc,
google_sheets, html, log, sensor_iot, yaml,
kafka, mqtt, image, video, audio
"""

from importlib import import_module
from typing import Any, Type, Dict

# Mapping of file/data type to module and class names
_CONNECTOR_MAP: Dict[str, str] = {
    "csv": "csv_connector.CSVConnector",
    "excel": "excel_connector.ExcelConnector",
    "json": "json_connector.JSONConnector",
    "xml": "xml_connector.XMLConnector",
    "pdf": "pdf_connector.PDFConnector",
    "parquet": "parquet_connector.ParquetConnector",
    "avro": "avro_connector.AvroConnector",
    "orc": "orc_connector.ORCConnector",
    "google_sheets": "google_sheets_connector.GoogleSheetsConnector",
    "html": "html_connector.HTMLConnector",
    "log": "log_connector.LogConnector",
    "sensor_iot": "sensor_iot_connector.SensorIoTConnector",
    "yaml": "yaml_connector.YAMLConnector",
    "kafka": "kafka_connector.KafkaConnector",
    "mqtt": "mqtt_connector.MQTTConnector",
    "image": "image_connector.ImageConnector",
    "video": "video_connector.VideoConnector",
    "audio": "audio_connector.AudioConnector",
}

def get_connector(file_type: str, *args, **kwargs) -> Any:
    """
    Factory function to return an instance of the appropriate connector class.

    Args:
        file_type (str): Type of file or data source (e.g., 'csv', 'json', 'kafka')
        *args: Positional arguments passed to the connector constructor
        **kwargs: Keyword arguments passed to the connector constructor

    Returns:
        Any: Instance of the requested connector class

    Raises:
        ValueError: If the requested file_type is unsupported
        ImportError: If the connector module cannot be imported
        AttributeError: If the connector class does not exist in the module
    """
    file_type = file_type.lower()
    if file_type not in _CONNECTOR_MAP:
        raise ValueError(f"Unsupported connector type: {file_type}")

    module_class = _CONNECTOR_MAP[file_type]
    module_name, class_name = module_class.rsplit(".", 1)

    try:
        module = import_module(f".{module_name}", package=__name__)
    except ImportError as e:
        raise ImportError(f"Failed to import connector module '{module_name}': {e}")

    try:
        connector_cls: Type = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"Connector class '{class_name}' not found in module '{module_name}'")

    return connector_cls(*args, **kwargs)


# Optional: dynamically import all connectors at package level
__all__ = list(_CONNECTOR_MAP.keys()) + ["get_connector"]
