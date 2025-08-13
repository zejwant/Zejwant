# __init__.py
"""
Enterprise Ingestion Package

This package provides core ingestion modules for APIs, streaming data, databases,
and file uploads. It initializes logging and exposes a high-level `ingest_data` function
for routing ingestion based on source type.

Modules:
- api_connector
- live_stream_connector
- db_connector
- upload_pipeline
- ingest_scheduler
"""

from typing import Any, Dict, Union
import logging
import asyncio

# Initialize package-wide logging
logger = logging.getLogger("ingestion")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Import core ingestion modules
from . import api_connector
from . import live_stream_connector
from . import db_connector
from . import upload_pipeline
from . import ingest_scheduler

# Source type constants
API = "api"
STREAM = "stream"
DATABASE = "database"
FILE = "file"


async def ingest_data(
    source_type: str,
    config: Dict[str, Any],
    async_mode: bool = True
) -> Union[Dict[str, Any], Any]:
    """
    High-level function to route ingestion based on source type.

    Args:
        source_type (str): Type of data source ("api", "stream", "database", "file")
        config (Dict[str, Any]): Configuration dictionary for the connector
        async_mode (bool): Whether to run asynchronously (default True)

    Returns:
        Union[Dict[str, Any], Any]: Ingested data as Pandas DataFrame or JSON/dict

    Raises:
        ValueError: If source_type is unsupported
    """
    source_type = source_type.lower()
    logger.info(f"Starting ingestion for source type: {source_type}")

    if source_type == API:
        connector = api_connector.APIConnector(**config)
        if async_mode:
            return await connector.fetch()
        return connector.fetch_sync()

    elif source_type == STREAM:
        connector = live_stream_connector.KafkaConnector(**config)  # example, can extend for MQTT
        if async_mode:
            await connector.connect()
            data = await connector.consume(max_messages=config.get("max_messages"))
            await connector.close()
            return data
        raise NotImplementedError("Synchronous stream ingestion not supported")

    elif source_type == DATABASE:
        connector = db_connector.DBConnector(**config)
        if async_mode:
            return await connector.fetch()
        return connector.fetch_sync()

    elif source_type == FILE:
        connector = upload_pipeline.FileUploadPipeline(**config)
        return connector.process_files()

    else:
        logger.error(f"Unsupported source type: {source_type}")
        raise ValueError(f"Unsupported source type: {source_type}")
      
