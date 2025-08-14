# storage/connectors/__init__.py

"""
Connectors Package
-----------------
Enterprise-grade module for connecting to external data warehouses.

Features:
- Import BigQuery, Snowflake, Redshift connectors
- Provide dynamic connector selection based on configuration
- Type hints and docstrings for maintainability
"""

from typing import Dict, Any, Optional, Union
from .bigquery_connector import BigQueryConnector
from .snowflake_connector import SnowflakeConnector
from .redshift_connector import RedshiftConnector


def get_connector(config: Dict[str, Any]) -> Union[BigQueryConnector, SnowflakeConnector, RedshiftConnector]:
    """
    Dynamically select and return a warehouse connector based on configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary. Must include key 'type' with values:
            - "bigquery"
            - "snowflake"
            - "redshift"

    Returns:
        Connector instance corresponding to the warehouse type.

    Raises:
        ValueError: If connector type is unsupported.
    """
    connector_type = config.get("type")
    if connector_type == "bigquery":
        return BigQueryConnector(config)
    elif connector_type == "snowflake":
        return SnowflakeConnector(config)
    elif connector_type == "redshift":
        return RedshiftConnector(config)
    else:
        raise ValueError(f"Unsupported connector type: {connector_type}")
      
