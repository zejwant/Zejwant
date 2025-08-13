# xml_connector.py
"""
XML Connector for enterprise-grade ingestion pipelines.

Features:
- Parse XML files and remote XML responses
- Convert XML tree into Pandas DataFrame
- Handle namespaces and attributes
- Schema validation and error handling
- Logging
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import xml.etree.ElementTree as ET
import requests
import logging
import os

# Logging setup
logger = logging.getLogger("xml_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class XMLConnector:
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize XML Connector.

        Args:
            schema (Dict[str, Any], optional): Expected column types for validation
        """
        self.schema = schema

    def read(
        self,
        source: str,
        remote: bool = False,
        namespace: Optional[Dict[str, str]] = None,
        record_tag: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read XML data from local file or remote URL and convert to DataFrame.

        Args:
            source (str): File path or URL
            remote (bool): If True, read from remote URL
            namespace (Dict[str,str], optional): XML namespace mapping
            record_tag (str, optional): Tag name for repeating records

        Returns:
            pd.DataFrame: Flattened DataFrame

        Raises:
            FileNotFoundError: If local file not found
            ValueError: If schema validation fails
        """
        try:
            if remote:
                logger.info(f"Fetching XML from remote source: {source}")
                resp = requests.get(source)
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
            else:
                if not os.path.exists(source):
                    raise FileNotFoundError(f"File not found: {source}")
                logger.info(f"Reading local XML file: {source}")
                tree = ET.parse(source)
                root = tree.getroot()

            df = self._xml_to_dataframe(root, namespace=namespace, record_tag=record_tag)
            df = self._validate_schema(df)

            logger.info(f"XML loaded successfully: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to read XML from '{source}': {e}")
            raise

    def _xml_to_dataframe(
        self,
        root: ET.Element,
        namespace: Optional[Dict[str, str]] = None,
        record_tag: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert XML tree to Pandas DataFrame.

        Args:
            root (ET.Element): Root element of XML
            namespace (Dict[str,str], optional): XML namespaces
            record_tag (str, optional): Tag name for repeating records

        Returns:
            pd.DataFrame: Flattened DataFrame
        """
        records: List[Dict[str, Any]] = []

        def _element_to_dict(element: ET.Element) -> Dict[str, Any]:
            data: Dict[str, Any] = {}
            # Attributes
            for k, v in element.attrib.items():
                data[f"@{k}"] = v
            # Child elements
            for child in element:
                tag = child.tag.split("}")[-1]  # remove namespace
                if list(child):  # has children
                    data[tag] = _element_to_dict(child)
                else:
                    data[tag] = child.text
            return data

        # If record_tag provided, iterate over those elements
        elements = root.findall(f".//{record_tag}", namespace) if record_tag else [root]
        for el in elements:
            records.append(_element_to_dict(el))

        df = pd.json_normalize(records)
        return df

    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate DataFrame schema against expected schema.

        Args:
            df (pd.DataFrame): DataFrame to validate

        Returns:
            pd.DataFrame: Validated DataFrame

        Raises:
            ValueError: If schema validation fails
        """
        if not self.schema:
            return df

        for col, dtype in self.schema.items():
            if col not in df.columns:
                raise ValueError(f"Missing expected column: {col}")
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                raise ValueError(f"Failed to cast column '{col}' to {dtype}: {e}")
        return df
  
