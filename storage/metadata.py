# storage/metadata.py

"""
Metadata Management
-------------------
Track database schemas, column types, transformations, and ETL data lineage.

Features:
- Maintain table schemas and schema versioning
- Record transformations applied to data
- Track data lineage across ETL pipelines
- Log schema changes and provide audit trails
- Detect schema anomalies and differences
- Enterprise-ready with type hints and robust logging
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Logger setup
logger = logging.getLogger("metadata")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# =========================
# Metadata Manager Class
# =========================
class MetadataManager:
    def __init__(self):
        """
        Initialize the MetadataManager.
        Stores schemas, transformations, and lineage information.
        """
        self.schemas: Dict[str, Dict[str, Any]] = {}  # table_name -> schema info
        self.lineage: Dict[str, List[str]] = {}       # table_name -> list of source tables
        self.history: List[Dict[str, Any]] = []      # schema change history

    # -------------------------
    # Schema Management
    # -------------------------
    def register_schema(
        self,
        table_name: str,
        columns: Dict[str, str],
        version: Optional[int] = None
    ) -> None:
        """
        Register or update a table schema.

        Args:
            table_name (str): Name of the table.
            columns (Dict[str, str]): Column name -> data type mapping.
            version (int, optional): Schema version. Auto-incremented if not provided.

        Returns:
            None
        """
        current_version = version or self.schemas.get(table_name, {}).get("version", 0) + 1
        self.schemas[table_name] = {
            "columns": columns,
            "version": current_version,
            "updated_at": datetime.utcnow()
        }
        self.history.append({
            "table_name": table_name,
            "version": current_version,
            "columns": columns,
            "timestamp": datetime.utcnow()
        })
        logger.info(f"Registered schema for table '{table_name}', version {current_version}")

    def get_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest schema for a table.

        Args:
            table_name (str): Table name.

        Returns:
            Dict[str, Any]: Schema info or None if not found.
        """
        return self.schemas.get(table_name)

    def get_schema_history(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve full schema change history for a table.

        Args:
            table_name (str): Table name.

        Returns:
            List[Dict[str, Any]]: List of schema versions.
        """
        return [h for h in self.history if h["table_name"] == table_name]

    def diff_schemas(
        self, old: Dict[str, str], new: Dict[str, str]
    ) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        """
        Compare two schemas and return differences.

        Args:
            old (Dict[str, str]): Old schema column types.
            new (Dict[str, str]): New schema column types.

        Returns:
            Dict[str, Tuple[Optional[str], Optional[str]]]: 
                key -> (old_type, new_type)
        """
        diff = {}
        all_cols = set(old.keys()).union(new.keys())
        for col in all_cols:
            old_type = old.get(col)
            new_type = new.get(col)
            if old_type != new_type:
                diff[col] = (old_type, new_type)
        return diff

    # -------------------------
    # Lineage Tracking
    # -------------------------
    def record_lineage(self, table_name: str, source_tables: List[str]) -> None:
        """
        Record the data lineage of a table.

        Args:
            table_name (str): Target table name.
            source_tables (List[str]): Source tables used to generate this table.

        Returns:
            None
        """
        self.lineage[table_name] = source_tables
        logger.info(f"Recorded lineage for '{table_name}' from sources: {source_tables}")

    def get_lineage(self, table_name: str) -> List[str]:
        """
        Retrieve the lineage (source tables) for a table.

        Args:
            table_name (str): Table name.

        Returns:
            List[str]: List of source table names.
        """
        return self.lineage.get(table_name, [])

    # -------------------------
    # Anomaly Detection
    # -------------------------
    def detect_schema_anomalies(
        self, table_name: str, new_columns: Dict[str, str]
    ) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        """
        Detect anomalies between the existing schema and a new proposed schema.

        Args:
            table_name (str): Table name.
            new_columns (Dict[str, str]): Proposed column definitions.

        Returns:
            Dict[str, Tuple[Optional[str], Optional[str]]]: Differences detected.
        """
        current_schema = self.schemas.get(table_name, {}).get("columns", {})
        diff = self.diff_schemas(current_schema, new_columns)
        if diff:
            logger.warning(f"Schema anomalies detected for '{table_name}': {diff}")
        return diff
      
