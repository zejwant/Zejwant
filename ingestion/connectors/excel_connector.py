# excel_connector.py
"""
Excel Connector for enterprise-grade ingestion pipelines.

Features:
- Read .xls and .xlsx files
- Support multiple sheets and cell ranges
- Efficient handling of large files
- Schema validation
- Logging and error handling
- Returns Pandas DataFrame
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import logging
import os

# Logging setup
logger = logging.getLogger("excel_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class ExcelConnector:
    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        clean_options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Excel Connector.

        Args:
            schema (Dict[str, Any], optional): Expected column types for validation
            clean_options (Dict[str, Any], optional): Data cleaning options
                e.g., {"drop_duplicates": True, "fillna": {"column": value}}
        """
        self.schema = schema
        self.clean_options = clean_options or {}

    def read(
        self,
        file_path: str,
        sheet_name: Union[str, int, List[Union[str, int]]] = 0,
        usecols: Optional[Union[str, List[str]]] = None,
        engine: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read Excel file into a Pandas DataFrame.

        Args:
            file_path (str): Path to the Excel file
            sheet_name (str | int | List): Sheet(s) to read
            usecols (str | List[str], optional): Columns or range to read
            engine (str, optional): Excel engine (openpyxl, xlrd)

        Returns:
            pd.DataFrame: Loaded and validated data

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If schema validation fails
        """
        if not os.path.exists(file_path):
            logger.error(f"Excel file not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            logger.info(f"Reading Excel file: {file_path}, sheet: {sheet_name}")
            df_dict = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                usecols=usecols,
                engine=engine,
                dtype=str  # read all as string initially for schema validation
            )

            # If multiple sheets, concatenate into single DataFrame
            if isinstance(df_dict, dict):
                df_list = []
                for sheet, sheet_df in df_dict.items():
                    sheet_df["__sheet_name"] = sheet
                    df_list.append(sheet_df)
                df = pd.concat(df_list, ignore_index=True)
            else:
                df = df_dict

            df = self._validate_schema(df)
            df = self._clean_data(df)

            logger.info(f"Excel file loaded successfully: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to read Excel file '{file_path}': {e}")
            raise

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

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data cleaning options.

        Args:
            df (pd.DataFrame): DataFrame to clean

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if self.clean_options.get("drop_duplicates", True):
            df = df.drop_duplicates()

        fillna_dict = self.clean_options.get("fillna", {})
        if fillna_dict:
            df = df.fillna(fillna_dict)

        # Additional cleaning options can be added here
        return df
      
