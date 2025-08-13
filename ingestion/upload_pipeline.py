# upload_pipeline.py
"""
Enterprise-grade bulk file ingestion pipeline.

Supports CSV, Excel, JSON, Parquet, XML, and PDFs.

Features:
- Async concurrent file processing
- Schema validation and data quality checks
- Logging and error handling
- Intermediate staging for processed files
- Returns Pandas DataFrame for downstream processing
"""

from typing import Any, Callable, Dict, List, Optional, Union
import asyncio
import pandas as pd
import json
import xmltodict
import logging
import os
import aiofiles
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Logging setup
logger = logging.getLogger("upload_pipeline")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class UploadPipeline:
    def __init__(
        self,
        staging_dir: Union[str, Path],
        max_workers: int = 5,
        validate_func: Optional[Callable[[pd.DataFrame], bool]] = None
    ):
        """
        Initialize upload pipeline.

        Args:
            staging_dir (str | Path): Directory to store intermediate staging files
            max_workers (int): Maximum concurrent file processing workers
            validate_func (Callable, optional): Custom validation function for DataFrame
        """
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.validate_func = validate_func

    async def process_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, pd.DataFrame]:
        """
        Process multiple files concurrently.

        Args:
            file_paths (List[str | Path]): List of file paths to ingest

        Returns:
            Dict[str, pd.DataFrame]: Mapping of file path -> DataFrame
        """
        results: Dict[str, pd.DataFrame] = {}
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, self._process_file_sync, file_path)
                for file_path in file_paths
            ]
            for file_path, future in zip(file_paths, asyncio.as_completed(tasks)):
                try:
                    df = await future
                    results[str(file_path)] = df
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        return results

    def _process_file_sync(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Synchronous file processing (to be called from executor).

        Args:
            file_path (str | Path): File path

        Returns:
            pd.DataFrame: Processed data
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        df: pd.DataFrame

        try:
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(file_path)
            elif ext == ".json":
                df = pd.read_json(file_path)
            elif ext == ".parquet":
                df = pd.read_parquet(file_path)
            elif ext == ".xml":
                with open(file_path, "r", encoding="utf-8") as f:
                    xml_data = xmltodict.parse(f.read())
                    # Flatten nested XML to list of dicts if possible
                    df = pd.json_normalize(xml_data)
            elif ext == ".pdf":
                # Minimal PDF parsing example
                import fitz  # PyMuPDF
                pdf = fitz.open(file_path)
                data = []
                for page in pdf:
                    data.append({"page_text": page.get_text()})
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise

        # Schema / data quality validation
        if self.validate_func:
            if not self.validate_func(df):
                raise ValueError(f"Validation failed for {file_path}")

        # Save intermediate staging file
        staging_file = self.staging_dir / file_path.name
        try:
            df.to_parquet(staging_file.with_suffix(".parquet"), index=False)
        except Exception as e:
            logger.warning(f"Failed to write staging file for {file_path}: {e}")

        logger.info(f"Successfully processed {file_path}, {len(df)} records")
        return df

    async def process_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Process a single file asynchronously.

        Args:
            file_path (str | Path): File path

        Returns:
            pd.DataFrame: Processed DataFrame
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._process_file_sync, file_path)
      
