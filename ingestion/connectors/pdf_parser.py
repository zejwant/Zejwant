# pdf_connector.py
"""
PDF Connector for enterprise-grade ingestion pipelines.

Features:
- Extract tables and text from PDFs
- Handle multi-page PDFs
- Optional OCR for scanned PDFs
- Returns structured data as Pandas DataFrame
- Logging and error handling
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import logging
import os

# PDF processing
import pdfplumber
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Logging setup
logger = logging.getLogger("pdf_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class PDFConnector:
    def __init__(self, use_ocr: bool = False):
        """
        Initialize PDF Connector.

        Args:
            use_ocr (bool): Enable OCR for scanned PDFs (requires pytesseract and PIL)
        """
        if use_ocr and not OCR_AVAILABLE:
            logger.warning("OCR dependencies not installed, falling back to text extraction only")
            use_ocr = False
        self.use_ocr = use_ocr

    def read(self, file_path: str) -> pd.DataFrame:
        """
        Extract text and tables from PDF file.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            pd.DataFrame: Extracted structured data

        Raises:
            FileNotFoundError: If PDF file does not exist
            ValueError: If PDF cannot be read
        """
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        all_data: List[Dict[str, Any]] = []

        try:
            logger.info(f"Reading PDF file: {file_path}")
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:  # skip empty tables
                            df_table = pd.DataFrame(table[1:], columns=table[0])
                            df_table["__page_number"] = page_num
                            all_data.append(df_table)

                    # Extract text
                    text = page.extract_text()
                    if text and not self.use_ocr:
                        all_data.append(pd.DataFrame([{"__page_number": page_num, "text": text}]))

                    # OCR fallback
                    elif self.use_ocr:
                        img = page.to_image(resolution=300).original
                        ocr_text = pytesseract.image_to_string(Image.fromarray(img))
                        all_data.append(pd.DataFrame([{"__page_number": page_num, "text": ocr_text}]))

            if all_data:
                df = pd.concat(all_data, ignore_index=True)
            else:
                df = pd.DataFrame()
                logger.warning("No tables or text extracted from PDF")

            logger.info(f"PDF processed successfully: {len(df)} rows extracted")
            return df

        except Exception as e:
            logger.error(f"Failed to read PDF '{file_path}': {e}")
            raise
                                                
