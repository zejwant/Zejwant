# image_parser.py
"""
Image Parser for enterprise-grade ingestion pipelines.

Features:
- Extract metadata (EXIF) and text (OCR) from images
- Support multiple formats (JPEG, PNG, TIFF)
- Logging and error handling
- Returns structured Pandas DataFrame
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import logging
import os
from PIL import Image, ExifTags
import pytesseract

# Logging setup
logger = logging.getLogger("image_parser")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class ImageParser:
    SUPPORTED_FORMATS = ("JPEG", "PNG", "TIFF")

    def __init__(self, ocr_lang: str = "eng"):
        """
        Initialize Image Parser.

        Args:
            ocr_lang (str): Language for OCR (default 'eng')
        """
        self.ocr_lang = ocr_lang

    def parse_images(self, paths: List[str]) -> pd.DataFrame:
        """
        Parse a list of images and extract EXIF metadata and OCR text.

        Args:
            paths (List[str]): List of image file paths

        Returns:
            pd.DataFrame: DataFrame containing metadata and extracted text
        """
        results: List[Dict[str, Any]] = []

        for path in paths:
            if not os.path.exists(path):
                logger.warning(f"Image file not found: {path}")
                continue

            try:
                logger.info(f"Processing image: {path}")
                with Image.open(path) as img:
                    if img.format not in self.SUPPORTED_FORMATS:
                        logger.warning(f"Unsupported image format {img.format}: {path}")
                        continue

                    # Extract EXIF metadata
                    exif_data = {}
                    if hasattr(img, "_getexif") and img._getexif():
                        for tag_id, value in img._getexif().items():
                            tag = ExifTags.TAGS.get(tag_id, tag_id)
                            exif_data[tag] = value

                    # OCR text extraction
                    text = pytesseract.image_to_string(img, lang=self.ocr_lang)

                    results.append({
                        "file_path": path,
                        "format": img.format,
                        "size": img.size,
                        "mode": img.mode,
                        "exif": exif_data,
                        "ocr_text": text.strip()
                    })

            except Exception as e:
                logger.error(f"Failed to parse image '{path}': {e}")

        if results:
            df = pd.DataFrame(results)
            logger.info(f"Image parsing completed: {len(df)} images processed")
        else:
            df = pd.DataFrame()
            logger.warning("No images were successfully parsed")

        return df
      
