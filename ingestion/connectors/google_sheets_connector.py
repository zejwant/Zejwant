# google_sheets_connector.py
"""
Google Sheets Connector for enterprise-grade ingestion pipelines.

Features:
- Read Google Sheets via API
- Handle multiple sheets and ranges
- Authenticate via OAuth2
- Logging and error handling
- Returns Pandas DataFrame
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import logging
import os

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Logging setup
logger = logging.getLogger("google_sheets_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class GoogleSheetsConnector:
    def __init__(
        self,
        credentials_path: str,
        scopes: Optional[List[str]] = None
    ):
        """
        Initialize Google Sheets Connector.

        Args:
            credentials_path (str): Path to service account JSON credentials
            scopes (List[str], optional): OAuth2 scopes. Default is read-only.
        """
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found: {credentials_path}")

        self.scopes = scopes or ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        self.credentials_path = credentials_path
        self.service = self._authenticate()

    def _authenticate(self):
        """
        Authenticate using service account credentials.

        Returns:
            googleapiclient.discovery.Resource: Google Sheets API service
        """
        try:
            creds = Credentials.from_service_account_file(
                self.credentials_path,
                scopes=self.scopes
            )
            service = build('sheets', 'v4', credentials=creds)
            logger.info("Google Sheets API authentication successful")
            return service
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Sheets API: {e}")
            raise

    def read(
        self,
        spreadsheet_id: str,
        ranges: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Read data from Google Sheets and return as Pandas DataFrame.

        Args:
            spreadsheet_id (str): The Google Sheets spreadsheet ID
            ranges (List[str], optional): List of A1 notation ranges or sheet names

        Returns:
            pd.DataFrame: Combined DataFrame from all requested ranges

        Raises:
            HttpError: If the API request fails
        """
        try:
            sheet = self.service.spreadsheets()
            result = sheet.values().batchGet(spreadsheetId=spreadsheet_id, ranges=ranges).execute()
            data_frames: List[pd.DataFrame] = []

            for value_range in result.get("valueRanges", []):
                values = value_range.get("values", [])
                if not values:
                    continue
                df = pd.DataFrame(values[1:], columns=values[0])
                data_frames.append(df)

            if data_frames:
                combined_df = pd.concat(data_frames, ignore_index=True)
            else:
                combined_df = pd.DataFrame()
                logger.warning("No data found in the specified ranges")

            logger.info(f"Google Sheets data loaded successfully: {len(combined_df)} rows")
            return combined_df

        except HttpError as e:
            logger.error(f"Google Sheets API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read Google Sheets: {e}")
            raise
          
