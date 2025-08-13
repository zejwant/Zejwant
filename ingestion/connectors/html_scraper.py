# html_scraper.py
"""
HTML Scraper Connector for enterprise-grade ingestion pipelines.

Features:
- Scrape HTML tables and elements
- Support multiple CSS and XPath selectors
- Logging and error handling
- Returns Pandas DataFrame
"""

from typing import Any, List, Optional, Dict
import pandas as pd
import logging
import requests
from lxml import html

# Logging setup
logger = logging.getLogger("html_scraper")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class HTMLScraper:
    def __init__(self):
        """
        Initialize HTML Scraper.
        """
        pass

    def scrape(
        self,
        url: str,
        selectors: Optional[List[Dict[str, str]]] = None,
        timeout: int = 10
    ) -> pd.DataFrame:
        """
        Scrape HTML from a URL and extract elements or tables.

        Args:
            url (str): URL of the HTML page
            selectors (List[Dict[str,str]], optional): List of selectors. Each dict can have:
                - 'type': 'css' or 'xpath'
                - 'query': the CSS or XPath query
            timeout (int): Request timeout in seconds

        Returns:
            pd.DataFrame: Extracted data as a DataFrame
        """
        try:
            logger.info(f"Fetching HTML content from: {url}")
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            tree = html.fromstring(resp.content)

            all_data = []

            if selectors:
                for sel in selectors:
                    sel_type = sel.get("type", "css").lower()
                    query = sel.get("query")
                    if not query:
                        continue

                    if sel_type == "css":
                        elements = tree.cssselect(query)
                        for el in elements:
                            all_data.append({"selector": query, "text": el.text_content().strip()})
                    elif sel_type == "xpath":
                        elements = tree.xpath(query)
                        for el in elements:
                            # If element is an lxml element, extract text
                            text = el.text_content().strip() if hasattr(el, "text_content") else str(el).strip()
                            all_data.append({"selector": query, "text": text})
                    else:
                        logger.warning(f"Unknown selector type: {sel_type}")

            # Extract HTML tables as well
            tables = pd.read_html(resp.text)
            for idx, table in enumerate(tables):
                table["__table_index"] = idx
                all_data.append(table)

            # Combine results
            df_list = []
            for item in all_data:
                if isinstance(item, pd.DataFrame):
                    df_list.append(item)
                elif isinstance(item, dict):
                    df_list.append(pd.DataFrame([item]))

            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)
            else:
                combined_df = pd.DataFrame()
                logger.warning("No data extracted from HTML")

            logger.info(f"HTML scraping completed: {len(combined_df)} rows extracted")
            return combined_df

        except requests.RequestException as e:
            logger.error(f"Failed to fetch HTML from {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to scrape HTML from {url}: {e}")
            raise
          
