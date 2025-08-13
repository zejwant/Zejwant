# api_connector.py
"""
Enterprise-grade API connector module supporting REST and GraphQL.

Features:
- Async requests with retry & exponential backoff
- Configurable headers, auth, OAuth2 support, and pagination
- JSON/XML response parsing with flattening
- Structured logging (JSON) & metrics
- Return data as Pandas DataFrame or JSON
"""

from typing import Any, Dict, List, Optional, Union
import asyncio
import aiohttp
import xmltodict
import pandas as pd
import logging
import json
import time

# Structured JSON logger
logger = logging.getLogger("api_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class APIConnector:
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Union[aiohttp.BasicAuth, Dict[str, str]]] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        timeout: int = 30,
        oauth_refresh_func: Optional[callable] = None
    ):
        """
        Enterprise API connector initialization.

        Args:
            base_url (str): Base API URL
            headers (dict, optional): HTTP headers
            auth (aiohttp.BasicAuth or dict, optional): Auth credentials
            max_retries (int): Retry attempts
            backoff_factor (float): Exponential backoff factor
            timeout (int): Request timeout in seconds
            oauth_refresh_func (callable, optional): Function to refresh OAuth2 token
        """
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.auth = auth
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.oauth_refresh_func = oauth_refresh_func

    async def fetch(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        return_type: str = "json",
        pagination: Optional[Dict[str, Any]] = None
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Fetch data from REST or GraphQL endpoint with retries, pagination, and parsing.

        Args:
            endpoint (str): API endpoint path
            method (str): HTTP method ("GET" or "POST")
            params (dict, optional): Query parameters
            payload (dict, optional): POST body (for GraphQL or REST POST)
            return_type (str): 'json' or 'dataframe'
            pagination (dict, optional): Pagination configuration
                Example: {'page_key': 'cursor', 'next_param': 'endCursor', 'limit': 100}

        Returns:
            Union[pd.DataFrame, List[Dict[str, Any]]]: API response
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        retries = 0
        all_results: List[Dict[str, Any]] = []

        while retries <= self.max_retries:
            try:
                # Refresh OAuth token if applicable
                if self.oauth_refresh_func:
                    token = self.oauth_refresh_func()
                    self.headers['Authorization'] = f"Bearer {token}"

                async with aiohttp.ClientSession(headers=self.headers, auth=self.auth) as session:
                    if method.upper() == "GET":
                        async with session.get(url, params=params, timeout=self.timeout) as resp:
                            data = await self._parse_response(resp)
                    elif method.upper() == "POST":
                        async with session.post(url, json=payload, params=params, timeout=self.timeout) as resp:
                            data = await self._parse_response(resp)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

                    # Flatten list/dict
                    if isinstance(data, dict) and 'data' in data:
                        data = self._flatten_data(data['data'])

                    all_results.extend(data if isinstance(data, list) else [data])

                    # Handle cursor-based pagination
                    if pagination and 'next_param' in pagination:
                        last_item = all_results[-1] if all_results else {}
                        next_val = last_item.get(pagination['next_param'])
                        if next_val:
                            params = params or {}
                            params[pagination['page_key']] = next_val
                            retries = 0  # reset retries for next page
                            continue

                    break  # exit loop if successful and no pagination

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait = self.backoff_factor * (2 ** retries)
                logger.warning(json.dumps({
                    "message": f"API request failed: {e}, retrying in {wait}s",
                    "endpoint": endpoint
                }))
                await asyncio.sleep(wait)
                retries += 1
                if retries > self.max_retries:
                    logger.error(json.dumps({
                        "message": "Max retries reached",
                        "endpoint": endpoint
                    }))
                    raise e

        # Convert to DataFrame if requested
        if return_type.lower() == "dataframe":
            return pd.DataFrame(all_results)
        return all_results

    async def _parse_response(self, resp: aiohttp.ClientResponse) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Parse JSON or XML response with flattening support."""
        content_type = resp.headers.get('Content-Type', '')
        text = await resp.text()
        if 'application/json' in content_type:
            data = await resp.json()
        elif 'application/xml' in content_type or 'text/xml' in content_type:
            data = xmltodict.parse(text)
        else:
            logger.warning(json.dumps({
                "message": f"Unknown content type {content_type}, returning raw text"
            }))
            data = text
        return data

    def _flatten_data(self, data: Any) -> List[Dict[str, Any]]:
        """
        Flatten nested dict/list structures into a list of dictionaries.
        """
        if isinstance(data, list):
            return [self._flatten_item(item) for item in data]
        elif isinstance(data, dict):
            return [self._flatten_item(data)]
        return [{"value": data}]

    def _flatten_item(self, item: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Recursively flatten a nested dictionary.
        """
        flat_dict = {}
        for k, v in item.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flat_dict.update(self._flatten_item(v, new_key, sep=sep))
            elif isinstance(v, list):
                # Convert lists to string or JSON
                flat_dict[new_key] = json.dumps(v)
            else:
                flat_dict[new_key] = v
        return flat_dict
      
