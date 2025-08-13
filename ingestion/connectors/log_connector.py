# log_connector.py
"""
Log Connector for enterprise-grade ingestion pipelines.

Features:
- Parse system and application logs
- Support multiple formats (plain text, JSON, syslog)
- Extract structured data
- Logging and error handling
- Returns Pandas DataFrame
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import json
import logging
import os
import re

# Logging setup
logger = logging.getLogger("log_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class LogConnector:
    def __init__(self, log_format: str = "text", regex_pattern: Optional[str] = None):
        """
        Initialize Log Connector.

        Args:
            log_format (str): Format of the log file ('text', 'json', 'syslog')
            regex_pattern (str, optional): Regex pattern for structured text parsing (required for 'text' format)
        """
        self.log_format = log_format.lower()
        self.regex_pattern = regex_pattern
        if self.log_format == "text" and not regex_pattern:
            logger.warning("No regex pattern provided for text log parsing. Raw lines will be returned.")

    def read(self, file_path: str) -> pd.DataFrame:
        """
        Parse log file and return structured DataFrame.

        Args:
            file_path (str): Path to the log file

        Returns:
            pd.DataFrame: Structured log data

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If parsing fails
        """
        if not os.path.exists(file_path):
            logger.error(f"Log file not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            logger.info(f"Reading log file: {file_path}")
            data: List[Dict[str, Any]] = []

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if self.log_format == "json":
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON line: {line}")
                            continue
                    elif self.log_format == "syslog":
                        # Basic syslog parsing: "<timestamp> <host> <process>: <message>"
                        syslog_regex = r"^(?P<timestamp>\S+ \S+) (?P<host>\S+) (?P<process>\S+): (?P<message>.*)$"
                        match = re.match(syslog_regex, line)
                        if match:
                            entry = match.groupdict()
                        else:
                            logger.warning(f"Skipping invalid syslog line: {line}")
                            continue
                    else:  # plain text with optional regex
                        if self.regex_pattern:
                            match = re.match(self.regex_pattern, line)
                            if match:
                                entry = match.groupdict()
                            else:
                                continue
                        else:
                            entry = {"raw_line": line}

                    data.append(entry)

            if data:
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame()
                logger.warning("No log entries parsed from file")

            logger.info(f"Log file parsed successfully: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to parse log file '{file_path}': {e}")
            raise
          
