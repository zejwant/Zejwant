# audio_metadata_parser.py
"""
Audio Metadata Parser for enterprise-grade ingestion pipelines.

Features:
- Extract metadata (duration, bitrate, sample rate, channels) from audio files
- Support multiple formats (MP3, WAV, FLAC)
- Logging and error handling
- Returns structured Pandas DataFrame
"""

from typing import Any, Dict, List
import pandas as pd
import logging
import os
from mutagen import File as MutagenFile

# Logging setup
logger = logging.getLogger("audio_metadata_parser")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class AudioMetadataParser:
    SUPPORTED_FORMATS = ("mp3", "wav", "flac")

    def __init__(self):
        """
        Initialize Audio Metadata Parser.
        """
        pass

    def parse_audios(self, paths: List[str]) -> pd.DataFrame:
        """
        Parse a list of audio files and extract metadata.

        Args:
            paths (List[str]): List of audio file paths

        Returns:
            pd.DataFrame: DataFrame containing metadata for each audio file
        """
        results: List[Dict[str, Any]] = []

        for path in paths:
            if not os.path.exists(path):
                logger.warning(f"Audio file not found: {path}")
                continue

            ext = os.path.splitext(path)[1][1:].lower()
            if ext not in self.SUPPORTED_FORMATS:
                logger.warning(f"Unsupported audio format {ext}: {path}")
                continue

            try:
                logger.info(f"Processing audio: {path}")
                audio = MutagenFile(path)
                if audio is None:
                    logger.warning(f"Cannot read audio metadata: {path}")
                    continue

                metadata = {
                    "file_path": path,
                    "format": ext,
                    "duration_sec": getattr(audio.info, "length", None),
                    "bitrate": getattr(audio.info, "bitrate", None),
                    "sample_rate": getattr(audio.info, "sample_rate", None),
                    "channels": getattr(audio.info, "channels", None),
                }

                results.append(metadata)

            except Exception as e:
                logger.error(f"Failed to parse audio '{path}': {e}")

        if results:
            df = pd.DataFrame(results)
            logger.info(f"Audio metadata parsing completed: {len(df)} audios processed")
        else:
            df = pd.DataFrame()
            logger.warning("No audio metadata was successfully extracted")

        return df
      
