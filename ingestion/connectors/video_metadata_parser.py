# video_metadata_parser.py
"""
Video Metadata Parser for enterprise-grade ingestion pipelines.

Features:
- Extract metadata (duration, codec, resolution, frame rate) from video files
- Support multiple formats (MP4, MOV, AVI)
- Logging and error handling
- Returns structured Pandas DataFrame
"""

from typing import Any, Dict, List
import pandas as pd
import logging
import os
import ffmpeg

# Logging setup
logger = logging.getLogger("video_metadata_parser")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class VideoMetadataParser:
    SUPPORTED_FORMATS = ("mp4", "mov", "avi")

    def __init__(self):
        """
        Initialize Video Metadata Parser.
        """
        pass

    def parse_videos(self, paths: List[str]) -> pd.DataFrame:
        """
        Parse a list of video files and extract metadata.

        Args:
            paths (List[str]): List of video file paths

        Returns:
            pd.DataFrame: DataFrame containing metadata for each video
        """
        results: List[Dict[str, Any]] = []

        for path in paths:
            if not os.path.exists(path):
                logger.warning(f"Video file not found: {path}")
                continue

            ext = os.path.splitext(path)[1][1:].lower()
            if ext not in self.SUPPORTED_FORMATS:
                logger.warning(f"Unsupported video format {ext}: {path}")
                continue

            try:
                logger.info(f"Processing video: {path}")
                probe = ffmpeg.probe(path)
                video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
                if not video_streams:
                    logger.warning(f"No video stream found in {path}")
                    continue
                v_stream = video_streams[0]

                metadata = {
                    "file_path": path,
                    "format": ext,
                    "duration_sec": float(probe["format"]["duration"]),
                    "codec": v_stream.get("codec_name"),
                    "width": int(v_stream.get("width", 0)),
                    "height": int(v_stream.get("height", 0)),
                    "frame_rate": eval(v_stream.get("r_frame_rate", "0")),
                }

                results.append(metadata)

            except Exception as e:
                logger.error(f"Failed to parse video '{path}': {e}")

        if results:
            df = pd.DataFrame(results)
            logger.info(f"Video metadata parsing completed: {len(df)} videos processed")
        else:
            df = pd.DataFrame()
            logger.warning("No video metadata was successfully extracted")

        return df
      
