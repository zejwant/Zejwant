"""
cleaning/deduplication.py

Enterprise-grade deduplication utilities for multiple formats and sources.

Features:
- Deduplication based on single/multiple columns
- Fuzzy matching for strings (Levenshtein, cosine similarity)
- Time-based deduplication for event logs
- Multi-format and multi-source deduplication
- Detailed deduplication summary
- Logging and error handling
"""

import logging
from typing import Any, List, Optional, Dict
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

# Logger setup
logger = logging.getLogger("deduplication")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Deduplicator:
    """Enterprise-grade deduplication strategies."""

    @staticmethod
    def drop_exact_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """Remove exact duplicates across entire df or subset of columns."""
        original_len = len(df)
        df_cleaned = df.drop_duplicates(subset=subset)
        removed = original_len - len(df_cleaned)
        logger.info(f"Dropped {removed} exact duplicates on columns: {subset}")
        return {"data": df_cleaned, "removed": removed}

    @staticmethod
    def keep_first_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """Keep first occurrence of duplicates and drop others."""
        original_len = len(df)
        df_cleaned = df.drop_duplicates(subset=subset, keep='first')
        removed = original_len - len(df_cleaned)
        logger.info(f"Kept first and removed {removed} duplicates on columns: {subset}")
        return {"data": df_cleaned, "removed": removed}

    @staticmethod
    def keep_last_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """Keep last occurrence of duplicates and drop others."""
        original_len = len(df)
        df_cleaned = df.drop_duplicates(subset=subset, keep='last')
        removed = original_len - len(df_cleaned)
        logger.info(f"Kept last and removed {removed} duplicates on columns: {subset}")
        return {"data": df_cleaned, "removed": removed}

    @staticmethod
    def fuzzy_deduplicate_levenshtein(df: pd.DataFrame, column: str, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Deduplicate using fuzzy matching (Levenshtein similarity) on a string column.
        threshold: 0-1 similarity score to consider as duplicate.
        """
        def is_similar(a, b):
            return SequenceMatcher(None, a, b).ratio() >= threshold

        to_drop = set()
        vals = df[column].astype(str).tolist()
        for i in range(len(vals)):
            if i in to_drop:
                continue
            for j in range(i + 1, len(vals)):
                if j in to_drop:
                    continue
                if is_similar(vals[i], vals[j]):
                    to_drop.add(j)
        df_cleaned = df.drop(df.index[list(to_drop)])
        logger.info(f"Fuzzy Levenshtein deduplication removed {len(to_drop)} rows from column: {column}")
        return {"data": df_cleaned, "removed": len(to_drop)}

    @staticmethod
    def fuzzy_deduplicate_cosine(df: pd.DataFrame, column: str, threshold: float = 0.85) -> Dict[str, Any]:
        """
        Deduplicate using cosine similarity on TF-IDF vectors of a string column.
        """
        vals = df[column].astype(str).tolist()
        vectorizer = TfidfVectorizer().fit_transform(vals)
        similarity_matrix = cosine_similarity(vectorizer)
        to_drop = set()
        for i in range(len(vals)):
            if i in to_drop:
                continue
            for j in range(i + 1, len(vals)):
                if j in to_drop:
                    continue
                if similarity_matrix[i, j] >= threshold:
                    to_drop.add(j)
        df_cleaned = df.drop(df.index[list(to_drop)])
        logger.info(f"Fuzzy cosine deduplication removed {len(to_drop)} rows from column: {column}")
        return {"data": df_cleaned, "removed": len(to_drop)}

    @staticmethod
    def deduplicate_time_window(df: pd.DataFrame, timestamp_col: str, window_seconds: int = 60,
                                subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Deduplicate events within a time window.
        """
        if timestamp_col not in df.columns:
            logger.error(f"Timestamp column {timestamp_col} not found")
            return {"data": df, "removed": 0}

        df_sorted = df.sort_values(timestamp_col)
        to_drop = set()
        last_seen = {}

        for idx, row in df_sorted.iterrows():
            key = tuple(row[col] for col in subset) if subset else None
            ts = pd.to_datetime(row[timestamp_col])
            if key in last_seen:
                last_ts = last_seen[key]
                if (ts - last_ts).total_seconds() <= window_seconds:
                    to_drop.add(idx)
                else:
                    last_seen[key] = ts
            else:
                last_seen[key] = ts

        df_cleaned = df.drop(df.index[list(to_drop)])
        logger.info(f"Time-window deduplication removed {len(to_drop)} rows using window {window_seconds}s")
        return {"data": df_cleaned, "removed": len(to_drop)}

    @staticmethod
    def deduplicate_numeric_tolerance(df: pd.DataFrame, column: str, tolerance: float = 0.001) -> Dict[str, Any]:
        """
        Deduplicate numeric column within a tolerance range.
        """
        df_sorted = df.sort_values(column)
        to_drop = set()
        prev_val = None
        prev_idx = None

        for idx, val in zip(df_sorted.index, df_sorted[column]):
            if prev_val is not None and abs(val - prev_val) <= tolerance:
                to_drop.add(idx)
            else:
                prev_val = val
                prev_idx = idx

        df_cleaned = df.drop(df.index[list(to_drop)])
        logger.info(f"Numeric tolerance deduplication removed {len(to_drop)} rows from column: {column}")
        return {"data": df_cleaned, "removed": len(to_drop)}

    @staticmethod
    def deduplicate_multi_column(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """
        Deduplicate based on combination of multiple columns.
        """
        return Deduplicator.drop_exact_duplicates(df, subset=columns)

    @staticmethod
    def deduplicate_ignore_case(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Deduplicate ignoring string case.
        """
        df[column] = df[column].astype(str).str.lower()
        return Deduplicator.drop_exact_duplicates(df, subset=[column])

    @staticmethod
    def deduplicate_trimmed(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Deduplicate after trimming whitespaces.
        """
        df[column] = df[column].astype(str).str.strip()
        return Deduplicator.drop_exact_duplicates(df, subset=[column])

    @staticmethod
    def deduplicate_combined_string(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """
        Deduplicate based on concatenated string of multiple columns.
        """
        combined = df[columns].astype(str).agg('_'.join, axis=1)
        df_temp = df.copy()
        df_temp['combined_key'] = combined
        result = Deduplicator.drop_exact_duplicates(df_temp, subset=['combined_key'])
        result['data'] = result['data'].drop(columns=['combined_key'])
        return result

    @staticmethod
    def deduplicate_by_hash(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """
        Deduplicate using hash of selected columns.
        """
        df_temp = df.copy()
        df_temp['_hash'] = df_temp[columns].astype(str).agg(lambda x: hash('_'.join(x)), axis=1)
        result = Deduplicator.drop_exact_duplicates(df_temp, subset=['_hash'])
        result['data'] = result['data'].drop(columns=['_hash'])
        return result

    @staticmethod
    def deduplicate_by_rounding(df: pd.DataFrame, column: str, decimals: int = 2) -> Dict[str, Any]:
        """
        Deduplicate numeric column after rounding.
        """
        df[column] = df[column].round(decimals)
        return Deduplicator.drop_exact_duplicates(df, subset=[column])

    @staticmethod
    def deduplicate_keep_max(df: pd.DataFrame, value_column: str, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Keep row with maximum value in a subset group.
        """
        df_sorted = df.sort_values(value_column, ascending=False)
        return Deduplicator.keep_first_duplicates(df_sorted, subset=subset)

    @staticmethod
    def deduplicate_keep_min(df: pd.DataFrame, value_column: str, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Keep row with minimum value in a subset group.
        """
        df_sorted = df.sort_values(value_column, ascending=True)
        return Deduplicator.keep_first_duplicates(df_sorted, subset=subset)

    @staticmethod
    def deduplicate_across_sources(dfs: List[pd.DataFrame], subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Deduplicate across multiple dataframes (multi-source).
        """
        combined_df = pd.concat(dfs, ignore_index=True)
        result = Deduplicator.drop_exact_duplicates(combined_df, subset=subset)
        logger.info(f"Deduplicated across {len(dfs)} sources, removed {result['removed']} duplicates")
        return result
      
