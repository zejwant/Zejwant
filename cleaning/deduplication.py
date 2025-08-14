"""
cleaning/deduplication_v1_1_full.py

Enterprise-grade deduplication utilities v1.1 Full

Features:
- Exact, fuzzy (Levenshtein & cosine), time-window, numeric, multi-column dedup
- Combined string, ignore-case, trimmed, hash-based, rounding-based
- Keep max/min per group
- Multi-source deduplication
- Optional in-place operations
- Strict mode via raise_on_error
- Runtime metrics logging
- Optimized for large datasets
"""

import logging
import time
from typing import Any, List, Optional, Dict, Union
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("deduplication_v1_1_full")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Deduplicator:
    """Enterprise-grade deduplication strategies v1.1 Full"""

    @staticmethod
    def drop_exact_duplicates(df: pd.DataFrame,
                              subset: Optional[List[str]] = None,
                              inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        start = time.time()
        df_copy = df if inplace else df.copy()
        original_len = len(df_copy)
        df_copy = df_copy.drop_duplicates(subset=subset)
        removed = original_len - len(df_copy)
        logger.info(f"Dropped {removed} exact duplicates on columns: {subset} in {time.time()-start:.2f}s")
        return {"data": df_copy, "removed": removed}

    @staticmethod
    def keep_first_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None,
                              inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        start = time.time()
        df_copy = df if inplace else df.copy()
        original_len = len(df_copy)
        df_copy = df_copy.drop_duplicates(subset=subset, keep='first')
        removed = original_len - len(df_copy)
        logger.info(f"Kept first duplicates, removed {removed} rows on columns: {subset} in {time.time()-start:.2f}s")
        return {"data": df_copy, "removed": removed}

    @staticmethod
    def keep_last_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None,
                             inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        start = time.time()
        df_copy = df if inplace else df.copy()
        original_len = len(df_copy)
        df_copy = df_copy.drop_duplicates(subset=subset, keep='last')
        removed = original_len - len(df_copy)
        logger.info(f"Kept last duplicates, removed {removed} rows on columns: {subset} in {time.time()-start:.2f}s")
        return {"data": df_copy, "removed": removed}

    @staticmethod
    def fuzzy_deduplicate_levenshtein(df: pd.DataFrame, column: str, threshold: float = 0.8,
                                      max_rows: int = 10000, inplace: bool = False,
                                      raise_on_error: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        start = time.time()
        df_copy = df if inplace else df.copy()
        try:
            if len(df_copy) > max_rows:
                logger.warning(f"Levenshtein deduplication may be slow for {len(df_copy)} rows")
            vals = df_copy[column].astype(str).tolist()
            to_drop = set()
            for i in range(len(vals)):
                if i in to_drop: continue
                for j in range(i+1, len(vals)):
                    if j in to_drop: continue
                    if SequenceMatcher(None, vals[i], vals[j]).ratio() >= threshold:
                        to_drop.add(j)
            df_copy = df_copy.drop(df_copy.index[list(to_drop)])
            logger.info(f"Fuzzy Levenshtein removed {len(to_drop)} rows from {column} in {time.time()-start:.2f}s")
            return {"data": df_copy, "removed": len(to_drop)}
        except Exception as e:
            if raise_on_error: raise e
            logger.error(f"Fuzzy Levenshtein failed: {e}")
            return {"data": df_copy, "removed": 0}

    @staticmethod
    def fuzzy_deduplicate_cosine(df: pd.DataFrame, column: str, threshold: float = 0.85,
                                 max_rows: int = 50000, inplace: bool = False,
                                 raise_on_error: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        start = time.time()
        df_copy = df if inplace else df.copy()
        try:
            if len(df_copy) > max_rows:
                logger.warning(f"Cosine dedup may be slow for {len(df_copy)} rows")
            vals = df_copy[column].astype(str).tolist()
            vectorizer = TfidfVectorizer().fit_transform(vals)
            similarity_matrix = cosine_similarity(vectorizer)
            to_drop = set()
            for i in range(len(vals)):
                if i in to_drop: continue
                for j in range(i+1, len(vals)):
                    if j in to_drop: continue
                    if similarity_matrix[i,j] >= threshold:
                        to_drop.add(j)
            df_copy = df_copy.drop(df_copy.index[list(to_drop)])
            logger.info(f"Fuzzy cosine removed {len(to_drop)} rows from {column} in {time.time()-start:.2f}s")
            return {"data": df_copy, "removed": len(to_drop)}
        except Exception as e:
            if raise_on_error: raise e
            logger.error(f"Fuzzy cosine failed: {e}")
            return {"data": df_copy, "removed": 0}

    @staticmethod
    def deduplicate_time_window(df: pd.DataFrame, timestamp_col: str, window_seconds: int = 60,
                                subset: Optional[List[str]] = None, inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        start = time.time()
        df_copy = df if inplace else df.copy()
        if timestamp_col not in df_copy.columns:
            logger.error(f"Timestamp column {timestamp_col} not found")
            return {"data": df_copy, "removed": 0}

        df_sorted = df_copy.sort_values(timestamp_col)
        to_drop = set()
        last_seen = {}
        for idx, row in df_sorted.iterrows():
            key = tuple(row[col] for col in subset) if subset else None
            ts = pd.to_datetime(row[timestamp_col])
            if key in last_seen and (ts - last_seen[key]).total_seconds() <= window_seconds:
                to_drop.add(idx)
            else:
                last_seen[key] = ts
        df_copy = df_copy.drop(df_copy.index[list(to_drop)])
        logger.info(f"Time-window removed {len(to_drop)} rows in {time.time()-start:.2f}s")
        return {"data": df_copy, "removed": len(to_drop)}

    @staticmethod
    def deduplicate_numeric_tolerance(df: pd.DataFrame, column: str, tolerance: float = 0.001,
                                      inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        start = time.time()
        df_copy = df if inplace else df.copy()
        df_sorted = df_copy.sort_values(column)
        to_drop = set()
        prev_val = None
        for idx, val in zip(df_sorted.index, df_sorted[column]):
            if prev_val is not None and abs(val - prev_val) <= tolerance:
                to_drop.add(idx)
            else:
                prev_val = val
        df_copy = df_copy.drop(df_copy.index[list(to_drop)])
        logger.info(f"Numeric tolerance removed {len(to_drop)} rows from {column} in {time.time()-start:.2f}s")
        return {"data": df_copy, "removed": len(to_drop)}

    @staticmethod
    def deduplicate_multi_column(df: pd.DataFrame, columns: List[str], inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        return Deduplicator.drop_exact_duplicates(df, subset=columns, inplace=inplace)

    @staticmethod
    def deduplicate_ignore_case(df: pd.DataFrame, column: str, inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        df_copy = df if inplace else df.copy()
        df_copy[column] = df_copy[column].astype(str).str.lower()
        return Deduplicator.drop_exact_duplicates(df_copy, subset=[column], inplace=True)

    @staticmethod
    def deduplicate_trimmed(df: pd.DataFrame, column: str, inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        df_copy = df if inplace else df.copy()
        df_copy[column] = df_copy[column].astype(str).str.strip()
        return Deduplicator.drop_exact_duplicates(df_copy, subset=[column], inplace=True)

    @staticmethod
    def deduplicate_combined_string(df: pd.DataFrame, columns: List[str], inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        df_copy = df if inplace else df.copy()
        combined = df_copy[columns].astype(str).agg('_'.join, axis=1)
        df_copy['combined_key'] = combined
        result = Deduplicator.drop_exact_duplicates(df_copy, subset=['combined_key'], inplace=True)
        result['data'] = result['data'].drop(columns=['combined_key'])
        return result

    @staticmethod
    def deduplicate_by_hash(df: pd.DataFrame, columns: List[str], inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        df_copy = df if inplace else df.copy()
        df_copy['_hash'] = df_copy[columns].astype(str).agg(lambda x: hash('_'.join(x)), axis=1)
        result = Deduplicator.drop_exact_duplicates(df_copy, subset=['_hash'], inplace=True)
        result['data'] = result['data'].drop(columns=['_hash'])
        return result

    @staticmethod
    def deduplicate_by_rounding(df: pd.DataFrame, column: str, decimals: int = 2, inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        df_copy = df if inplace else df.copy()
        df_copy[column] = df_copy[column].round(decimals)
        return Deduplicator.drop_exact_duplicates(df_copy, subset=[column], inplace=True)

    @staticmethod
    def deduplicate_keep_max(df: pd.DataFrame, value_column: str, subset: Optional[List[str]] = None, inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        df_copy = df if inplace else df.copy()
        df_sorted = df_copy.sort_values(value_column, ascending=False)
        return Deduplicator.keep_first_duplicates(df_sorted, subset=subset, inplace=True)

    @staticmethod
    def deduplicate_keep_min(df: pd.DataFrame, value_column: str, subset: Optional[List[str]] = None, inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        df_copy = df if inplace else df.copy()
        df_sorted = df_copy.sort_values(value_column, ascending=True)
        return Deduplicator.keep_first_duplicates(df_sorted, subset=subset, inplace=True)

    @staticmethod
    def deduplicate_across_sources(dfs: List[pd.DataFrame], subset: Optional[List[str]] = None, inplace: bool = False) -> Dict[str, Union[pd.DataFrame, int]]:
        start = time.time()
        combined_df = pd.concat(dfs, ignore_index=True)
        result = Deduplicator.drop_exact_duplicates(combined_df, subset=subset, inplace=inplace)
        logger.info(f"Deduplicated across {len(dfs)} sources, removed {result['removed']} duplicates in {time.time()-start:.2f}s")
        return result
            

def deduplicate_data(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Remove duplicates from a DataFrame.
    """
    return df.drop_duplicates(subset=subset)


