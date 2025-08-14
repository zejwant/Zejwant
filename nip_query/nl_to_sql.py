"""
nl_to_sql.py
------------

Unified translator for converting natural language queries to SQL.

This module:
    - Accepts plain-English input and classification metadata.
    - Uses `query_router` to decide between rule-based and LLM-based SQL generation.
    - Normalizes SQL formatting (capitalization, indentation).
    - Returns SQL string + parameter dict in a structured format.
    - Fully unit-testable with mock backends.

Author: Varun-engineer mode (20+ years experience)
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Tuple

from .llm_based import generate_sql    # OK if inside nip_query
# OR use absolute from main code:
from nip_query.llm_based import generate_sql
from nip_query.query_router import decide_route
from nip_query.llm_based import generate_sql as llm_based_sql
from nip_query.rule_based import generate_sql as rule_based_sql
from nip_query.result_formatter import format_sql
from nip_query.query_router import decide_route
from nip_query.sql_formatter import format_sql  # helper module to normalize formatting




logger = logging.getLogger(__name__)


def nl_to_sql(user_input: str, classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a natural language query into SQL based on classification metadata.

    Parameters
    ----------
    user_input : str
        The natural language query provided by the user.
    classification : Dict[str, Any]
        Metadata from query_classifier containing:
            - query_type: str
            - complexity: str ("low", "medium", "high")
            - sensitivity: str ("none", "pii", "financial")
            - any other domain-specific flags

    Returns
    -------
    Dict[str, Any]
        {
            "sql": str,               # Formatted SQL query
            "params": Dict[str, Any], # Parameters for parameterized execution
            "engine": str             # "rule_based" or "llm_based"
        }

    Raises
    ------
    ValueError
        If SQL generation fails in both rule-based and LLM-based approaches.
    """
    logger.debug("Starting NL-to-SQL translation", extra={"classification": classification})

    # Decide which engine to use
    engine_choice = decide_route(classification)
    logger.info(f"Routing to {engine_choice} engine", extra={"engine": engine_choice})

    try:
        if engine_choice == "rule_based":
            result = rule_based_sql(user_input, classification)
        elif engine_choice == "llm_based":
            result = llm_based_sql(user_input, classification)
        else:
            raise ValueError(f"Unknown engine choice: {engine_choice}")

        # Format SQL for consistency
        formatted_sql = format_sql(result["sql"])
        result["sql"] = formatted_sql
        result["engine"] = engine_choice

        logger.debug("SQL generation successful", extra={"sql": formatted_sql, "params": result["params"]})
        return result

    except Exception as e:
        logger.error("Primary engine failed. Attempting fallback...", exc_info=True)

        # Fallback: If rule-based failed, try LLM; if LLM failed, try rule-based
        fallback_engine = "llm_based" if engine_choice == "rule_based" else "rule_based"
        try:
            logger.info(f"Trying fallback engine: {fallback_engine}")
            if fallback_engine == "rule_based":
                result = rule_based_sql(user_input, classification)
            else:
                result = llm_based_sql(user_input, classification)

            formatted_sql = format_sql(result["sql"])
            result["sql"] = formatted_sql
            result["engine"] = fallback_engine
            return result
        except Exception:
            logger.critical("Both engines failed to generate SQL", exc_info=True)
            raise ValueError("Failed to generate SQL from natural language query")


# ---------- Unit Test Stubs ----------
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.DEBUG)

    mock_classification = {
        "query_type": "aggregation",
        "complexity": "low",
        "sensitivity": "none"
    }

    mock_query = "Show me the total sales by region for 2024."

    # Simulate with mock engines (replace with actual mocks in tests)
    try:
        result = nl_to_sql(mock_query, mock_classification)
        print(json.dumps(result, indent=2))
    except ValueError as e:
        print(f"Error: {e}")
      
# nip_query/nl_to_sql.py

def generate_sql(query: str) -> str:
    # local import prevents circular import errors
    from .query_router import decide_route

    route = decide_route(query)
    if route == "rule":
        return "SELECT * FROM my_table"  # example placeholder
    else:
        return "SELECT COUNT(*) FROM my_table"
        

def translate_nl_to_sql(nl_query: str):
    """
    Convert natural language query to SQL using router.
    """
    route = decide_route(nl_query)  # 'rule' or 'llm'
    if route == "rule":
        return rule_based_sql(nl_query), {}
    elif route == "llm":
        return llm_based_sql(nl_query), {}
    else:
        raise ValueError(f"Unknown route: {route}")
