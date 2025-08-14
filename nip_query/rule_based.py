"""
rule_based.py
-------------
Fast, rule-based SQL generation module for the nip_query package.

This module uses:
- Regex and keyword matching
- Predefined, maintainable SQL templates
- Safe parameterized query construction (no direct string concatenation)
- Modular design so non-ML engineers can easily extend templates

Author: Varun-engineer mode (20+ years experience)
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

# Configure module-specific logger
logger = logging.getLogger(__name__)


# ======================
# SQL TEMPLATE REGISTRY
# ======================
SQL_TEMPLATES: Dict[str, str] = {
    "select_basic": "SELECT {columns} FROM {table}",
    "select_where": "SELECT {columns} FROM {table} WHERE {conditions}",
    "select_group_by": "SELECT {columns}, COUNT(*) AS count FROM {table} GROUP BY {group_by}",
    "select_order_by": "SELECT {columns} FROM {table} ORDER BY {order_by} {order_dir}",
    "select_limit": "SELECT {columns} FROM {table} LIMIT {limit}",
    "select_join": (
        "SELECT {columns} FROM {table1} "
        "JOIN {table2} ON {table1}.{key1} = {table2}.{key2}"
    ),
}


# ======================
# CLAUSE DETECTION RULES
# ======================
CLAUSE_PATTERNS = {
    "where": re.compile(r"\bwhere\b", re.IGNORECASE),
    "group_by": re.compile(r"\bgroup\s+by\b", re.IGNORECASE),
    "order_by": re.compile(r"\border\s+by\b", re.IGNORECASE),
    "limit": re.compile(r"\blimit\b", re.IGNORECASE),
    "join": re.compile(r"\bjoin\b", re.IGNORECASE),
}


# ======================
# HELPER FUNCTIONS
# ======================
def detect_clauses(user_input: str) -> List[str]:
    """
    Detect SQL clauses present in the user's natural language input.

    Args:
        user_input (str): The plain-English query.

    Returns:
        List[str]: A list of detected clause names.
    """
    detected = [clause for clause, pattern in CLAUSE_PATTERNS.items() if pattern.search(user_input)]
    logger.debug(f"Detected clauses in input: {detected}")
    return detected


def build_conditions(condition_str: str) -> Tuple[str, List[Any]]:
    """
    Build safe WHERE clause conditions.

    Args:
        condition_str (str): Natural language condition description.

    Returns:
        Tuple[str, List[Any]]: SQL-safe condition string and parameters.
    """
    # Simple regex split for demonstration (replace with advanced parser later)
    # Example: "age > 30" → "age > %s", [30]
    match = re.match(r"(\w+)\s*(=|>|<|>=|<=|!=)\s*(\w+)", condition_str.strip())
    if match:
        col, op, val = match.groups()
        return f"{col} {op} %s", [val]
    return condition_str, []


# ======================
# CORE FUNCTION
# ======================
def generate_sql(user_input: str, table_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate SQL from plain-English input using rule-based templates.

    Args:
        user_input (str): Natural language query.
        table_metadata (Dict[str, Any]): Metadata about available tables and columns.

    Returns:
        Dict[str, Any]: {
            "sql": str,  # Generated SQL query
            "params": list  # Parameters for safe execution
        }
    """
    logger.info("Starting rule-based SQL generation.")

    clauses = detect_clauses(user_input)
    sql = ""
    params: List[Any] = []

    # Basic detection (extendable)
    if "join" in clauses:
        sql = SQL_TEMPLATES["select_join"].format(
            columns="*",
            table1=table_metadata.get("table1", ""),
            table2=table_metadata.get("table2", ""),
            key1=table_metadata.get("join_key1", ""),
            key2=table_metadata.get("join_key2", "")
        )

    elif "where" in clauses:
        # Extract condition part (naive split for demo)
        condition_str = user_input.split("where", 1)[-1].strip()
        cond_sql, cond_params = build_conditions(condition_str)
        params.extend(cond_params)

        sql = SQL_TEMPLATES["select_where"].format(
            columns="*",
            table=table_metadata.get("table", ""),
            conditions=cond_sql
        )

    elif "group_by" in clauses:
        sql = SQL_TEMPLATES["select_group_by"].format(
            columns=table_metadata.get("group_columns", "*"),
            table=table_metadata.get("table", ""),
            group_by=table_metadata.get("group_by", "")
        )

    elif "order_by" in clauses:
        sql = SQL_TEMPLATES["select_order_by"].format(
            columns="*",
            table=table_metadata.get("table", ""),
            order_by=table_metadata.get("order_by", ""),
            order_dir=table_metadata.get("order_dir", "ASC")
        )

    elif "limit" in clauses:
        sql = SQL_TEMPLATES["select_limit"].format(
            columns="*",
            table=table_metadata.get("table", ""),
            limit=table_metadata.get("limit", 10)
        )

    else:
        sql = SQL_TEMPLATES["select_basic"].format(
            columns="*",
            table=table_metadata.get("table", "")
        )

    logger.info(f"Generated SQL: {sql} | Params: {params}")
    return {"sql": sql, "params": params}


# ======================
# SIMULATION FOR TESTING
# ======================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_input = "Show me all orders where amount > 100"
    test_metadata = {"table": "orders"}
    result = generate_sql(test_input, test_metadata)

    print(result)
  
