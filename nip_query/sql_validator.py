"""
sql_validator.py
----------------

Enterprise SQL validator to ensure only safe, authorized queries are executed.

Features:
    - Parses SQL using `sqlparse` for structural analysis.
    - Syntax validation hook (can be extended to actual DB dry-run checks).
    - Blocks dangerous commands (DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE).
    - Enforces allowed tables and columns from config/metadata.
    - Fully logged decision-making.
    - Unit test stubs for safe/unsafe SQL.

Author: Varun-engineer mode (20+ years experience)
"""

from __future__ import annotations
import logging
from typing import List, Dict, Optional

import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import Keyword, DML

logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
ALLOWED_TABLES = {
    "sales": ["id", "region", "amount", "date"],
    "customers": ["id", "name", "region", "signup_date"]
}
BLOCKED_KEYWORDS = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"}


def _extract_identifiers(parsed) -> List[str]:
    """Extract table/column identifiers from parsed SQL."""
    identifiers = []
    for token in parsed.tokens:
        if isinstance(token, Identifier):
            identifiers.append(token.get_real_name())
        elif isinstance(token, IdentifierList):
            for ident in token.get_identifiers():
                identifiers.append(ident.get_real_name())
        elif token.is_group:
            identifiers.extend(_extract_identifiers(token))
    return identifiers


def validate_sql(sql: str, allowed_tables: Optional[Dict[str, List[str]]] = None) -> bool:
    """
    Validate SQL query for safety and allowed access.

    Parameters
    ----------
    sql : str
        The SQL query string.
    allowed_tables : dict, optional
        Allowed tables and columns. Defaults to global ALLOWED_TABLES.

    Returns
    -------
    bool
        True if SQL is valid, False otherwise.

    Raises
    ------
    ValueError
        If SQL is unsafe or violates rules.
    """
    allowed_tables = allowed_tables or ALLOWED_TABLES
    logger.debug(f"Validating SQL: {sql}")

    # Quick keyword block
    upper_sql = sql.upper()
    for kw in BLOCKED_KEYWORDS:
        if kw in upper_sql:
            logger.warning(f"Blocked keyword detected: {kw}")
            raise ValueError(f"SQL contains blocked keyword: {kw}")

    # Parse SQL
    parsed_statements = sqlparse.parse(sql)
    if not parsed_statements:
        raise ValueError("Invalid SQL syntax: unable to parse.")

    for statement in parsed_statements:
        # Only allow SELECT
        dml_tokens = [token for token in statement.tokens if token.ttype is Keyword or token.ttype is DML]
        if any(token.value.upper() not in {"SELECT", "FROM", "WHERE", "GROUP", "ORDER", "BY", "LIMIT", "OFFSET"} for token in dml_tokens):
            logger.warning(f"Non-SELECT operation detected: {dml_tokens}")
            raise ValueError("Only SELECT operations are allowed.")

        # Extract identifiers
        identifiers = _extract_identifiers(statement)
        logger.debug(f"Identifiers found: {identifiers}")

        # Table and column check
        for ident in identifiers:
            if ident is None:
                continue
            found_table = False
            for table, cols in allowed_tables.items():
                if ident == table or ident in cols:
                    found_table = True
                    break
            if not found_table:
                logger.error(f"Unauthorized table/column: {ident}")
                raise ValueError(f"Unauthorized table/column: {ident}")

    logger.info("SQL validation passed.")
    return True


# ---------- Unit Test Stubs ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    safe_sql = "SELECT id, amount FROM sales WHERE region = 'North' LIMIT 5"
    unsafe_sql = "DELETE FROM sales WHERE id = 1"

    try:
        print("Safe SQL test:", validate_sql(safe_sql))
    except Exception as e:
        print("Safe SQL failed:", e)

    try:
        print("Unsafe SQL test:", validate_sql(unsafe_sql))
    except Exception as e:
        print("Unsafe SQL failed as expected:", e)

# nip_query/sql_validator_helper.py

def validate_sql(sql: str) -> bool:
    """
    Very basic SQL validation.
    Returns True if the query starts with SELECT, INSERT, UPDATE, or DELETE.
    """
    sql = sql.strip().lower()
    return sql.startswith(("select", "insert", "update", "delete"))
    
