# nip_query/__init__.py

"""
NIP Query Package
-----------------
Enterprise-grade natural language to SQL (NL→SQL) query engine.

Responsibilities:
- Translate natural language questions into SQL queries
- Validate and optimize generated SQL
- Execute queries safely against connected databases
- Format results into plain-English insights or structured output
- Logging and monitoring for all query operations
"""

import logging
from typing import Any, Dict, Optional

# Core modules
from . import nl_to_sql
from . import query_processor
from . import query_router
from . import sql_templates
from . import validation
from . import result_formatter
from . import utils

# Initialize logger
logger = logging.getLogger("nip_query")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# =========================
# High-Level API
# =========================

def ask_question(question: str, user_context: Optional[Dict[str, Any]] = None) -> Any:
    """
    Translate a natural language question into SQL, validate, execute, and return formatted results.

    Args:
        question (str): Natural language question to ask.
        user_context (Optional[Dict[str, Any]]): Optional context such as user, database, schema, or table hints.

    Returns:
        Any: Structured result or formatted insight (e.g., dict, DataFrame, plain-English summary).
    """
    logger.info(f"Received question: {question}")

    try:
        # Step 1: Translate NL → SQL
        sql_query = nl_to_sql.translate(question, context=user_context)
        logger.info(f"Translated SQL: {sql_query}")

        # Step 2: Validate SQL
        validation.validate_sql(sql_query)

        # Step 3: Optimize and preprocess query
        processed_query = query_processor.process(sql_query)

        # Step 4: Route query to the appropriate database
        result_set = query_router.execute(processed_query, context=user_context)

        # Step 5: Format results into structured output or plain-English
        formatted_result = result_formatter.format_result(result_set, context=user_context)

        logger.info("Query executed and results formatted successfully")
        return formatted_result

    except Exception as e:
        logger.error(f"Failed to execute question '{question}': {e}")
        raise
