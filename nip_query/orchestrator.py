"""
orchestrator.py
----------------
Enterprise NL → SQL runner: full pipeline

1. Accepts user input (plain-English query)
2. Classifies query type & complexity
3. Routes to rule-based or LLM SQL generator
4. Validates SQL safety
5. Executes SQL on PostgreSQL
6. Formats results (text summary, charts, downloadable files)

Author: Varun-engineer mode (20+ years experience)
"""

from nip_query import nl_to_sql
from sql_validator import validate_sql
from query_processor import execute_query
from result_formatter import summarize_dataframe, generate_chart, export_dataframe
from config import get_settings

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def run_query(user_query: str, page: int = 1, page_size: int = 1000, lang: str = "en") -> dict:
    """
    Full NL → SQL pipeline execution.
    
    Parameters
    ----------
    user_query : str
        Natural language query.
    page : int
        Page number for pagination.
    page_size : int
        Number of rows per page.
    lang : str
        Language code for summaries (e.g., 'en', 'fr').
    
    Returns
    -------
    dict
        {
            "sql": str,
            "params": dict,
            "dataframe": pd.DataFrame,
            "summary": str,
            "charts": list of chart buffers,
            "downloads": dict of CSV/Excel bytes
        }
    """
    settings = get_settings()

    # Step 1: Translate NL → SQL
    try:
        sql, params = nl_to_sql.translate_nl_to_sql(user_query)
        logger.info(f"Generated SQL: {sql} with params {params}")
    except Exception as e:
        logger.error(f"Failed to generate SQL: {e}")
        raise

    # Step 2: Validate SQL
    try:
        validate_sql(sql)
        logger.info("SQL validation passed")
    except Exception as e:
        logger.error(f"SQL validation failed: {e}")
        raise

    # Step 3: Execute SQL
    try:
        df = execute_query(sql, params=params, page=page, page_size=page_size)
        logger.info(f"Query returned {len(df)} rows")
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        raise

    # Step 4: Format results
    summary = summarize_dataframe(df, lang=lang)
    
    charts = []
    if not df.empty and 'date' in df.columns and 'amount' in df.columns:
        try:
            chart_buf = generate_chart(df, x='date', y='amount', chart_type='line', interactive=False)
            charts.append(chart_buf)
        except Exception as e:
            logger.warning(f"Failed to generate chart: {e}")

    downloads = {}
    try:
        downloads['csv'] = export_dataframe(df, format='csv')
        downloads['excel'] = export_dataframe(df, format='excel')
    except Exception as e:
        logger.warning(f"Failed to generate downloads: {e}")

    return {
        "sql": sql,
        "params": params,
        "dataframe": df,
        "summary": summary,
        "charts": charts,
        "downloads": downloads
    }


# ---------- Test Run ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_query = "Show total sales per region for last month"
    result = run_query(test_query, page=1, page_size=10)
    
    print("Summary:\n", result["summary"])
    print("DataFrame:\n", result["dataframe"].head())
    print("SQL used:\n", result["sql"])
                                       
