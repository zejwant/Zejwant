from nip_query import nl_to_sql
from sql_validator import validate_sql
from query_processor import execute_query
from result_formatter import summarize_dataframe, generate_chart, export_dataframe
from config import get_settings

import logging

logger = logging.getLogger(__name__)

def run_query(user_query: str):
    settings = get_settings()
    
    # Step 1: NL → SQL
    sql, params = nl_to_sql.translate_nl_to_sql(user_query)
    
    # Step 2: Validate SQL
    validate_sql(sql)
    
    # Step 3: Execute SQL
    df = execute_query(sql, params=params)
    
    # Step 4: Format results
    summary = summarize_dataframe(df)
    charts = []
    downloads = {
        "csv": export_dataframe(df, format='csv'),
        "excel": export_dataframe(df, format='excel')
    }
    
    return {
        "sql": sql,
        "params": params,
        "dataframe": df,
        "summary": summary,
        "charts": charts,
        "downloads": downloads
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_query = "Show total sales last month by region"
    output = run_query(test_query)
    print("Summary:\n", output["summary"])
    print("DataFrame:\n", output["dataframe"].head())
  
