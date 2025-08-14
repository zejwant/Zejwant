# nip_query/sql_formatter.py

def format_sql(sql: str) -> str:
    """
    Simple SQL formatter (placeholder for more advanced formatting)
    
    Args:
        sql (str): Raw SQL string
    
    Returns:
        str: Formatted SQL string
    """
    # Remove extra spaces and newlines
    formatted = " ".join(sql.strip().split())
    return formatted
  
