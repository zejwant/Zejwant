# nip_query/sql_validator_helper.py

def validate_sql(sql: str) -> bool:
    """
    Very basic SQL validation placeholder.
    Only checks if it starts with SELECT/INSERT/UPDATE/DELETE.
    """
    sql = sql.strip().lower()
    return sql.startswith(("select", "insert", "update", "delete"))
  
