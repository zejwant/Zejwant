# main.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from orchestrator import run_query
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float


# SQLite engine
engine = create_engine("sqlite:///mydb.db")
metadata = MetaData()



app = FastAPI(title="Enterprise Data Platform")

@app.get("/")
def read_root():
    return {"message": "Platform ready"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    table_name, row_count = process_file(file)
    return {"filename": file.filename, "rows": row_count, "table": table_name}

@app.post("/query")
def execute_query(query: str):
    return run_query(query)
    







# ----- Cleaning functions -----
def clean_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def clean_strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    return df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

def clean_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna("N/A", inplace=True)
        else:
            df[col].fillna(0, inplace=True)
    return df

CLEANING_METHODS = [
    clean_remove_duplicates,
    clean_strip_whitespace,
    clean_fill_missing
]


# ----- Utility to detect file type -----
def read_file(file: UploadFile) -> pd.DataFrame:
    if file.filename.endswith(".csv"):
        return pd.read_csv(file.file)
    elif file.filename.endswith((".xls", ".xlsx")):
        return pd.read_excel(file.file)
    else:
        raise ValueError(f"Unsupported file type: {file.filename}")


# ----- Store DataFrame dynamically in SQL -----
def store_data(df: pd.DataFrame, table_name: str):
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    return f"Stored {len(df)} rows in table '{table_name}'"


# ----- FastAPI Endpoints -----
@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 1. Read file
        df = read_file(file)
        
        # 2. Apply cleaning methods
        for func in CLEANING_METHODS:
            df = func(df)
        
        # 3. Store in SQL
        table_name = file.filename.replace(".", "_")
        status = store_data(df, table_name)
        
        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "table_stored": table_name,
            "status": status
        }
    except Exception as e:
        return {"error": str(e)}


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def execute(request: QueryRequest):
    return run_query(request.query)
    
