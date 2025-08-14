# main.py
from pydantic import BaseModel
from orchestrator import run_query
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float



from fastapi import FastAPI, UploadFile, File
from cleaning.cleaner_controller import CleanerController
from ingestion.upload_pipeline import upload_file
from storage.sql_manager import SQLManager
from nip_query.nl_to_sql import NLtoSQL
from fastapi.responses import JSONResponse

app = FastAPI(title="Data Platform Test API")

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    # Step 1: Save raw file temporarily
    content = await file.read()
    temp_path = f"temp_uploads/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(content)

    # Step 2: Ingest raw file
    data = upload_file(temp_path)  # calls the appropriate connector

    # Step 3: Clean data
    cleaner = CleanerController()
    cleaned_data = cleaner.clean(data, file.filename)

    # Step 4: Store in SQL
    sql_manager = SQLManager()
    table_name = sql_manager.save_dataframe(cleaned_data, file.filename)

    return {"status": "success", "table": table_name}
    

@app.post("/query/")
async def query_nl(question: str):
    nl_engine = NLtoSQL()
    sql_query = nl_engine.translate(question)
    
    sql_manager = SQLManager()
    results = sql_manager.execute_query(sql_query)
    
    return JSONResponse(content={"question": question, "sql": sql_query, "results": results})
    

