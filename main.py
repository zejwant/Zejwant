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

from fastapi import FastAPI, File, UploadFile
import pandas as pd
from cleaning.validation import DataValidatorV1
from storage.sql_manager import store_dataframe

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename

    # 1️⃣ Detect file type
    if filename.endswith(".csv"):
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(pd.io.common.BytesIO(contents))
    elif filename.endswith(".json"):
        df = pd.read_json(pd.io.common.BytesIO(contents))
    else:
        return {"error": f"Unsupported file type: {filename}"}

    # 2️⃣ Run validation
    validator = DataValidatorV1()
    schema = {
        "id": {"type": "numeric", "min": 1},
        "name": {"type": "string", "pattern": r"^[A-Za-z ]+$"}
    }
    validation_report = validator.batch_validate(df, schema)

    # 3️⃣ Store in SQL
    store_dataframe(df, table_name="uploaded_data")

    # 4️⃣ Return feedback
    return {
        "filename": filename,
        "rows_uploaded": len(df),
        "validation_report": validation_report
    }
    





