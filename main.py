from fastapi import FastAPI
from pydantic import BaseModel
from orchestrator import run_query
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
    

class QueryRequest(BaseModel):
    query: str



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    # pass contents to ingestion pipeline
    return {"filename": file.filename}
    
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
engine = create_engine("sqlite:///mydb.db")
metadata = MetaData()

def store_data(data):
    # dynamically create table based on data schema
    pass
    



@app.post("/query")
def execute(request: QueryRequest):
    return run_query(request.query)
  
