from fastapi import FastAPI
from pydantic import BaseModel
from orchestrator import run_query
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
    

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def execute(request: QueryRequest):
    return run_query(request.query)
  
