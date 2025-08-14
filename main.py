from fastapi import FastAPI
from pydantic import BaseModel
from orchestrator import run_query

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def execute(request: QueryRequest):
    return run_query(request.query)
  
