from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add the parent directory to sys.path so we can import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import run_analysis

app = FastAPI(title="Agentic AI API")

class AnalysisRequest(BaseModel):
    ticker: str

@app.post("/api/analyze")
async def analyze_ticker(request: AnalysisRequest):
    try:
        # Execute the LangGraph analysis pipeline
        report = await run_analysis(request.ticker.upper())
        return {"report": report}
    except Exception as e:
        print(f"Error analyzing {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
