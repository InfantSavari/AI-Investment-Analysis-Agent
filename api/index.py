from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os

# Add the parent directory to sys.path so we can import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import run_analysis

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agentic AI API")

# Add CORS middleware to allow requests from any frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

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

# Mount the Next.js static HTML output if it exists
frontend_build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
if os.path.exists(frontend_build_dir):
    app.mount("/", StaticFiles(directory=frontend_build_dir, html=True), name="static")
else:
    @app.get("/")
    def read_root():
        return {"message": "API is running, but Next.js frontend build ('out' directory) was not found."}
