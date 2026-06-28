# AI Investment Signal Generating Agent

## Overview — What It Does
This project is an AI-driven, multi-agent investment analysis tool. It takes a stock ticker and runs a comprehensive, institutional-grade analysis using LangGraph. The system orchestrates multiple specialized AI agents (Qualitative, Quantitative, and Risk Analysts) to evaluate a company from different angles. An Arbiter agent then synthesizes these perspectives mathematically, and a Decision Maker agent formats a final markdown research report with a final verdict (INVEST, HOLD, or PASS).

## How to Run It — Setup and Run Steps

### Prerequisites
- Python 3.10+
- Node.js (for Next.js frontend)

### Environment Variables
Create a `.env` file in the root directory with the following keys:
```env
GOOGLE_API_KEY=your_gemini_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
```

### Backend Setup (FastAPI & LangGraph)
1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend API server using Uvicorn:
   ```bash
   uvicorn api.index:app --reload
   ```

### Frontend Setup (Next.js)
1. Install Node dependencies:
   ```bash
   npm install
   ```
2. Run the development server:
   ```bash
   npm run dev
   ```

*(Alternatively, you can test just the graph locally by running `python graph.py`)*

## How It Works — Approach and Architecture
The system uses a graph-based agent orchestration model built with **LangGraph**, powered by Google's **Gemini 3.5 Flash**. The architecture mimics an institutional hedge fund's research team:

1. **Data Layer**: `main.py` fetches real-time financial statements (Income Statement, Balance Sheet, Cash Flow) and news sentiment via the Alpha Vantage API.
2. **Qualitative Agent**: Analyzes news sentiment and the company's business model to assess the strength of its "Moat" and strategic positioning.
3. **Quantitative Agent**: Performs rigorous fundamental analysis on financial statements to classify the valuation and determine a holding duration.
4. **Risk Agent**: Acts as an adversarial short-seller, hunting for idiosyncratic and systemic threats to the investment thesis.
5. **Synthesizer (The Arbiter)**: Ingests the scores (and confidences) from the three analysts and executes a weighted mathematical formula (Quant 40%, Qual 30%, Risk 30%). It can also trigger a "Fatal Flaw Penalty" if the risk is overwhelmingly high.
6. **Decision Maker**: Compiles the graph state into a polished markdown report, issuing a final verdict based on the Arbiter's score.

## Key Decisions & Trade-offs
- **Model Choice**: Selected Gemini-3.5-Flash for all agents due to its large context window (helpful for reading financial statements) and fast inference times. 
- **Deterministic Synthesis**: The Arbiter uses `temperature=0.0` and a strict mathematical formula to synthesize scores. This prevents LLM hallucination during the critical step of weighing conflicting opinions and ensures the final score is trackable.
- **SQLite Caching**: Financial data changes slowly (quarterly/daily). We implemented a 1-day SQLite cache for generated reports (`cache.py`) to reduce expensive LLM calls and API limits.
- **What was left out**: We excluded real-time price charting, technical analysis, and complex options chains to focus strictly on fundamental, long-term value investing principles.

## Example Runs

*(Here is a representation of typical agent output behaviors based on the system's logic)*

- **AAPL (Apple)**: The Qualitative Agent highly rates the ecosystem moat and brand strength. The Quant agent might flag it as "FAIRLY VALUED" or "OVERVALUED" due to high P/E multiples relative to growth. The resulting verdict is often a **HOLD**, citing the need for a stronger catalyst or better margin of safety.
- **NVDA (Nvidia)**: Qualitative and Quantitative scores soar due to monopoly-like AI chip positioning and explosive revenue growth. However, the Risk Agent flags high valuation multiples and geopolitical supply chain risks. The Arbiter balances these, typically resulting in an **INVEST** or **HOLD** depending on the exact day's sentiment and P/E ratio.
- **INTC (Intel)**: Risk Agent triggers the "Fatal Flaw Penalty" due to eroding market share, negative free cash flow, and turnaround uncertainties, overpowering the Quant agent's "DEEP VALUE" classification. The final verdict is heavily pushed to **PASS**.

## What I Would Improve with More Time
- **Expanded Data Sources**: Integrate EDGAR SEC filings (10-K, 10-Q) parsing for deeper fundamental insights, and transcripts from earnings calls.
- **Multi-Model Routing**: Use a stronger reasoning model (e.g., Gemini 1.5 Pro) specifically for the Risk and Synthesizer agents, while keeping Flash for the data-heavy Quant/Qual agents.
- **Portfolio Optimization**: Expand the tool to analyze a basket of stocks simultaneously and suggest portfolio weights.
- **Interactive UI**: Enhance the Next.js frontend to show the "live thought process" of each agent executing in real-time, displaying their individual confidence scores before the final report is generated.
