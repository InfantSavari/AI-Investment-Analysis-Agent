import os
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Import the data fetching function we created previously
from main import fetch_alpha_vantage

load_dotenv()

AGENT_SLEEP_TIME = 15


def _get_text(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)

# Define the state that our graph will pass between agents
class GraphState(TypedDict):
    ticker: str
    qualitative_analysis: str
    quantitative_analysis: str
    risk_analysis: str
    synthesized_report: str
    final_report: str

def get_llm(temperature: float, model: str = 'gemini-3.5-flash'):
    """Helper to get a fresh Gemini or Groq client with automatic retries and model fallbacks."""
    provider = os.getenv("PRIMARY_PROVIDER", "gemini").lower()
    
    if provider == "groq":
        # Only llama-3.1-8b-instant is allowed on this Groq project
        primary = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=temperature,
            max_retries=3,
        )
        
        # Fallback to Gemini if Groq fails
        fallbacks = []
        try:
            fallbacks.append(
                ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    max_retries=3,
                )
            )
        except Exception:
            pass
            
        return primary.with_fallbacks(fallbacks) if fallbacks else primary
        
    else:
        # Default Gemini path
        primary = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_retries=3,
        )
        # Define fallback models to handle 503 Service Unavailable or high demand
        if 'pro' in model:
            gemini_fallbacks = [
                ChatGoogleGenerativeAI(model='gemini-3.1-pro', temperature=temperature, max_retries=3),
                ChatGoogleGenerativeAI(model='gemini-3.1-flash', temperature=temperature, max_retries=3),
                ChatGoogleGenerativeAI(model='gemini-1.5-flash-lite', temperature=temperature, max_retries=3)
            ]
        else:
            gemini_fallbacks = [
                ChatGoogleGenerativeAI(model='gemini-3.1-flash', temperature=temperature, max_retries=3),
                ChatGoogleGenerativeAI(model='gemini-1.5-flash-lite', temperature=temperature, max_retries=3)
            ]
            
        # Append Groq llama-3.1-8b-instant as the final fallback in case of Gemini rate limit exhaustion
        groq_fallbacks = []
        try:
            groq_fallbacks.append(
                ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=temperature,
                    max_retries=3,
                )
            )
        except Exception:
            pass
            
        fallbacks = gemini_fallbacks + groq_fallbacks
        return primary.with_fallbacks(fallbacks)

async def qualitative_agent_node(state: GraphState):
    """Qualitative Analysis Agent using Gemini"""
    ticker = state["ticker"]
    print(f"[{ticker}] Qualitative Agent is running...")
    
    # Fetch news/sentiment data (limit to 10 articles to avoid token bloat)
    news_data = await fetch_alpha_vantage("NEWS_SENTIMENT", ticker, limit="10", sort="LATEST")
    context = str(news_data)[:10000] # Trim to avoid context limits if extremely large
    
    # Use Gemini model (Google GenAI) with automatic fallbacks
    llm = get_llm(temperature=0.2)
    print(f"[{ticker}] Sleeping for {AGENT_SLEEP_TIME} seconds to respect API limits...")
    await asyncio.sleep(AGENT_SLEEP_TIME)
    
    prompt = f"""<role>
You are the Lead Strategic Equities Analyst. You specialize in business models, competitive moats, and market narratives. You look beyond the spreadsheet to understand product-market fit, management execution, and brand power.
</role>

<context>
You will be provided with:
1. {ticker}'s business description and sector.
2. A summary of recent news headlines and the Market Sentiment Score for {ticker}.

Data:
{context}
</context>

<mandate>
Analyze {ticker}'s structural advantages using Porter's Five Forces as a mental model. 
1. Identify {ticker}'s "Moat" (e.g., switching costs, network effects, brand equity, cost advantage).
2. Synthesize the recent news headlines to determine if the company is currently executing on its strategy or facing operational friction.
3. Determine if the business model is highly scalable or capital-intensive.
</mandate>

<rules>
- Do not perform mathematical valuation; leave that to the Quant team.
- If the news headlines indicate a new product launch, leadership change, or positive market tailwinds, factor this heavily into your growth narrative.
- If {ticker}'s business model is vague or confusing, reduce your Confidence Score.
</rules>

<output_format>
Return a JSON object matching this schema:
{{
  "moat_strength": "[Weak / Moderate / Strong / Monopoly]",
  "analysis_text": "[2-3 paragraphs explaining the growth narrative, business model scalability, and strategic positioning]",
  "qualitative_score": [Float 1.0 to 10.0, where 10 is an impenetrable moat],
  "confidence_score": [Float 0.0 to 1.0, based on clarity of the business model and news relevance]
}}
</output_format>
"""
    
    response = await llm.ainvoke([
        SystemMessage(content="You are a top-tier stock market qualitative analyst. Provide a concise but insightful analysis."),
        HumanMessage(content=prompt)
    ])
    
    return {"qualitative_analysis": _get_text(response.content)}

async def quantitative_agent_node(state: GraphState):
    """Quantitative Analysis Agent using Gemini (Independent context)"""
    ticker = state["ticker"]
    print(f"[{ticker}] Quantitative Agent is running...")
    
    # Fetch financial statements and technicals concurrently
    results = await asyncio.gather(
        fetch_alpha_vantage("INCOME_STATEMENT", ticker),
        fetch_alpha_vantage("BALANCE_SHEET", ticker),
        fetch_alpha_vantage("CASH_FLOW", ticker),
        fetch_alpha_vantage("GLOBAL_QUOTE", ticker),
        fetch_alpha_vantage("RSI", ticker, interval="daily", time_period="14", series_type="close"),
        fetch_alpha_vantage("SMA", ticker, interval="daily", time_period="50", series_type="close"),
        fetch_alpha_vantage("MACD", ticker, interval="daily", series_type="close")
    )
    income_data, balance_data, cashflow_data, quote_data, rsi_data, sma_data, macd_data = results
    context = f"Income Statement:\n{str(income_data)[:5000]}\n\nBalance Sheet:\n{str(balance_data)[:5000]}\n\nCash Flow:\n{str(cashflow_data)[:5000]}\n\nQuote (Current Price):\n{str(quote_data)[:500]}\n\nRSI (14-day):\n{str(rsi_data)[:500]}\n\nSMA (50-day):\n{str(sma_data)[:500]}\n\nMACD:\n{str(macd_data)[:500]}"
    
    # Use a separate Gemini instance with automatic fallbacks
    llm = get_llm(temperature=0.2)
    print(f"[{ticker}] Sleeping for {AGENT_SLEEP_TIME} seconds to respect API limits...")
    await asyncio.sleep(AGENT_SLEEP_TIME)
    
    prompt = f"""<role>
You are the Lead Quantitative Analyst for an institutional hedge fund. You are entirely objective, emotionally detached, and rely strictly on empirical financial data. You do not care about the company's "story" or "vision"—you only care about the numbers.
</role>

<context>
You will be provided with:
1. A Markdown table of {ticker}'s latest financial statements (Revenue, Net Income, Debt, etc.).
2. The current Market Sentiment Score (1-10) derived from recent news for {ticker}.
3. Current price and technical indicators (RSI, SMA).

Financial & Technical Data:
{context}
</context>

<mandate>
Step 1: Calculate historical trends (YoY growth, margin expansion/compression).
Step 2: Perform relative valuation (analyze P/E, PEG ratio, FCF yield) to classify {ticker} as strictly one of the following: [DEEP VALUE, FAIRLY VALUED, OVERVALUED].
Step 3: Analyze technical indicators (RSI, SMA) to determine if the current price is a good entry point or if the buyer should wait for a better target price.
Step 4: Calculate the Recommended Holding Duration. 
- If OVERVALUED: 0 Months.
- If DEEP VALUE & Low Sentiment (< 4): 3 to 5 Years.
- If DEEP VALUE & High Sentiment (> 7): 6 to 12 Months.
- If FAIRLY VALUED & Neutral Sentiment: 1 to 3 Years.
- If FAIRLY VALUED & High Sentiment: Momentum Hold (Review in 3 Months).
</mandate>

<rules>
- Do not speculate on future products. Stick to historical/current financials and technicals.
- If financial data is missing or highly volatile, you MUST penalize your Confidence Score.
</rules>

<output_format>
Return a JSON object matching this schema:
{{
  "valuation_classification": "[Classification]",
  "holding_duration": "[Duration]",
  "current_price_assessment": "[Brief sentence on whether current price is good or wait for a target price]",
  "target_entry_price": "[Float or string specifying target price]",
  "analysis_text": "[2-3 paragraphs of rigorous financial and technical analysis detailing margins, valuation metrics, and technical entry points]",
  "quantitative_score": [Float 1.0 to 10.0, where 10 is flawless financial health],
  "confidence_score": [Float 0.0 to 1.0, based on data completeness]
}}
</output_format>"""
    
    response = await llm.ainvoke([
        SystemMessage(content="You are a stock market quantitative analyst. Base your analysis solely on the provided numerical data."),
        HumanMessage(content=prompt)
    ])
    
    return {"quantitative_analysis": _get_text(response.content)}

async def risk_agent_node(state: GraphState):
    """Risk Analysis Agent using Groq"""
    ticker = state["ticker"]
    print(f"[{ticker}] Risk Agent is running...")
    
    # Risk agent might look at recent market news for volatility and headwinds (limit to 10 articles to avoid token bloat)
    news_data = await fetch_alpha_vantage("NEWS_SENTIMENT", ticker, limit="10", sort="LATEST")
    context = str(news_data)[:8000]
    qual_analysis = state.get("qualitative_analysis", "No qualitative analysis provided.")
    
    # Use Gemini model with automatic fallbacks
    llm = get_llm(temperature=0.2)
    print(f"[{ticker}] Sleeping for {AGENT_SLEEP_TIME} seconds to respect API limits...")
    await asyncio.sleep(AGENT_SLEEP_TIME)
    
    prompt = f"""<role>
You are the Chief Risk Officer (CRO) and an adversarial short-seller. Your explicit goal is to destroy the investment thesis for {ticker}. You assume every company is a value trap or a structural failure waiting to happen until proven otherwise.
</role>

<context>
You will receive:
1. Raw financial data and recent news headlines for {ticker}.
2. The optimistic analysis from the Qualitative Agent.

Data:
{context}

Qualitative Analysis:
{qual_analysis}
</context>

<mandate>
Hunt for fatal flaws by analyzing two buckets of risk for {ticker}:
1. Idiosyncratic Risk (Company-specific): Aggressive accounting, high leverage, customer concentration, key-man risk, or eroding margins.
2. Systemic & Macro Risk: Interest rate sensitivity, geopolitical exposure, supply chain fragility, or looming regulatory/antitrust crackdowns.
</mandate>

<rules>
- You are adversarial. Do not write about {ticker}'s upside. 
- Scan the news specifically for lawsuits, downgrades, executive departures, or competitor breakthroughs.
- If you find a massive, existential threat in the news or debt profile, you MUST output a low Risk Score (<= 3) and a high Confidence Score (>= 0.85).
- (Note: A low score means the stock is highly risky/bad. A high score means it is safe.)
</rules>

<output_format>
Return a JSON object matching this schema:
{{
  "top_three_threats": ["[Threat 1]", "[Threat 2]", "[Threat 3]"],
  "analysis_text": "[2-3 paragraphs of ruthless critique outlining exactly how this investment could fail]",
  "risk_score": [Float 1.0 to 10.0, where 1.0 means imminent bankruptcy/catastrophe and 10.0 means risk-free],
  "confidence_score": [Float 0.0 to 1.0, based on the severity and proof of the threats found]
}}
</output_format>"""
    
    response = await llm.ainvoke([
        SystemMessage(content="You are a stock market risk analyst. Your job is to identify red flags and potential pitfalls for investors."),
        HumanMessage(content=prompt)
    ])
    
    return {"risk_analysis": _get_text(response.content)}

async def synthesizer_node(state: GraphState):
    """Synthesizer (The Arbiter) Agent using Gemini"""
    ticker = state["ticker"]
    qual = state.get("qualitative_analysis", "")
    quant = state.get("quantitative_analysis", "")
    risk = state.get("risk_analysis", "")
    print(f"[{ticker}] The Arbiter (Synthesizer) is running...")
    
    # Use Gemini model. Temperature 0.0 for strict logical workflow, with automatic fallbacks
    llm = get_llm(temperature=0.0)
    print(f"[{ticker}] Sleeping for {AGENT_SLEEP_TIME} seconds to respect API limits...")
    await asyncio.sleep(AGENT_SLEEP_TIME)
    
    prompt = f"""<role>
You are the Portfolio Synthesizer (The Arbiter). You are a strict logic gate. You do not generate original financial analysis; your job is to ingest the scores from your specialized agents regarding {ticker}, execute a weighted mathematical formula, and resolve contradictions.
</role>

<context>
You will receive a JSON state containing the outputs of the Quant Agent, Qual Agent, and Risk Agent for {ticker}. Each output contains a Score (1-10) and a Confidence Rating (0.0-1.0).

Quant Agent Output:
{quant}

Qual Agent Output:
{qual}

Risk Agent Output:
{risk}
</context>

<mandate>
Step 1: Calculate the Confidence-Weighted Average Score using these base weights: Quant (0.4), Qual (0.3), Risk (0.3).
Formula: [(Quant_S * Quant_C * 0.4) + (Qual_S * Qual_C * 0.3) + (Risk_S * Risk_C * 0.3)] / [(Quant_C * 0.4) + (Qual_C * 0.3) + (Risk_C * 0.3)]

Step 2: Evaluate the Fatal Flaw Penalty. 
IF the Risk Agent's Score is <= 3.0 AND the Risk Agent's Confidence is >= 0.85:
- The investment in {ticker} is deemed too dangerous regardless of the math. 
- You MUST multiply the calculated Confidence-Weighted Average Score by 0.5.
- Set `penalty_triggered` to TRUE.
</mandate>

<rules>
- Accept the agents' scores as absolute truth. Do not re-analyze {ticker}.
- If an agent's confidence is below 0.5, explicitly state in your rationale that their input was largely discounted.
</rules>

<output_format>
Return a JSON object matching this schema:
{{
  "raw_calculated_score": [Float],
  "penalty_triggered": [Boolean],
  "final_arbitrated_score": [Float 1.0 to 10.0],
  "arbiter_rationale": "[1 paragraph explaining how the math was settled, specifically noting if the fatal flaw penalty was triggered or if an agent was discounted due to low confidence]"
}}
</output_format>
"""
    
    response = await llm.ainvoke([
        SystemMessage(content="You are a strict, logical mathematical synthesizer. You only output structured data, extracted numbers, and the final weighted score formula. You do not make final buy/sell decisions."),
        HumanMessage(content=prompt)
    ])
    
    return {"synthesized_report": _get_text(response.content)}

async def decision_maker_node(state: GraphState):
    """Decision Maker Agent using Gemini"""
    ticker = state["ticker"]
    synth = state.get("synthesized_report", "")
    qual = state.get("qualitative_analysis", "")
    quant = state.get("quantitative_analysis", "")
    risk = state.get("risk_analysis", "")
    print(f"[{ticker}] Decision Maker Agent is running...")
    
    # Use Gemini model (Google GenAI) with automatic fallbacks, using gemini-3.5-pro for final advice
    llm = get_llm(temperature=0.3, model='gemini-3.5-pro')
    print(f"[{ticker}] Sleeping for {AGENT_SLEEP_TIME} seconds to respect API limits...")
    await asyncio.sleep(AGENT_SLEEP_TIME)
    
    prompt = f"""<role>
You are an expert AI Investment Advisor. Your job is to compile the research of your specialized desks regarding {ticker} into a friendly, conversational chat response for the user.
</role>

<context>
You will receive the entire graph state for {ticker}, which includes:
- Raw financial data & news.
- The Quant Agent's text, duration forecast, and technical price assessment.
- The Qual Agent's thesis.
- The Risk Agent's critique.
- The Arbiter's Final Score and Rationale.

Quant Agent Output:
{quant}

Qual Agent Output:
{qual}

Risk Agent Output:
{risk}

Synthesizer Output:
{synth}
</context>

<mandate>
Write the final response in a normal, conversational chat format. You must determine the final VERDICT for {ticker} based strictly on the Arbiter's Final Score:
- Score 7.5 to 10.0: INVEST (Strong Conviction)
- Score 5.5 to 7.4: HOLD (Wait for Catalyst)
- Score 1.0 to 5.4: PASS (Capital Destruction Risk)

Address the user directly. Explain the verdict, discuss the bull and bear cases briefly, and explicitly provide the Quant Agent's assessment on the current price and target entry price based on technical indicators. 
</mandate>

<rules>
- Do NOT output a rigid Markdown report structure (e.g. no # Executive Summary). Write naturally like a knowledgeable financial advisor chatting with a client.
- You must include the "Recommended Holding Duration" generated by the Quant agent.
- Explicitly state whether the user should buy at the current price or wait for the target entry price.
- If the Arbiter triggered the "Fatal Flaw Penalty", highlight this as the primary reason for a PASS verdict.
</rules>

<output_format>
Output a conversational message (you may use basic markdown like bolding or bullet points for readability, but keep it chatty).
</output_format>
"""
    
    response = await llm.ainvoke([
        SystemMessage(content="You are a Lead Investment Decision Maker. You write authoritative, structured markdown reports based on The Arbiter's synthesizer logic."),
        HumanMessage(content=prompt)
    ])
    
    return {"final_report": _get_text(response.content)}

# Build the Graph
workflow = StateGraph(GraphState)

workflow.add_node("qualitative", qualitative_agent_node)
workflow.add_node("quantitative", quantitative_agent_node)
workflow.add_node("risk", risk_agent_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("decision", decision_maker_node)

# We link them sequentially
workflow.set_entry_point("qualitative")
workflow.add_edge("qualitative", "quantitative")
workflow.add_edge("quantitative", "risk")
workflow.add_edge("risk", "synthesizer")
workflow.add_edge("synthesizer", "decision")
workflow.add_edge("decision", END)

from cache import get_cached_report, save_report

# Compile the graph
app = workflow.compile()

async def run_analysis(ticker: str):
    """Helper function to run the full graph for a given ticker, utilizing a 1-day SQLite cache."""
    print(f"Starting Multi-Agent Analysis for {ticker}...")
    
    
    cached_report = get_cached_report(ticker)
    if cached_report:
        print(f"\n[CACHE HIT] Found existing report generated today for {ticker}. Skipping agents...")
        print("\n--- FINAL REPORT (CACHED) ---\n")
        print(cached_report)
        return cached_report
        

    print(f"\n[CACHE MISS] No report generated today for {ticker}. Invoking Agents...\n")
    initial_state = {"ticker": ticker}
    

    final_state = await app.ainvoke(initial_state)
    report = final_state.get("final_report", "No report generated.")
    
    if report != "No report generated.":
        save_report(ticker, report)
        print(f"\n[CACHE STORE] Saved {ticker} report to database for today.")
    
    print("\n--- FINAL REPORT ---\n")
    print(report)
    return report

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    asyncio.run(run_analysis(ticker))
