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

# Define the state that our graph will pass between agents
class GraphState(TypedDict):
    ticker: str
    qualitative_analysis: str
    quantitative_analysis: str
    risk_analysis: str
    synthesized_report: str
    final_report: str

async def qualitative_agent_node(state: GraphState):
    """Qualitative Analysis Agent using Gemini"""
    ticker = state["ticker"]
    print(f"[{ticker}] Qualitative Agent is running...")
    
    # Fetch news/sentiment data
    news_data = await fetch_alpha_vantage("NEWS_SENTIMENT", ticker)
    context = str(news_data)[:10000] # Trim to avoid context limits if extremely large
    
    # Use Gemini model (Google GenAI)
    llm = ChatGoogleGenerativeAI(model='gemini-3.5-flash', temperature=0.2)
    await asyncio.sleep(15)
    
    prompt = f"""<role>
You are the Lead Strategic Equities Analyst. You specialize in business models, competitive moats, and market narratives. You look beyond the spreadsheet to understand product-market fit, management execution, and brand power.
</role>

<context>
You will be provided with:
1. {ticker}'s business description and sector.
2. A summary of recent news headlines and the Market Sentiment Score for {ticker}.
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
    
    return {"qualitative_analysis": response.content}

async def quantitative_agent_node(state: GraphState):
    """Quantitative Analysis Agent using Gemini (Independent context)"""
    ticker = state["ticker"]
    print(f"[{ticker}] Quantitative Agent is running...")
    
    # Fetch financial statements concurrently
    results = await asyncio.gather(
        fetch_alpha_vantage("INCOME_STATEMENT", ticker),
        fetch_alpha_vantage("BALANCE_SHEET", ticker),
        fetch_alpha_vantage("CASH_FLOW", ticker)
    )
    income_data, balance_data, cashflow_data = results
    context = f"Income Statement:\n{str(income_data)[:5000]}\n\nBalance Sheet:\n{str(balance_data)[:5000]}\n\nCash Flow:\n{str(cashflow_data)[:5000]}"
    
    # Use a separate Gemini instance
    llm = ChatGoogleGenerativeAI(model='gemini-3.5-flash', temperature=0.2)
    await asyncio.sleep(15)
    
    prompt = quantitative_prompt = f"""<role>
You are the Lead Quantitative Analyst for an institutional hedge fund. You are entirely objective, emotionally detached, and rely strictly on empirical financial data. You do not care about the company's "story" or "vision"—you only care about the numbers.
</role>

<context>
You will be provided with:
1. A Markdown table of {ticker}'s latest financial statements (Revenue, Net Income, Debt, etc.).
2. The current Market Sentiment Score (1-10) derived from recent news for {ticker}.
</context>

<mandate>
Step 1: Calculate historical trends (YoY growth, margin expansion/compression).
Step 2: Perform relative valuation (analyze P/E, PEG ratio, FCF yield) to classify {ticker} as strictly one of the following: [DEEP VALUE, FAIRLY VALUED, OVERVALUED].
Step 3: Calculate the Recommended Holding Duration. 
- If OVERVALUED: 0 Months.
- If DEEP VALUE & Low Sentiment (< 4): 3 to 5 Years.
- If DEEP VALUE & High Sentiment (> 7): 6 to 12 Months.
- If FAIRLY VALUED & Neutral Sentiment: 1 to 3 Years.
- If FAIRLY VALUED & High Sentiment: Momentum Hold (Review in 3 Months).
</mandate>

<rules>
- Do not speculate on future products. Stick to historical/current financials.
- If financial data is missing or highly volatile, you MUST penalize your Confidence Score.
</rules>

<output_format>
Return a JSON object matching this schema:
{{
  "valuation_classification": "[Classification]",
  "holding_duration": "[Duration]",
  "analysis_text": "[2-3 paragraphs of rigorous financial analysis detailing margins, debt health, and valuation metrics]",
  "quantitative_score": [Float 1.0 to 10.0, where 10 is flawless financial health],
  "confidence_score": [Float 0.0 to 1.0, based on data completeness]
}}
</output_format>"""
    
    response = await llm.ainvoke([
        SystemMessage(content="You are a stock market quantitative analyst. Base your analysis solely on the provided numerical data."),
        HumanMessage(content=prompt)
    ])
    
    return {"quantitative_analysis": response.content}

async def risk_agent_node(state: GraphState):
    """Risk Analysis Agent using Groq"""
    ticker = state["ticker"]
    print(f"[{ticker}] Risk Agent is running...")
    
    # Risk agent might look at recent market news for volatility and headwinds
    news_data = await fetch_alpha_vantage("NEWS_SENTIMENT", ticker)
    context = str(news_data)[:8000]
    
    # Use Groq model
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    await asyncio.sleep(15)
    
    prompt = f"""<role>
You are the Chief Risk Officer (CRO) and an adversarial short-seller. Your explicit goal is to destroy the investment thesis for {ticker}. You assume every company is a value trap or a structural failure waiting to happen until proven otherwise.
</role>

<context>
You will receive:
1. Raw financial data and recent news headlines for {ticker}.
2. The optimistic analysis from the Qualitative Agent.
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
    
    return {"risk_analysis": response.content}

async def synthesizer_node(state: GraphState):
    """Synthesizer (The Arbiter) Agent using Gemini"""
    ticker = state["ticker"]
    qual = state.get("qualitative_analysis", "")
    quant = state.get("quantitative_analysis", "")
    risk = state.get("risk_analysis", "")
    print(f"[{ticker}] The Arbiter (Synthesizer) is running...")
    
    # Use Gemini model. Temperature 0.0 for strict logical workflow
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
    await asyncio.sleep(15)
    
    prompt = f"""<role>
You are the Portfolio Synthesizer (The Arbiter). You are a strict logic gate. You do not generate original financial analysis; your job is to ingest the scores from your specialized agents regarding {ticker}, execute a weighted mathematical formula, and resolve contradictions.
</role>

<context>
You will receive a JSON state containing the outputs of the Quant Agent, Qual Agent, and Risk Agent for {ticker}. Each output contains a Score (1-10) and a Confidence Rating (0.0-1.0).
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
    
    return {"synthesized_report": response.content}

async def decision_maker_node(state: GraphState):
    """Decision Maker Agent using Gemini"""
    ticker = state["ticker"]
    synth = state.get("synthesized_report", "")
    print(f"[{ticker}] Decision Maker Agent is running...")
    
    # Use Gemini model (Google GenAI)
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    await asyncio.sleep(15)
    
    prompt = f"""<role>
You are the Lead Portfolio Manager and Chief Investment Officer. Your job is to compile the research of your specialized desks regarding {ticker} into a beautifully formatted, institutional-grade Markdown research report for the firm's clients.
</role>

<context>
You will receive the entire graph state for {ticker}, which includes:
- Raw financial data & news.
- The Quant Agent's text and duration forecast.
- The Qual Agent's thesis.
- The Risk Agent's critique.
- The Arbiter's Final Score and Rationale.
</context>

<mandate>
Write the final Research Report. You must determine the final VERDICT for {ticker} based strictly on the Arbiter's Final Score:
- Score 7.5 to 10.0: INVEST (Strong Conviction)
- Score 5.5 to 7.4: HOLD (Wait for Catalyst)
- Score 1.0 to 5.4: PASS (Capital Destruction Risk)

Format the report using strict Markdown. Ensure it reads like a cohesive document, not a disjointed set of agent outputs.
</mandate>

<rules>
- The VERDICT must be displayed prominently at the top.
- You must include the "Recommended Holding Duration" generated by the Quant agent.
- If the Arbiter triggered the "Fatal Flaw Penalty", you must highlight this in the Executive Summary as the primary reason for a PASS verdict.
</rules>

<output_format>
Output ONLY valid Markdown. Use this exact structure:

# Investment Research Report: {ticker}
**Verdict:** [INVEST / HOLD / PASS]
**Arbiter Confidence Score:** [Final Score]/10
**Recommended Time Horizon:** [Duration from Quant Agent]

## Executive Summary
[Synthesize the Arbiter's rationale and the overarching narrative in 2 paragraphs]

## The Bull Case (Growth & Financial Health)
[Combine the best points from the Quant and Qual agents. Highlight the moat and financial trends]

## Risk Factors & Margin of Safety
[Present the Risk Agent's critique. Highlight the top threats and macro headwinds]

## Final Committee Justification
[Explain exactly why the committee chose to Invest, Hold, or Pass, referencing the contradiction resolution if applicable]
</output_format>
"""
    
    response = await llm.ainvoke([
        SystemMessage(content="You are a Lead Investment Decision Maker. You write authoritative, structured markdown reports based on The Arbiter's synthesizer logic."),
        HumanMessage(content=prompt)
    ])
    
    return {"final_report": response.content}

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
    
    # 1. Check if we already generated a report today
    cached_report = get_cached_report(ticker)
    if cached_report:
        print(f"\n[CACHE HIT] Found existing report generated today for {ticker}. Skipping agents...")
        print("\n--- FINAL REPORT (CACHED) ---\n")
        print(cached_report)
        return cached_report
        
    # 2. If no cache, run the LangGraph agents
    print(f"\n[CACHE MISS] No report generated today for {ticker}. Invoking Agents...\n")
    initial_state = {"ticker": ticker}
    
    # Invoke the graph asynchronously
    final_state = await app.ainvoke(initial_state)
    report = final_state.get("final_report", "No report generated.")
    
    # 3. Save the newly generated report to the database
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
