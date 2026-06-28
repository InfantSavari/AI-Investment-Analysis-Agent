import os
import asyncio
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from main import fetch_alpha_vantage
from graph import _get_text

load_dotenv()

class DebateGraphState(TypedDict):
    ticker: str
    round_count: int
    max_rounds: int
    data_context: str
    debate_history: List[str]
    final_report: str

async def data_gathering_node(state: DebateGraphState):
    """Fetches all necessary data once to avoid repeated API calls."""
    ticker = state["ticker"]
    print(f"[{ticker}] Gathering data for debate...")
    
    results = await asyncio.gather(
        fetch_alpha_vantage("NEWS_SENTIMENT", ticker),
        fetch_alpha_vantage("INCOME_STATEMENT", ticker),
        fetch_alpha_vantage("BALANCE_SHEET", ticker)
    )
    news_data, income_data, balance_data = results
    
    context = f"NEWS:\n{str(news_data)[:5000]}\n\nFINANCIALS:\nIncome: {str(income_data)[:3000]}\nBalance: {str(balance_data)[:3000]}"
    return {"data_context": context, "round_count": 0, "debate_history": []}

async def bull_case_node(state: DebateGraphState):
    ticker = state["ticker"]
    round_count = state.get("round_count", 0)
    history = "\n\n".join(state.get("debate_history", []))
    context = state["data_context"]
    
    print(f"[{ticker}] Bull Agent (Round {round_count + 1})...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    await asyncio.sleep(5) # Delay to respect rate limits
    
    prompt = f"""<role>You are the Bull Case Analyst for {ticker}. You believe strongly in the upside potential.</role>
    <context>{context}</context>
    <history>Here is the debate so far:\n{history}</history>
    <task>
    If this is the first round, present your initial Bull thesis (growth, moat, valuation).
    If there is debate history, explicitly respond to the Bear Analyst's attacks and defend your thesis. 
    Keep it concise (2-3 paragraphs).
    </task>"""
    
    resp = await llm.ainvoke([
        SystemMessage(content="You are a Bull Analyst. Focus on upside, growth, and structural advantages."),
        HumanMessage(content=prompt)
    ])
    content = _get_text(resp.content)
    
    # Append new response to the history list
    new_history = state.get("debate_history", []) + [f"**Bull Analyst (Round {round_count + 1}):**\n{content}"]
    return {"debate_history": new_history}

async def bear_case_node(state: DebateGraphState):
    ticker = state["ticker"]
    round_count = state.get("round_count", 0)
    history = "\n\n".join(state.get("debate_history", []))
    context = state["data_context"]
    
    print(f"[{ticker}] Bear Agent (Round {round_count + 1})...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    await asyncio.sleep(5)
    
    prompt = f"""<role>You are the Bear Case Analyst (Short Seller) for {ticker}. You look for fatal flaws and downside risks.</role>
    <context>{context}</context>
    <history>Here is the debate so far:\n{history}</history>
    <task>
    Read the Bull Analyst's latest argument. Attack their weak points, highlight risks they ignored, and present your Bear thesis.
    Keep it concise (2-3 paragraphs).
    </task>"""
    
    resp = await llm.ainvoke([
        SystemMessage(content="You are a Bear Analyst. Focus on downside, structural risks, and overvaluation."),
        HumanMessage(content=prompt)
    ])
    content = _get_text(resp.content)
    
    new_history = state.get("debate_history", []) + [f"**Bear Analyst (Round {round_count + 1}):**\n{content}"]
    
    # Increment round count here, since a full round is Bull -> Bear
    return {"debate_history": new_history, "round_count": round_count + 1}

def should_continue_debate(state: DebateGraphState):
    if state["round_count"] >= state["max_rounds"]:
        return "end"
    return "continue"

async def decision_maker_node(state: DebateGraphState):
    ticker = state["ticker"]
    history = "\n\n".join(state.get("debate_history", []))
    
    print(f"[{ticker}] CIO making final decision...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
    await asyncio.sleep(5)
    
    prompt = f"""<role>You are the Chief Investment Officer.</role>
    <task>
    Review the following debate between your Bull and Bear analysts regarding {ticker}.
    Write a final markdown research report.
    1. Summarize the best arguments from both sides.
    2. Make a final verdict: INVEST, HOLD, or PASS.
    3. Justify your verdict based on which analyst made the stronger case.
    </task>
    
    <debate>
    {history}
    </debate>"""
    
    resp = await llm.ainvoke([
        SystemMessage(content="You are the CIO. Output only valid markdown with your final verdict prominently at the top."),
        HumanMessage(content=prompt)
    ])
    content = _get_text(resp.content)
    
    return {"final_report": content}

# Build the Graph
workflow = StateGraph(DebateGraphState)

workflow.add_node("data", data_gathering_node)
workflow.add_node("bull", bull_case_node)
workflow.add_node("bear", bear_case_node)
workflow.add_node("decision", decision_maker_node)

# Set the flow
workflow.set_entry_point("data")
workflow.add_edge("data", "bull")
workflow.add_edge("bull", "bear")

# Conditional loop
workflow.add_conditional_edges(
    "bear",
    should_continue_debate,
    {
        "continue": "bull",
        "end": "decision"
    }
)
workflow.add_edge("decision", END)

# Compile
debate_app = workflow.compile()

async def run_debate(ticker: str, max_rounds: int = 2):
    print(f"Starting Multi-Round Debate for {ticker} (Max Rounds: {max_rounds})...")
    initial_state = {
        "ticker": ticker, 
        "max_rounds": max_rounds, 
        "debate_history": [], 
        "round_count": 0
    }
    
    final_state = await debate_app.ainvoke(initial_state)
    report = final_state.get("final_report", "No report generated.")
    
    print("\n--- FINAL DEBATE REPORT ---\n")
    print(report)
    return report

if __name__ == "__main__":
    # Example test run
    asyncio.run(run_debate("TSLA", max_rounds=2))
