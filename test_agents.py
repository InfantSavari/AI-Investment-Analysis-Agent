import asyncio
from graph import app

async def test_agents():
    # Use a well-known ticker to test the agents
    ticker = "NVDA"
    print(f"--- Testing Multi-Agent Architecture for {ticker} ---")
    
    initial_state = {"ticker": ticker}
    
    # Run the compiled LangGraph app
    try:
        final_state = await app.ainvoke(initial_state)
        
        print("\n--- QUALITATIVE ANALYSIS ---")
        print(final_state.get("qualitative_analysis", "No output generated."))
        
        print("\n--- QUANTITATIVE ANALYSIS ---")
        print(final_state.get("quantitative_analysis", "No output generated."))
        
        print("\n--- RISK ANALYSIS ---")
        print(final_state.get("risk_analysis", "No output generated."))
        
        print("\n--- FINAL DECISION REPORT ---")
        print(final_state.get("final_report", "No report generated."))
        
        print("\n--- TEST COMPLETED SUCCESSFULLY ---")
    except Exception as e:
        print(f"\n--- ERROR DURING EXECUTION ---")
        print(str(e))

if __name__ == "__main__":
    asyncio.run(test_agents())
