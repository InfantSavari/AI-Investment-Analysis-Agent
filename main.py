import os
import asyncio
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API")

# Initialize FastMCP server
mcp = FastMCP("AlphaVantage Server")

BASE_URL = "https://www.alphavantage.co/query"

async def fetch_alpha_vantage(function: str, ticker: str, **kwargs) -> dict:
    """Helper function to fetch data from Alpha Vantage API asynchronously."""
    if not ALPHA_VANTAGE_API_KEY:
        return {"Error": "API Key not found in environment variables. Please set ALPHAVANTAGE_API_KEY in .env"}
    
    params = {
        "function": function,
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
        **kwargs
    }
    
    # Special case for NEWS_SENTIMENT which uses 'tickers' instead of 'symbol'
    if function == "NEWS_SENTIMENT":
        del params["symbol"]
        params["tickers"] = ticker

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(BASE_URL, params=params, timeout=15.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"Error": str(e)}

@mcp.tool()
async def get_financial_statements(ticker: str) -> str:
    """
    Retrieve the Income Statement, Balance Sheet, and Cash Flow for a given stock ticker.
    Returns the financial statements summary in markdown format.
    """
    # Fetch all three statements concurrently
    results = await asyncio.gather(
        fetch_alpha_vantage("INCOME_STATEMENT", ticker),
        fetch_alpha_vantage("BALANCE_SHEET", ticker),
        fetch_alpha_vantage("CASH_FLOW", ticker)
    )
    
    income_data, balance_data, cashflow_data = results
    
    md_output = f"# Financial Statements for {ticker.upper()}\n\n"
    
    def format_report(title: str, data: dict, report_key: str) -> str:
        res = f"## {title}\n"
        if "Error" in data or "Information" in data:
            res += f"**API Message**: {data.get('Error', data.get('Information'))}\n\n"
            return res
            
        reports = data.get(report_key, [])
        if not reports:
            res += "No data available.\n\n"
            return res
            
        latest = reports[0]
        fiscal_date = latest.get("fiscalDateEnding", "Unknown")
        currency = latest.get("reportedCurrency", "USD")
        res += f"**Fiscal Date Ending:** {fiscal_date} | **Currency:** {currency}\n\n"
        
        # Add key metrics based on report type
        if "totalRevenue" in latest: # Income Statement
            res += f"- **Total Revenue**: {latest.get('totalRevenue')}\n"
            res += f"- **Gross Profit**: {latest.get('grossProfit')}\n"
            res += f"- **Operating Income**: {latest.get('operatingIncome')}\n"
            res += f"- **Net Income**: {latest.get('netIncome')}\n"
        elif "totalAssets" in latest: # Balance Sheet
            res += f"- **Total Assets**: {latest.get('totalAssets')}\n"
            res += f"- **Total Liabilities**: {latest.get('totalLiabilities')}\n"
            res += f"- **Total Shareholder Equity**: {latest.get('totalShareholderEquity')}\n"
            res += f"- **Retained Earnings**: {latest.get('retainedEarnings')}\n"
        elif "operatingCashflow" in latest: # Cash Flow
            res += f"- **Operating Cashflow**: {latest.get('operatingCashflow')}\n"
            res += f"- **Capital Expenditures**: {latest.get('capitalExpenditures')}\n"
            res += f"- **Net Income**: {latest.get('netIncome')}\n"
            res += f"- **Dividend Payout**: {latest.get('dividendPayout')}\n"
                
        res += "\n"
        return res

    md_output += format_report("Income Statement (Latest Annual)", income_data, "annualReports")
    md_output += format_report("Balance Sheet (Latest Annual)", balance_data, "annualReports")
    md_output += format_report("Cash Flow (Latest Annual)", cashflow_data, "annualReports")
    
    return md_output

@mcp.tool()
async def get_latest_news_and_sentiments(ticker: str) -> str:
    """
    Retrieve the latest news articles and market sentiment for a given stock ticker.
    Returns the news and sentiment analysis in markdown format.
    """
    data = await fetch_alpha_vantage("NEWS_SENTIMENT", ticker)
    
    md_output = f"# News & Market Sentiment for {ticker.upper()}\n\n"
    
    if "Error" in data or "Information" in data:
        md_output += f"**API Message**: {data.get('Error', data.get('Information'))}\n"
        return md_output
        
    articles = data.get("feed", [])
    if not articles:
        md_output += "No recent news found.\n"
        return md_output
        
    md_output += f"Found {len(articles)} recent articles. Showing top 5:\n\n"
    
    for idx, article in enumerate(articles[:5]):
        title = article.get("title", "No Title")
        url = article.get("url", "#")
        summary = article.get("summary", "No summary available.")
        overall_sentiment_score = article.get("overall_sentiment_score", "N/A")
        overall_sentiment_label = article.get("overall_sentiment_label", "N/A")
        
        md_output += f"### {idx+1}. [{title}]({url})\n"
        md_output += f"- **Summary**: {summary}\n"
        md_output += f"- **Overall Sentiment**: {overall_sentiment_label} (Score: {overall_sentiment_score})\n"
        
        # Find ticker specific sentiment
        ticker_sentiment = next((t for t in article.get("ticker_sentiment", []) if t.get("ticker") == ticker.upper()), None)
        if ticker_sentiment:
            md_output += f"- **{ticker.upper()} Specific Sentiment**: {ticker_sentiment.get('ticker_sentiment_label')} (Score: {ticker_sentiment.get('ticker_sentiment_score')})\n"
            
        md_output += "\n"
        
    return md_output

if __name__ == "__main__":
    # Start the FastMCP server
    mcp.run()
