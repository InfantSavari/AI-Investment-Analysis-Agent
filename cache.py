import sqlite3
import datetime

DB_FILE = "reports_cache.db"

def init_db():
    """Initializes the SQLite database and creates the caching table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Table stores the ticker as primary key, meaning there's only ever one report per ticker stored.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            ticker TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            report_content TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def get_today_date_string() -> str:
    """Returns today's date in YYYY-MM-DD format."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

def get_cached_report(ticker: str) -> str | None:
    """
    Checks if there's a cached report for the given ticker generated today.
    Returns the markdown report if found, otherwise None.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    today = get_today_date_string()
    
    cursor.execute('''
        SELECT report_content FROM reports 
        WHERE ticker = ? AND date = ?
    ''', (ticker.upper(), today))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row[0]
    return None

def save_report(ticker: str, report_content: str):
    """
    Saves or overwrites the report for the given ticker with today's date.
    This effectively invalidates any older report for this ticker.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    today = get_today_date_string()
    
    cursor.execute('''
        INSERT OR REPLACE INTO reports (ticker, date, report_content)
        VALUES (?, ?, ?)
    ''', (ticker.upper(), today, report_content))
    
    conn.commit()
    conn.close()

# Initialize the DB as soon as this module is imported
init_db()
