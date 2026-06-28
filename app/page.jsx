'use client';

import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

export default function Home() {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);

  // Simulated progress bar for the ~90 second wait time
  useEffect(() => {
    let interval;
    if (loading) {
      setProgress(0);
      interval = setInterval(() => {
        setProgress((prev) => {
          // Slow down progress as it gets closer to 95%
          if (prev >= 95) return prev;
          const increment = prev < 50 ? 2 : prev < 80 ? 1 : 0.5;
          return Math.min(prev + increment, 95);
        });
      }, 1000); // Update every second
    } else {
      setProgress(100);
      setTimeout(() => setProgress(0), 1000);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const handleSearch = async (e, directTicker = null) => {
    if (e) e.preventDefault();
    const targetTicker = directTicker || ticker;
    if (!targetTicker) return;

    setTicker(targetTicker);
    setLoading(true);
    setError(null);
    setReport(null);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticker: targetTicker.toUpperCase() }),
      });

      if (!res.ok) {
        throw new Error('Failed to fetch analysis. The AI agents might have encountered an error or timed out.');
      }

      const data = await res.json();
      setReport(data.report);
    } catch (err) {
      setError(err.message || 'An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  const popularTickers = ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'AMD'];

  return (
    <main className="container">
      <header className="header">
        <h1 className="title">Agentic AI Analyst</h1>
        <p className="subtitle">Institutional-Grade Multi-Agent Research Platform</p>
      </header>

      <form onSubmit={(e) => handleSearch(e)} className="search-container">
        <input
          type="text"
          className="search-input"
          placeholder="Enter Stock Ticker (e.g. AMZN)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          disabled={loading}
          autoFocus
        />
        <button type="submit" className="search-button" disabled={loading || !ticker}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>

      {!loading && !report && (
        <div className="popular-tickers">
          <p>Or try a popular stock:</p>
          <div className="ticker-badges">
            {popularTickers.map((t) => (
              <button 
                key={t} 
                className="ticker-badge"
                onClick={() => handleSearch(null, t)}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <div className="loader-container">
          <div className="spinner"></div>
          <div className="loading-text">Agents are collaborating. This takes ~1.5 minutes...</div>
          
          <div className="progress-track">
            <div 
              className="progress-fill" 
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <p className="progress-status">
            {progress < 20 && "Gathering financial data..."}
            {progress >= 20 && progress < 45 && "Quant Agent calculating valuations..."}
            {progress >= 45 && progress < 70 && "Risk Agent scanning for red flags..."}
            {progress >= 70 && progress < 90 && "Qual Agent analyzing market narrative..."}
            {progress >= 90 && "Synthesizing final committee report..."}
          </p>
        </div>
      )}

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {report && (
        <article className="report-panel">
          <div className="markdown">
            <ReactMarkdown>{report}</ReactMarkdown>
          </div>
        </article>
      )}
    </main>
  );
}
