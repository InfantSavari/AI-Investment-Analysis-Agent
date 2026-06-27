'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

export default function Home() {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!ticker) return;

    setLoading(true);
    setError(null);
    setReport(null);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticker: ticker.toUpperCase() }),
      });

      if (!res.ok) {
        throw new Error('Failed to fetch analysis. API might have timed out.');
      }

      const data = await res.json();
      setReport(data.report);
    } catch (err) {
      setError(err.message || 'An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <header className="header">
        <h1 className="title">Agentic AI Analyst</h1>
        <p className="subtitle">Institutional-Grade Multi-Agent Research Platform</p>
      </header>

      <form onSubmit={handleSearch} className="search-container">
        <input
          type="text"
          className="search-input"
          placeholder="Enter Stock Ticker (e.g. NVDA)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          disabled={loading}
          autoFocus
        />
        <button type="submit" className="search-button" disabled={loading || !ticker}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>

      {loading && (
        <div className="loader-container">
          <div className="spinner"></div>
          <div className="loading-text">Agents are collaborating. This takes ~1.5 minutes...</div>
        </div>
      )}

      {error && (
        <div className="error-message">
          {error}
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
