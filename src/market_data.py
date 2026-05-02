"""
Market data wrapper — fetches live prices and metadata via yfinance.

WHY yfinance:
- Free, no API key required (assignment says "self-host everything")
- Covers all exchanges in fixtures: NASDAQ, NYSE, LSE (.L), Euronext (.AS), TSE (.T)
- Returns sector, industry, dividend data — all needed for portfolio health

CACHING:
- Simple TTL cache to avoid hammering Yahoo Finance during a single request
- Cache keyed by ticker, expires after 5 minutes
- In production, this would be Redis-backed with proper invalidation
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# TTL cache: {ticker: (data, timestamp)}
_cache: dict[str, tuple["TickerData", float]] = {}
_CACHE_TTL = 300  # 5 minutes


@dataclass
class TickerData:
    """Standardised market data for a single ticker."""
    ticker: str
    current_price: float = 0.0
    currency: str = "USD"
    name: str = ""
    sector: str = "Unknown"
    industry: str = "Unknown"
    market_cap: float = 0.0
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    fifty_two_week_high: float = 0.0
    fifty_two_week_low: float = 0.0
    error: str = ""


def get_ticker_data(ticker: str) -> TickerData:
    """
    Fetch current market data for a ticker. Returns TickerData even on failure
    (with error field set) — callers never crash from bad market data.
    """
    # Check cache
    if ticker in _cache:
        data, ts = _cache[ticker]
        if time.time() - ts < _CACHE_TTL:
            return data

    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}

        data = TickerData(
            ticker=ticker,
            current_price=info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", 0.0),
            currency=info.get("currency", "USD"),
            name=info.get("shortName") or info.get("longName", ticker),
            sector=info.get("sector", "Unknown"),
            industry=info.get("industry", "Unknown"),
            market_cap=info.get("marketCap", 0.0),
            pe_ratio=info.get("trailingPE"),
            dividend_yield=info.get("dividendYield"),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh", 0.0),
            fifty_two_week_low=info.get("fiftyTwoWeekLow", 0.0),
        )
        _cache[ticker] = (data, time.time())
        return data

    except Exception as e:
        logger.warning(f"Failed to fetch market data for {ticker}: {e}")
        return TickerData(ticker=ticker, error=str(e))


def get_benchmark_return(benchmark_ticker: str, start_date: str) -> Optional[float]:
    """
    Calculate the total return of a benchmark from start_date to now.
    Returns percentage (e.g. 14.2 for 14.2%) or None on failure.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(benchmark_ticker)
        hist = t.history(start=start_date)
        if hist.empty or len(hist) < 2:
            return None
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        return ((end_price - start_price) / start_price) * 100
    except Exception as e:
        logger.warning(f"Failed to fetch benchmark return for {benchmark_ticker}: {e}")
        return None


# Maps user-facing benchmark names to Yahoo Finance tickers
BENCHMARK_TICKERS: dict[str, str] = {
    "S&P 500": "^GSPC",
    "QQQ": "QQQ",
    "FTSE 100": "^FTSE",
    "NIKKEI 225": "^N225",
    "MSCI World": "URTH",  # iShares MSCI World ETF as proxy
}


def clear_cache() -> None:
    """Clear the market data cache (useful in tests)."""
    _cache.clear()
