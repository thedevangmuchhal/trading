from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_engine import generate_signals
from data_fetcher import fetch_market_data, get_angel_session
import json
import uuid
from datetime import datetime
from typing import Optional

app = FastAPI(title="AI Trading API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# PAPER TRADING — In-memory store (resets on server restart)
# ─────────────────────────────────────────────────────────────────────────────
paper_trades: list[dict] = []

class PaperTradeRequest(BaseModel):
    ticker: str = "^NSEI"
    action: str  # "BUY"
    option_type: str  # "CE" or "PE"
    strike: float
    entry_price: float
    lot_size: int = 25
    stop_loss: float = 0
    target: float = 0

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Trading Backend is Live. Please access the frontend dashboard."}

@app.get("/api/signal")
def get_signal(ticker: str = "^NSEI"):
    """
    Returns the latest AI-generated trading signal and market data.
    """
    data = generate_signals(ticker)
    return data

@app.get("/api/candles")
def get_candles(ticker: str = "^NSEI", interval: str = "15m"):
    """
    Returns raw OHLCV candlestick data for frontend charting.
    Accepts interval parameter: 5m, 15m, 30m, 1h
    """
    # Validate interval
    valid_intervals = {"5m": "2d", "15m": "5d", "30m": "30d", "1h": "60d"}
    if interval not in valid_intervals:
        interval = "15m"
    period = valid_intervals[interval]

    df = fetch_market_data(ticker, interval=interval, period=period)
    if df.empty:
        return {"candles": [], "interval": interval}

    candles = []
    for idx, row in df.iterrows():
        candles.append({
            "time": int(idx.timestamp()),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row.get("Volume", 0))
        })
    return {"candles": candles, "interval": interval}

@app.get("/api/oi")
def get_oi(ticker: str = "^NSEI"):
    """
    Returns Options Open Interest data for multiple strikes around ATM.
    Uses Angel One SmartAPI.
    """
    from datetime import datetime
    import requests as req

    # Determine base symbol
    base_symbol = "NIFTY"
    if ticker == "^NSEI": base_symbol = "NIFTY"
    elif ticker == "^BSESN": base_symbol = "SENSEX"
    elif ticker == "^NSEBANK": base_symbol = "BANKNIFTY"
    elif ticker.endswith(".NS"): base_symbol = ticker.replace(".NS", "")
    else:
        return {"oi_data": [], "pcr": None}

    angel = get_angel_session()
    if not angel:
        return {"oi_data": [], "pcr": None, "error": "Angel One not connected. Check ENV variables."}

    # Get current price from yfinance
    df = fetch_market_data(ticker, interval="15m", period="5d")
    if df.empty:
        return {"oi_data": [], "pcr": None}
    current_price = float(df["Close"].iloc[-1])

    from data_fetcher import fetch_advanced_oi

    try:
        oi_metrics = fetch_advanced_oi(ticker, current_price)
        if oi_metrics:
            return oi_metrics
        else:
            return {"oi_data": [], "pcr": None, "error": "Failed to fetch OI data from Angel One"}
    except Exception as e:
        return {"oi_data": [], "pcr": None, "error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# PAPER TRADING ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/paper-trade")
def create_paper_trade(trade: PaperTradeRequest):
    """Execute a paper trade with real option prices."""
    trade_obj = {
        "id": str(uuid.uuid4())[:8],
        "ticker": trade.ticker,
        "action": trade.action,
        "option_type": trade.option_type,
        "strike": trade.strike,
        "entry_price": trade.entry_price,
        "current_price": trade.entry_price,
        "lot_size": trade.lot_size,
        "stop_loss": trade.stop_loss,
        "target": trade.target,
        "status": "OPEN",
        "pnl": 0.0,
        "opened_at": datetime.now().isoformat(),
        "closed_at": None,
    }
    paper_trades.append(trade_obj)
    return trade_obj

@app.get("/api/paper-trades")
def get_paper_trades():
    """Return all paper trades (open + closed)."""
    return {"trades": paper_trades}

@app.post("/api/paper-trade/{trade_id}/close")
def close_paper_trade(trade_id: str, close_price: float = 0):
    """Close an open paper trade at the given price."""
    for trade in paper_trades:
        if trade["id"] == trade_id and trade["status"] == "OPEN":
            trade["status"] = "CLOSED"
            trade["closed_at"] = datetime.now().isoformat()
            if close_price > 0:
                trade["current_price"] = close_price
            # P&L = (current - entry) * lot_size for BUY
            diff = trade["current_price"] - trade["entry_price"]
            trade["pnl"] = round(diff * trade["lot_size"], 2)
            return trade
    raise HTTPException(status_code=404, detail="Trade not found or already closed")

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/backtest")
def get_backtest(ticker: str = "^NSEI"):
    """Returns 1-year backtest results for the signal strategy (cached 6h)."""
    from backtester import run_backtest
    result = run_backtest(ticker)
    return result

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
