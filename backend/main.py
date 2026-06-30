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
# SIGNAL HISTORY LOG (#6) — Track signal accuracy over time
# ─────────────────────────────────────────────────────────────────────────────
signal_history: list[dict] = []
MAX_SIGNAL_HISTORY = 50

def _evaluate_past_signals(current_price: float):
    """Evaluate if past signals were correct based on price movement."""
    for sig in signal_history:
        if sig.get("was_correct") is not None:
            continue  # Already evaluated
        if sig.get("action") == "WAIT":
            # WAIT is correct if price moved < 0.3% (stayed flat)
            if sig.get("price_at_signal") and current_price:
                pct_move = abs(current_price - sig["price_at_signal"]) / sig["price_at_signal"] * 100
                sig["price_after"] = current_price
                sig["was_correct"] = pct_move < 0.3
        elif sig.get("action") == "BUY":
            sig["price_after"] = current_price
            sig["was_correct"] = current_price > sig.get("price_at_signal", 0)
        elif sig.get("action") == "SELL":
            sig["price_after"] = current_price
            sig["was_correct"] = current_price < sig.get("price_at_signal", 0)

# ─────────────────────────────────────────────────────────────────────────────
# RENDER FREE TIER KEEPALIVE — self-ping during market hours
# ─────────────────────────────────────────────────────────────────────────────
import threading, time, os, requests as _requests

def _keepalive_loop():
    """Ping own server every 4 minutes during market hours to prevent Render sleep."""
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "")
    if not render_url:
        print("[Keepalive] No RENDER_EXTERNAL_URL set. Self-ping disabled.")
        return

    health_url = f"{render_url}/health"
    print(f"[Keepalive] 🏓 Started. Will ping {health_url} during market hours.")

    while True:
        try:
            from datetime import timedelta
            utc_now = datetime.utcnow()
            ist_now = utc_now + timedelta(hours=5, minutes=30)
            h, m, day = ist_now.hour, ist_now.minute, ist_now.isoweekday()
            total_mins = h * 60 + m

            # Ping during 9:00 - 15:45 IST on weekdays
            if day <= 5 and 540 <= total_mins <= 945:
                _requests.get(health_url, timeout=10)
        except Exception:
            pass
        time.sleep(240)  # Every 4 minutes

# Start keepalive on app boot
_keepalive_thread = threading.Thread(target=_keepalive_loop, daemon=True)
_keepalive_thread.start()

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "alive", "time": datetime.now().isoformat()}

@app.get("/")
def root():
    return {"status": "Trading Backend is Live. Please access the frontend dashboard."}

@app.get("/api/signal")
def get_signal(ticker: str = "^NSEI"):
    """
    Returns the latest AI-generated trading signal and market data.
    Also tracks signal history for accuracy measurement.
    """
    data = generate_signals(ticker)

    # Evaluate past signals with current price
    current_price = data.get("current_price", 0)
    if current_price > 0:
        _evaluate_past_signals(current_price)

    # Record this signal
    entry = {
        "timestamp": data.get("timestamp", ""),
        "action": data.get("action", "WAIT"),
        "confidence": data.get("confidence_score", 50),
        "price_at_signal": current_price,
        "price_after": None,
        "was_correct": None,
        "signal_strength": data.get("signal_strength", 0),
    }
    signal_history.append(entry)
    if len(signal_history) > MAX_SIGNAL_HISTORY:
        signal_history.pop(0)

    # Compute accuracy stats
    evaluated = [s for s in signal_history if s.get("was_correct") is not None]
    correct = sum(1 for s in evaluated if s["was_correct"])
    total = len(evaluated)
    accuracy = round(correct / total * 100, 1) if total > 0 else 0

    data["signal_history_accuracy"] = {
        "correct": correct,
        "total": total,
        "accuracy_pct": accuracy,
        "label": f"{correct}/{total} correct ({accuracy}%)" if total > 0 else "Collecting data…",
    }

    return data

@app.get("/api/signal-history")
def get_signal_history():
    """Return the signal history log with accuracy stats."""
    evaluated = [s for s in signal_history if s.get("was_correct") is not None]
    correct = sum(1 for s in evaluated if s["was_correct"])
    total = len(evaluated)
    accuracy = round(correct / total * 100, 1) if total > 0 else 0
    return {
        "history": signal_history[-20:],
        "accuracy": {"correct": correct, "total": total, "accuracy_pct": accuracy},
    }

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
# PAPER TRADING with Trailing SL + Capital Preservation (#4, #3)
# ─────────────────────────────────────────────────────────────────────────────
MAX_DAILY_LOSS = 5000  # ₹ configurable
MAX_OPEN_POSITIONS = 2

def _get_daily_pnl():
    """Sum P&L of trades closed today."""
    today = datetime.now().date().isoformat()
    return sum(
        t["pnl"] for t in paper_trades
        if t["status"] == "CLOSED" and t.get("closed_at", "").startswith(today)
    )

@app.post("/api/paper-trade")
def create_paper_trade(trade: PaperTradeRequest):
    """Execute a paper trade with real option prices + capital preservation."""
    # Gate: Max daily loss
    if _get_daily_pnl() <= -MAX_DAILY_LOSS:
        raise HTTPException(status_code=403, detail=f"Daily loss limit (₹{MAX_DAILY_LOSS}) reached. No new trades today.")

    # Gate: Max open positions
    open_count = sum(1 for t in paper_trades if t["status"] == "OPEN")
    if open_count >= MAX_OPEN_POSITIONS:
        raise HTTPException(status_code=403, detail=f"Max {MAX_OPEN_POSITIONS} open positions allowed.")

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
        "initial_sl": trade.stop_loss,       # Original SL (never moves up)
        "trailing_sl": trade.stop_loss,       # Trailing SL (moves up)
        "target": trade.target,
        "peak_price": trade.entry_price,      # Highest price seen
        "status": "OPEN",
        "pnl": 0.0,
        "opened_at": datetime.now().isoformat(),
        "closed_at": None,
    }
    paper_trades.append(trade_obj)
    return trade_obj

@app.get("/api/paper-trades")
def get_paper_trades():
    """Return all paper trades (open + closed) + daily P&L."""
    return {
        "trades": paper_trades,
        "daily_pnl": round(_get_daily_pnl(), 2),
        "daily_loss_limit": MAX_DAILY_LOSS,
        "daily_limit_hit": _get_daily_pnl() <= -MAX_DAILY_LOSS,
    }

@app.post("/api/paper-trade/{trade_id}/update")
def update_paper_trade(trade_id: str, current_price: float = 0):
    """Update current price and apply trailing SL logic."""
    for trade in paper_trades:
        if trade["id"] == trade_id and trade["status"] == "OPEN":
            if current_price <= 0:
                return trade

            trade["current_price"] = current_price
            entry = trade["entry_price"]
            diff = current_price - entry
            trade["pnl"] = round(diff * trade["lot_size"], 2)

            # Track peak price
            if current_price > trade["peak_price"]:
                trade["peak_price"] = current_price

            # ── Trailing SL Logic ─────────────────────────────────
            # Estimate ATR as 1% of entry for paper trading
            atr_estimate = entry * 0.01

            profit = current_price - entry
            if profit > 2 * atr_estimate:
                # Trail SL at entry + 1×ATR
                new_sl = entry + atr_estimate
                trade["trailing_sl"] = max(trade["trailing_sl"], new_sl)
            elif profit > 1 * atr_estimate:
                # Move SL to breakeven
                trade["trailing_sl"] = max(trade["trailing_sl"], entry)

            # ── Auto-close if trailing SL hit ─────────────────────
            if current_price <= trade["trailing_sl"]:
                trade["status"] = "CLOSED"
                trade["closed_at"] = datetime.now().isoformat()
                trade["current_price"] = trade["trailing_sl"]
                diff = trade["trailing_sl"] - entry
                trade["pnl"] = round(diff * trade["lot_size"], 2)

            return trade
    raise HTTPException(status_code=404, detail="Trade not found or already closed")

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

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL LOGGER ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class LoggerConfig(BaseModel):
    sheet_url: str
    ticker: str = "^NSEI"
    lot_size: int = 25

@app.post("/api/logger/start")
def start_signal_logger(config: LoggerConfig):
    """Start the background signal logger. Requires Google Apps Script web app URL."""
    from signal_logger import start_logger
    result = start_logger(config.sheet_url, config.ticker, config.lot_size)
    return result

@app.post("/api/logger/stop")
def stop_signal_logger():
    """Stop the background signal logger."""
    from signal_logger import stop_logger
    return stop_logger()

@app.get("/api/logger/status")
def get_logger_status():
    """Get current logger status (running, position, daily P&L)."""
    from signal_logger import get_logger_status
    return get_logger_status()

@app.get("/api/logger/log")
def get_logger_log():
    """Get today's full signal log."""
    from signal_logger import get_today_log
    return {"log": get_today_log()}

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

