from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_engine import generate_signals
from data_fetcher import fetch_market_data, get_angel_session
import json

app = FastAPI(title="AI Trading API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/signal")
def get_signal(ticker: str = "^NSEI"):
    """
    Returns the latest AI-generated trading signal and market data.
    """
    data = generate_signals(ticker)
    return data

@app.get("/api/candles")
def get_candles(ticker: str = "^NSEI"):
    """
    Returns raw OHLCV candlestick data for frontend charting.
    """
    df = fetch_market_data(ticker)
    if df.empty:
        return {"candles": []}
    
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
    return {"candles": candles}

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
    df = fetch_market_data(ticker, interval="15m", period="1d")
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

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
