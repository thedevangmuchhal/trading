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
    
    # Get options from the memory-efficient streaming parser
    from data_fetcher import get_filtered_angel_options
    opts = get_filtered_angel_options(base_symbol)
    
    if not opts:
        return {"oi_data": [], "pcr": None}
    
    def parse_expiry(d):
        try: return datetime.strptime(d, "%d%b%Y")
        except: return datetime.max
    
    now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    valid = [x for x in opts if x.get("expiry") and parse_expiry(x["expiry"]) >= now]
    if not valid:
        return {"oi_data": [], "pcr": None}
    valid.sort(key=lambda x: parse_expiry(x["expiry"]))
    nearest_expiry = valid[0]["expiry"]
    expiry_opts = [x for x in valid if x["expiry"] == nearest_expiry]
    
    def get_strike(o):
        try: return float(o["strike"]) / 100
        except: return 0
    
    # Get 5 strikes above and below ATM (11 strikes total)
    atm = round(current_price / 50) * 50  # Round to nearest 50
    strikes_to_fetch = [atm + (i * 50) for i in range(-5, 6)]
    
    tokens_to_fetch = []
    strike_map = {}  # token -> {strike, type}
    
    for strike in strikes_to_fetch:
        ce = next((x for x in expiry_opts if get_strike(x) == strike and x["symbol"].endswith("CE")), None)
        pe = next((x for x in expiry_opts if get_strike(x) == strike and x["symbol"].endswith("PE")), None)
        if ce:
            tokens_to_fetch.append(ce["token"])
            strike_map[ce["token"]] = {"strike": strike, "type": "CE"}
        if pe:
            tokens_to_fetch.append(pe["token"])
            strike_map[pe["token"]] = {"strike": strike, "type": "PE"}
    
    if not tokens_to_fetch:
        return {"oi_data": [], "pcr": None}
    
    try:
        payload = {"NFO": tokens_to_fetch}
        data = angel.getMarketData("FULL", payload)
        
        oi_data = {}
        total_ce_oi = 0
        total_pe_oi = 0
        
        if data.get("status") and data.get("data") and data["data"].get("fetched"):
            for item in data["data"]["fetched"]:
                token = item.get("exchangeToken") or str(item.get("symbolToken", ""))
                if token in strike_map:
                    info = strike_map[token]
                    strike = info["strike"]
                    oi_val = item.get("opnInterest", 0)
                    
                    if strike not in oi_data:
                        oi_data[strike] = {"strike": strike, "ce_oi": 0, "pe_oi": 0}
                    
                    if info["type"] == "CE":
                        oi_data[strike]["ce_oi"] = oi_val
                        total_ce_oi += oi_val
                    else:
                        oi_data[strike]["pe_oi"] = oi_val
                        total_pe_oi += oi_val
        
        oi_list = sorted(oi_data.values(), key=lambda x: x["strike"])
        pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else None
        
        return {
            "oi_data": oi_list,
            "pcr": pcr,
            "atm_strike": atm,
            "expiry": nearest_expiry,
            "total_ce_oi": total_ce_oi,
            "total_pe_oi": total_pe_oi
        }
    except Exception as e:
        return {"oi_data": [], "pcr": None, "error": str(e)}

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
