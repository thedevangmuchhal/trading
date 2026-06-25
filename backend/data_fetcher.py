import os
import yfinance as yf
import pandas as pd
import pyotp
import requests
from SmartApi import SmartConnect

# Setup custom session to bypass Yahoo Finance rate-limiting on Render
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
})

# -------------------------------------------------------------
# SECURE ENVIRONMENT VARIABLES (Do NOT hardcode passwords here!)
# -------------------------------------------------------------
ANGEL_API_KEY = os.environ.get("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID = os.environ.get("ANGEL_CLIENT_ID", "")
ANGEL_PIN = os.environ.get("ANGEL_PIN", "")
ANGEL_TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET", "")

def get_angel_session():
    """
    Establishes a secure, SEBI-compliant connection to Angel One using TOTP.
    """
    if not all([ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PIN, ANGEL_TOTP_SECRET]):
        return None
    try:
        obj = SmartConnect(api_key=ANGEL_API_KEY)
        totp_code = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_PIN, totp_code)
        if data.get('status'):
            print("Successfully connected to Angel One SmartAPI!")
            return obj
    except Exception as e:
        print(f"Angel Login Failed: {e}")
    return None

import time as _time

# Smart cache: stores {ticker: {data: df, timestamp: epoch}}
_market_cache = {}
_CACHE_TTL = 60  # seconds

def fetch_market_data(ticker_symbol="^NSEI", interval="15m", period="5d"):
    """
    Fetches candlestick data with a 60-second cache to avoid Yahoo rate-limiting.
    """
    cache_key = f"{ticker_symbol}_{interval}_{period}"
    now = _time.time()
    
    if cache_key in _market_cache:
        cached = _market_cache[cache_key]
        if now - cached["ts"] < _CACHE_TTL:
            return cached["data"]
    
    try:
        ticker = yf.Ticker(ticker_symbol, session=session)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            ticker = yf.Ticker("SPY", session=session)
            df = ticker.history(period=period, interval=interval)
        _market_cache[cache_key] = {"data": df, "ts": now}
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Return stale cache if available
        if cache_key in _market_cache:
            return _market_cache[cache_key]["data"]
        return pd.DataFrame()

import time
from datetime import datetime

# Global cache for the massive 30MB token map
angel_token_map = None

def get_angel_tokens(base_symbol, current_price):
    """
    Downloads and caches Angel One's massive JSON token list.
    Finds the exact CE and PE tokens for the nearest Expiry At-The-Money (ATM) strike.
    """
    global angel_token_map
    if angel_token_map is None:
        try:
            print("Downloading Angel One Scrip Master JSON (First time only)...")
            res = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json", timeout=20)
            angel_token_map = res.json()
        except Exception as e:
            print("Failed to download Angel tokens:", e)
            return None, None
            
    # Filter options for this symbol
    opts = [x for x in angel_token_map if x.get("name") == base_symbol and x.get("instrumenttype") in ["OPTIDX", "OPTSTK"]]
    if not opts:
        return None, None
        
    def parse_expiry(date_str):
        try:
            return datetime.strptime(date_str, "%d%b%Y")
        except:
            return datetime.max

    valid_opts = [x for x in opts if x.get("expiry")]
    now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    valid_opts = [x for x in valid_opts if parse_expiry(x["expiry"]) >= now]
    if not valid_opts: return None, None
    
    valid_opts.sort(key=lambda x: parse_expiry(x["expiry"]))
    nearest_expiry = valid_opts[0]["expiry"]
    
    current_expiry_opts = [x for x in valid_opts if x["expiry"] == nearest_expiry]
    
    def get_strike(opt):
        try:
            return float(opt["strike"]) / 100
        except:
            return 0
            
    current_expiry_opts.sort(key=lambda x: abs(get_strike(x) - current_price))
    if not current_expiry_opts: return None, None
    
    atm_strike = get_strike(current_expiry_opts[0])
    
    ce_token = next((x["token"] for x in current_expiry_opts if get_strike(x) == atm_strike and x["symbol"].endswith("CE")), None)
    pe_token = next((x["token"] for x in current_expiry_opts if get_strike(x) == atm_strike and x["symbol"].endswith("PE")), None)
    
    return ce_token, pe_token

def fetch_pcr(ticker_symbol, current_price):
    """
    Connects to Angel One SmartAPI, gets live Open Interest for ATM CE and PE,
    and calculates Put-Call Ratio.
    """
    session = get_angel_session()
    if not session:
        return None
        
    base_symbol = "NIFTY"
    if ticker_symbol == "^NSEI": base_symbol = "NIFTY"
    elif ticker_symbol == "^BSESN": base_symbol = "SENSEX"
    elif ticker_symbol == "^NSEBANK": base_symbol = "BANKNIFTY"
    elif ticker_symbol.endswith(".NS"): base_symbol = ticker_symbol.replace(".NS", "")
    else:
        return None # Not supported
        
    ce_token, pe_token = get_angel_tokens(base_symbol, current_price)
    if not ce_token or not pe_token:
        return None
        
    try:
        # Fetch Live OI from Angel
        payload = {
            "exchangeTokens": {
                "NFO": [ce_token, pe_token]
            }
        }
        data = session.getMarketData("FULL", payload)
        if data.get('status') and data.get('data') and data['data'].get('fetched'):
            fetched = data['data']['fetched']
            ce_oi = 0
            pe_oi = 0
            for item in fetched:
                if item['exchangeToken'] == ce_token: ce_oi = item['opnInterest']
                if item['exchangeToken'] == pe_token: pe_oi = item['opnInterest']
                
            if ce_oi > 0:
                return round(pe_oi / ce_oi, 2)
    except Exception as e:
        print(f"Error fetching Live OI from Angel: {e}")
        
    return None

def fetch_news(ticker_symbol="^NSEI"):
    try:
        ticker = yf.Ticker(ticker_symbol, session=session)
        news_items = ticker.news
        headlines = []
        if news_items:
            for item in news_items:
                title = item.get('title')
                if title:
                    headlines.append(title)
        
        if not headlines:
            headlines = [
                "Markets trade in a tight range ahead of global cues",
                "FIIs remain active in the derivatives market",
                "Option sellers dominate out-of-the-money strikes",
                "Institutional volume steadily rising at VWAP levels"
            ]
        return headlines
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

if __name__ == "__main__":
    df = fetch_market_data()
    print("Market Data Tail:")
    print(df.tail())
    print("\nNews:")
    print(fetch_news())
