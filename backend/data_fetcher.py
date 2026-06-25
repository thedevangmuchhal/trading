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
import ijson
import urllib.request
import tempfile
import os

# Global cache for only the filtered NIFTY tokens (saves massive memory)
angel_filtered_opts = None

def get_filtered_angel_options(base_symbol):
    """
    Downloads Angel One's massive JSON but streams it to save memory (Render 512MB limit).
    Filters only for options matching the base_symbol and caches the small resulting list.
    """
    global angel_filtered_opts
    
    # Base symbol handling (Angel One uses "NIFTY", not "^NSEI")
    angel_base_symbol = "NIFTY" if base_symbol == "^NSEI" else base_symbol

    if angel_filtered_opts is None:
        print("Streaming Angel One Scrip Master JSON to save memory...")
        try:
            url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            
            import ijson
            import urllib.request
            import tempfile
            import os
            
            # Download to a temporary file on disk rather than holding 35MB in RAM
            temp_path = os.path.join(tempfile.gettempdir(), "angel_tokens.json")
            if not os.path.exists(temp_path):
                urllib.request.urlretrieve(url, temp_path)
            
            angel_filtered_opts = []
            
            # Stream parse using ijson
            with open(temp_path, "rb") as f:
                # The JSON is an array of objects
                objects = ijson.items(f, "item")
                for obj in objects:
                    # Filter down instantly to only what we need
                    if obj.get("name") == angel_base_symbol and obj.get("instrumenttype") in ["OPTIDX", "OPTSTK"]:
                        angel_filtered_opts.append(obj)
            
            print(f"Successfully loaded {len(angel_filtered_opts)} options for {angel_base_symbol}")
            
        except Exception as e:
            print("Failed to download or parse Angel tokens:", e)
            return None
            
    return angel_filtered_opts

def get_angel_tokens(base_symbol, current_price):
    """
    Finds the exact CE and PE tokens for the nearest Expiry At-The-Money (ATM) strike.
    """
    opts = get_filtered_angel_options(base_symbol)
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
            "NFO": [ce_token, pe_token]
        }
        data = session.getMarketData("FULL", payload)
        if data.get('status') and data.get('data') and data['data'].get('fetched'):
            fetched = data['data']['fetched']
            ce_oi = 0
            pe_oi = 0
            for item in fetched:
                token = item.get('exchangeToken') or str(item.get('symbolToken', ''))
                if token == ce_token: ce_oi = item.get('opnInterest', 0)
                if token == pe_token: pe_oi = item.get('opnInterest', 0)
                
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
