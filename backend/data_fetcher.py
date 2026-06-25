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

_angel_session_cache = None
_oi_cache = {} # Cache for advanced OI data to prevent rate-limiting
_morning_oi_cache = {} # Track 9:15 AM OI for Buildup tracking

def get_angel_session():
    """
    Establishes a secure, SEBI-compliant connection to Angel One using TOTP.
    Caches the session to prevent exceeding API access rate limits.
    """
    global _angel_session_cache
    if _angel_session_cache is not None:
        return _angel_session_cache
        
    if not all([ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PIN, ANGEL_TOTP_SECRET]):
        return None
    try:
        obj = SmartConnect(api_key=ANGEL_API_KEY)
        totp_code = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_PIN, totp_code)
        if data.get('status'):
            print("Successfully connected to Angel One SmartAPI!")
            _angel_session_cache = obj
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

import threading
_oi_lock = threading.Lock()

from options_math import calculate_iv, calculate_delta

def fetch_advanced_oi(ticker_symbol, current_price):
    """
    Connects to Angel One SmartAPI, gets live Open Interest for +/- 5 strikes from ATM,
    and calculates broad PCR, Max Pain, Highest CE OI, and Highest PE OI.
    Results are cached for 15 seconds to prevent rate-limiting from parallel requests.
    """
    global _oi_cache
    
    with _oi_lock:
        now_ts = datetime.now().timestamp()
        if ticker_symbol in _oi_cache:
            cached = _oi_cache[ticker_symbol]
            if (now_ts - cached['timestamp']) < 15:
                return cached['data']
                
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
        
    opts = get_filtered_angel_options(base_symbol)
    if not opts:
        return None
        
    def parse_expiry(date_str):
        try: return datetime.strptime(date_str, "%d%b%Y")
        except: return datetime.max

    valid_opts = [x for x in opts if x.get("expiry")]
    now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    valid_opts = [x for x in valid_opts if parse_expiry(x["expiry"]) >= now]
    if not valid_opts: return None
    
    valid_opts.sort(key=lambda x: parse_expiry(x["expiry"]))
    nearest_expiry = valid_opts[0]["expiry"]
    current_expiry_opts = [x for x in valid_opts if x["expiry"] == nearest_expiry]
    
    def get_strike(opt):
        try: return float(opt["strike"]) / 100
        except: return 0
            
    atm = round(current_price / 50) * 50
    strikes_to_fetch = [atm + (i * 50) for i in range(-5, 6)]
    
    tokens_to_fetch = []
    strike_map = {}
    
    for strike in strikes_to_fetch:
        ce = next((x for x in current_expiry_opts if get_strike(x) == strike and x["symbol"].endswith("CE")), None)
        pe = next((x for x in current_expiry_opts if get_strike(x) == strike and x["symbol"].endswith("PE")), None)
        if ce:
            tokens_to_fetch.append(ce["token"])
            strike_map[ce["token"]] = {"strike": strike, "type": "CE"}
        if pe:
            tokens_to_fetch.append(pe["token"])
            strike_map[pe["token"]] = {"strike": strike, "type": "PE"}
            
    if not tokens_to_fetch: return None
        
    try:
        payload = {"NFO": tokens_to_fetch}
        data = session.getMarketData("FULL", payload)
        
        if data.get('status') and data.get('data') and data['data'].get('fetched'):
            fetched = data['data']['fetched']
            oi_data = {}
            total_ce_oi = 0
            total_pe_oi = 0
            total_ce_vol = 0
            total_pe_vol = 0
            highest_ce_oi = {"strike": 0, "oi": 0}
            highest_pe_oi = {"strike": 0, "oi": 0}
            atm_ce_ltp = 0
            atm_ce_vwap = 0
            atm_pe_ltp = 0
            atm_pe_vwap = 0
            
            for item in fetched:
                token = item.get('exchangeToken') or str(item.get('symbolToken', ''))
                if token in strike_map:
                    info = strike_map[token]
                    strike = info["strike"]
                    oi_val = item.get("opnInterest", 0)
                    vol_val = item.get("tradeVolume", 0)
                    
                    if strike == atm:
                        if info["type"] == "CE":
                            atm_ce_ltp = item.get("lastTradedPrice", 0)
                            atm_ce_vwap = item.get("averageTradedPrice", 0)
                        else:
                            atm_pe_ltp = item.get("lastTradedPrice", 0)
                            atm_pe_vwap = item.get("averageTradedPrice", 0)
                            
                    # Track Morning OI for buildup
                    cache_key = f"{ticker_symbol}_{strike}_{info['type']}"
                    if cache_key not in _morning_oi_cache:
                        _morning_oi_cache[cache_key] = oi_val
                    
                    if strike not in oi_data:
                        oi_data[strike] = {"strike": strike, "ce_oi": 0, "pe_oi": 0, "ce_oi_change": 0, "pe_oi_change": 0}
                    
                    oi_change = oi_val - _morning_oi_cache[cache_key]
                    
                    if info["type"] == "CE":
                        oi_data[strike]["ce_oi"] = oi_val
                        oi_data[strike]["ce_oi_change"] = oi_change
                        total_ce_oi += oi_val
                        total_ce_vol += vol_val
                        if oi_val > highest_ce_oi["oi"]:
                            highest_ce_oi = {"strike": strike, "oi": oi_val}
                    else:
                        oi_data[strike]["pe_oi"] = oi_val
                        oi_data[strike]["pe_oi_change"] = oi_change
                        total_pe_oi += oi_val
                        total_pe_vol += vol_val
                        if oi_val > highest_pe_oi["oi"]:
                            highest_pe_oi = {"strike": strike, "oi": oi_val}
            # Calculate Max Pain
            min_pain = float('inf')
            max_pain_strike = 0
            for test_strike in strikes_to_fetch:
                total_pain = 0
                for s in strikes_to_fetch:
                    if s not in oi_data: continue
                    # Call buyers lose if price < strike. Sellers lose if price > strike. Pain = value of ITM options
                    if s < test_strike:
                        total_pain += (test_strike - s) * oi_data[s]["ce_oi"]
                    if s > test_strike:
                        total_pain += (s - test_strike) * oi_data[s]["pe_oi"]
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = test_strike

            broad_pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else None
            
            oi_list = sorted(oi_data.values(), key=lambda x: x["strike"])
            
            # Calculate Option Greeks for ATM Strikes
            try:
                exp_date = datetime.strptime(nearest_expiry, "%d%b%Y")
                days_to_exp = max(0.5, (exp_date - datetime.now()).days)
                T = days_to_exp / 365.0
            except:
                T = 1.0 / 365.0
                
            r = 0.07 # 7% risk-free rate assumption for India
            
            # CE Greeks
            atm_ce_iv = calculate_iv(atm_ce_ltp, current_price, atm, T, r, "CE")
            atm_ce_delta = calculate_delta(current_price, atm, T, r, atm_ce_iv, "CE")
            
            # PE Greeks
            atm_pe_iv = calculate_iv(atm_pe_ltp, current_price, atm, T, r, "PE")
            atm_pe_delta = calculate_delta(current_price, atm, T, r, atm_pe_iv, "PE")
            
            # Intraday Buildup Totals
            total_ce_change = sum(d["ce_oi_change"] for d in oi_data.values())
            total_pe_change = sum(d["pe_oi_change"] for d in oi_data.values())
            
            vpcr = round(total_pe_vol / total_ce_vol, 2) if total_ce_vol > 0 else 0
            
            result = {
                "pcr": broad_pcr,
                "vpcr": vpcr,
                "max_pain": max_pain_strike,
                "highest_ce_strike": highest_ce_oi["strike"],
                "highest_pe_strike": highest_pe_oi["strike"],
                "oi_data": oi_list,
                "atm_strike": atm,
                "expiry": nearest_expiry,
                "total_ce_oi": total_ce_oi,
                "total_pe_oi": total_pe_oi,
                "total_ce_vol": total_ce_vol,
                "total_pe_vol": total_pe_vol,
                "atm_ce_ltp": atm_ce_ltp,
                "atm_ce_vwap": atm_ce_vwap,
                "atm_pe_ltp": atm_pe_ltp,
                "atm_pe_vwap": atm_pe_vwap,
                "greeks": {
                    "ce_iv": round(atm_ce_iv * 100, 2),
                    "pe_iv": round(atm_pe_iv * 100, 2),
                    "ce_delta": round(atm_ce_delta, 2),
                    "pe_delta": round(atm_pe_delta, 2)
                },
                "buildup": {
                    "ce_change": total_ce_change,
                    "pe_change": total_pe_change
                }
            }
            
            # Cache the result
            _oi_cache[ticker_symbol] = {
                "timestamp": now_ts,
                "data": result
            }
            
            return result
    except Exception as e:
        print(f"Error fetching Advanced Live OI from Angel: {e}")
        
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

def fetch_fii_dii():
    """Fetches end of day FII / DII cash data from NSE API."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        import requests
        s = requests.Session()
        s.headers.update(headers)
        # Hit homepage first to get cookies, required to bypass NSE WAF
        s.get('https://www.nseindia.com', timeout=5)
        # Fetch actual data
        res = s.get('https://www.nseindia.com/api/fiidiiTradeReact', timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"Error fetching FII/DII: {e}")
    return []

if __name__ == "__main__":
    df = fetch_market_data()
    print("Market Data Tail:")
    print(df.tail())
    print("\nNews:")
    print(fetch_news())
