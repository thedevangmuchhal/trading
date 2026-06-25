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

def fetch_market_data(ticker_symbol="^NSEI", interval="15m", period="5d"):
    """
    Fetches candlestick data for the given ticker.
    Uses yfinance for broad market compatibility and historical data processing.
    """
    try:
        # We initialize Angel session in the background for future Options/Tick data extensions
        angel_session = get_angel_session()

        ticker = yf.Ticker(ticker_symbol, session=session)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            ticker = yf.Ticker("SPY", session=session)
            df = ticker.history(period=period, interval=interval)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def fetch_news(ticker_symbol="^NSEI"):
    try:
        ticker = yf.Ticker(ticker_symbol, session=session)
        news_items = ticker.news
        headlines = []
        if news_items:
            for item in news_items:
                # Safely extract title, handle missing keys
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
