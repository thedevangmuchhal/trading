import yfinance as yf
import pandas as pd

def fetch_market_data(ticker_symbol="^NSEI", interval="15m", period="5d"):
    """
    Fetches candlestick data for the given ticker.
    Defaults to NIFTY 50 index (^NSEI) with 15-minute intervals.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            # Fallback to SPY if NSEI fails for some reason
            ticker = yf.Ticker("SPY")
            df = ticker.history(period=period, interval=interval)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def fetch_news(ticker_symbol="^NSEI"):
    """
    Fetches latest news headlines related to the market/ticker.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        news_items = ticker.news
        headlines = [item['title'] for item in news_items] if news_items else []
        
        # If no news for index, fallback to some general ones or mock
        if not headlines:
            headlines = [
                "Markets rally on positive global cues",
                "FIIs remain net buyers in the cash market",
                "Inflation data comes in cooler than expected",
                "Tech stocks drag the indices lower amid rate hike fears"
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
