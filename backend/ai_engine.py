import pandas as pd
import pandas_ta as ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from data_fetcher import fetch_market_data, fetch_news
import math

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(headlines):
    """
    Analyzes a list of news headlines using VADER sentiment analysis.
    Returns a score from -100 to +100.
    """
    if not headlines:
        return 0

    total_compound = 0
    for title in headlines:
        score = analyzer.polarity_scores(title)
        total_compound += score['compound']
    
    avg_compound = total_compound / len(headlines)
    # Convert -1.0 to 1.0 range into -100 to 100
    return round(avg_compound * 100, 2)

def analyze_technicals(df):
    """
    Calculates technical indicators on the dataframe.
    Returns a dict with latest technical data.
    """
    if df.empty:
        return {}

    # Calculate indicators
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    current_price = latest['Close']
    
    # Trend Bias
    trend = "Neutral"
    if latest['EMA_20'] > latest['EMA_50']:
        trend = "Bullish"
    elif latest['EMA_20'] < latest['EMA_50']:
        trend = "Bearish"

    # Support / Resistance Pivots (Simple calculation)
    # Let's use standard pivot points from the previous day/period
    high = df['High'].max()
    low = df['Low'].min()
    close = latest['Close']
    
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    
    return {
        "current_price": round(current_price, 2),
        "trend": trend,
        "rsi": round(latest['RSI'], 2) if not pd.isna(latest['RSI']) else 50,
        "atr": round(latest['ATR'], 2) if not pd.isna(latest['ATR']) else 10,
        "support": round(s1, 2),
        "resistance": round(r1, 2)
    }

def generate_signals(ticker="^NSEI"):
    """
    Master function to fetch data and generate Buy/Sell signals.
    """
    df = fetch_market_data(ticker)
    headlines = fetch_news(ticker)
    
    sentiment_score = analyze_sentiment(headlines)
    tech_data = analyze_technicals(df)
    
    if not tech_data:
        return {"error": "Failed to fetch market data"}

    # Calculate AI Confidence Score
    confidence = 50 # Base
    
    # Add sentiment
    if sentiment_score > 20: confidence += 15
    elif sentiment_score > 5: confidence += 5
    elif sentiment_score < -20: confidence -= 15
    elif sentiment_score < -5: confidence -= 5

    # Add technicals
    if tech_data['trend'] == "Bullish": confidence += 20
    elif tech_data['trend'] == "Bearish": confidence -= 20
    
    if tech_data['rsi'] < 40: confidence += 10 # Oversold, favor upside
    elif tech_data['rsi'] > 60: confidence -= 10 # Overbought, favor downside

    # Determine Signal
    action = "WAIT"
    strike_type = None
    if confidence >= 70:
        action = "BUY"
        strike_type = "CE"
    elif confidence <= 30:
        action = "SELL" # For options, buy Put
        strike_type = "PE"

    # Determine Strike (Round to nearest 100 for NIFTY)
    current_price = tech_data['current_price']
    strike_price = round(current_price / 100) * 100

    # Determine Entry, Stop, Target based on ATR
    atr = tech_data['atr']
    if action == "BUY":
        entry = current_price
        stop_loss = entry - (atr * 1.5)
        target = entry + (atr * 3.0)
    elif action == "SELL":
        entry = current_price
        stop_loss = entry + (atr * 1.5)
        target = entry - (atr * 3.0)
    else:
        entry = stop_loss = target = 0

    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "ticker": ticker,
        "current_price": current_price,
        "sentiment_score": sentiment_score,
        "tech_trend": tech_data['trend'],
        "rsi": tech_data['rsi'],
        "confidence_score": confidence,
        "action": action,
        "strike_recommendation": f"{strike_price} {strike_type}" if strike_type else "None",
        "entry_level": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "support": tech_data['support'],
        "resistance": tech_data['resistance'],
        "recent_headlines": headlines[:3]
    }

if __name__ == "__main__":
    import json
    print(json.dumps(generate_signals(), indent=2))
