import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from data_fetcher import fetch_market_data, fetch_news

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(headlines):
    if not headlines:
        return 0
    total_compound = 0
    for title in headlines:
        score = analyzer.polarity_scores(title)
        total_compound += score['compound']
    avg_compound = total_compound / len(headlines)
    return round(avg_compound * 100, 2)

def calculate_rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()

def analyze_technicals(df):
    if df.empty:
        return {}

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ATR'] = calculate_atr(df, 14)
    df['RSI'] = calculate_rsi(df['Close'], 14)

    latest = df.iloc[-1]
    current_price = latest['Close']
    
    trend = "Neutral"
    if latest['EMA_20'] > latest['EMA_50']:
        trend = "Bullish"
    elif latest['EMA_20'] < latest['EMA_50']:
        trend = "Bearish"

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
    df = fetch_market_data(ticker)
    headlines = fetch_news(ticker)
    
    sentiment_score = analyze_sentiment(headlines)
    tech_data = analyze_technicals(df)
    
    if not tech_data:
        return {"error": "Failed to fetch market data"}

    confidence = 50 
    
    if sentiment_score > 20: confidence += 15
    elif sentiment_score > 5: confidence += 5
    elif sentiment_score < -20: confidence -= 15
    elif sentiment_score < -5: confidence -= 5

    if tech_data['trend'] == "Bullish": confidence += 20
    elif tech_data['trend'] == "Bearish": confidence -= 20
    
    if tech_data['rsi'] < 40: confidence += 10 
    elif tech_data['rsi'] > 60: confidence -= 10 

    action = "WAIT"
    strike_type = None
    if confidence >= 70:
        action = "BUY"
        strike_type = "CE"
    elif confidence <= 30:
        action = "SELL" 
        strike_type = "PE"

    current_price = tech_data['current_price']
    strike_price = round(current_price / 100) * 100

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
