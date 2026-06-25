import pandas as pd
import numpy as np
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

def calculate_macd(data):
    ema12 = data.ewm(span=12, adjust=False).mean()
    ema26 = data.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_adx(df, window=14):
    # Average Directional Index (ADX) to measure trend strength
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    
    pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    pos_dm = pd.Series(pos_dm, index=df.index).ewm(alpha=1/window, adjust=False).mean()
    neg_dm = pd.Series(neg_dm, index=df.index).ewm(alpha=1/window, adjust=False).mean()
    
    tr = calculate_atr(df, window)
    
    pos_di = 100 * (pos_dm / tr)
    neg_di = 100 * (neg_dm / tr)
    
    dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10))
    adx = dx.ewm(alpha=1/window, adjust=False).mean()
    return adx

def calculate_vwap(df):
    if df['Volume'].sum() == 0:
        return pd.Series(index=df.index, dtype=float) # No volume data
    q = df['Volume']
    p = (df['High'] + df['Low'] + df['Close']) / 3
    return (p * q).cumsum() / q.cumsum()

def analyze_technicals(df):
    if df.empty:
        return {}

    # Calculate indicators using pure pandas
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ATR'] = calculate_atr(df, 14)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    macd, signal_line, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = hist
    df['ADX'] = calculate_adx(df, 14)
    df['VWAP'] = calculate_vwap(df)

    latest = df.iloc[-1]
    current_price = latest['Close']
    
    # Advanced Trend Detection
    trend = "Neutral"
    if latest['EMA_20'] > latest['EMA_50']:
        trend = "Bullish"
    elif latest['EMA_20'] < latest['EMA_50']:
        trend = "Bearish"

    # MACD Momentum
    momentum = "Neutral"
    if latest['MACD_Hist'] > 0 and latest['MACD'] > latest['MACD_Signal']:
        momentum = "Strong Bullish"
    elif latest['MACD_Hist'] < 0 and latest['MACD'] < latest['MACD_Signal']:
        momentum = "Strong Bearish"

    high = df['High'].max()
    low = df['Low'].min()
    close = latest['Close']
    
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    
    return {
        "current_price": round(current_price, 2),
        "trend": trend,
        "momentum": momentum,
        "rsi": round(latest['RSI'], 2) if not pd.isna(latest['RSI']) else 50,
        "atr": round(latest['ATR'], 2) if not pd.isna(latest['ATR']) else 10,
        "adx": round(latest['ADX'], 2) if not pd.isna(latest['ADX']) else 0,
        "vwap": round(latest['VWAP'], 2) if not pd.isna(latest['VWAP']) else None,
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

    # Base Confidence
    confidence = 50 
    
    # 1. Sentiment Weight (Max +/- 15)
    if sentiment_score > 20: confidence += 15
    elif sentiment_score > 5: confidence += 5
    elif sentiment_score < -20: confidence -= 15
    elif sentiment_score < -5: confidence -= 5

    # 2. Trend Weight (Max +/- 20)
    if tech_data['trend'] == "Bullish": confidence += 20
    elif tech_data['trend'] == "Bearish": confidence -= 20

    # 3. MACD Momentum Weight (Max +/- 15) - Adds accuracy
    if tech_data['momentum'] == "Strong Bullish": confidence += 15
    elif tech_data['momentum'] == "Strong Bearish": confidence -= 15
    
    # 4. RSI Overbought/Oversold Reversals (Max +/- 10)
    if tech_data['rsi'] < 35: confidence += 10 
    elif tech_data['rsi'] > 65: confidence -= 10 

    # 5. VWAP Institutional Volume Filter (Max +/- 10)
    if tech_data['vwap'] is not None and not np.isnan(tech_data['vwap']):
        if tech_data['current_price'] > tech_data['vwap']: confidence += 10
        else: confidence -= 10

    # 6. ADX Choppy Market Filter (OVERRIDE)
    # If ADX is below 20, the market is entirely sideways. Do not trade.
    market_condition = "Trending"
    if tech_data['adx'] < 20:
        confidence -= 50 # Brutally penalize confidence
        market_condition = "Choppy/Sideways"

    # Clamp confidence to 0-100
    confidence = max(0, min(100, confidence))

    # Stricter Signal Generation for higher accuracy
    action = "WAIT"
    strike_type = None
    if confidence >= 80: 
        action = "BUY"
        strike_type = "CE"
    elif confidence <= 20: 
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
        "tech_trend": f"{tech_data['trend']} ({tech_data['momentum']}) [{market_condition}]",
        "rsi": tech_data['rsi'],
        "confidence_score": int(confidence),
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
