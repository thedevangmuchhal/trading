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
    avg_compound = total_compound / len(headlines) if headlines else 0
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
        return pd.Series(index=df.index, dtype=float) 
    q = df['Volume']
    p = (df['High'] + df['Low'] + df['Close']) / 3
    return (p * q).cumsum() / q.cumsum()

def analyze_technicals(df):
    if df.empty:
        return {}

    # Calculate indicators using pure pandas
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    df['ATR'] = calculate_atr(df, 14)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    macd, signal_line, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = hist
    df['ADX'] = calculate_adx(df, 14)
    df['VWAP'] = calculate_vwap(df)

    # Chart Patterns
    body = (df['Close'] - df['Open']).abs()
    shadow_upper = df['High'] - df[['Close', 'Open']].max(axis=1)
    shadow_lower = df[['Close', 'Open']].min(axis=1) - df['Low']
    
    df['Is_Doji'] = body < (0.1 * (df['High'] - df['Low']))
    df['Is_Hammer'] = (shadow_lower > (2 * body)) & (shadow_upper < (0.2 * body))
    
    prev_body = (df['Close'].shift(1) - df['Open'].shift(1))
    curr_body = (df['Close'] - df['Open'])
    df['Bullish_Engulfing'] = (prev_body < 0) & (curr_body > 0) & (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))
    df['Bearish_Engulfing'] = (prev_body > 0) & (curr_body < 0) & (df['Close'] < df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1))

    latest = df.iloc[-1]
    current_price = latest['Close']
    
    # Advanced Trend Detection (9/21 EMA crossover)
    trend = "Neutral"
    if latest['EMA_9'] > latest['EMA_21']:
        trend = "Bullish"
    elif latest['EMA_9'] < latest['EMA_21']:
        trend = "Bearish"

    # MACD Momentum
    momentum = "Neutral"
    if latest['MACD_Hist'] > 0 and latest['MACD'] > latest['MACD_Signal']:
        momentum = "Strong Bullish"
    elif latest['MACD_Hist'] < 0 and latest['MACD'] < latest['MACD_Signal']:
        momentum = "Strong Bearish"
        
    chart_pattern = "None"
    if latest['Bullish_Engulfing']: chart_pattern = "Bullish Engulfing"
    elif latest['Bearish_Engulfing']: chart_pattern = "Bearish Engulfing"
    elif latest['Is_Hammer']: chart_pattern = "Hammer"
    elif latest['Is_Doji']: chart_pattern = "Doji"

    # Pine Script ORB Logic (Opening Range Breakout)
    orb_breakout = "None"
    or_high = None
    or_low = None
    try:
        today = df.index[-1].date()
        today_df = df[df.index.date == today]
        # Get candles before 10:00 AM (first 45 mins)
        or_df = today_df.between_time("09:15", "10:00", inclusive="left")
        if not or_df.empty:
            or_high = or_df['High'].max()
            or_low = or_df['Low'].min()
            
            # If current time is after 10:00 AM, check for breakout
            if latest.name.time() >= pd.to_datetime("10:00").time():
                if current_price > or_high:
                    orb_breakout = "Bullish"
                elif current_price < or_low:
                    orb_breakout = "Bearish"
    except Exception as e:
        pass

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
        "chart_pattern": chart_pattern,
        "orb_breakout": orb_breakout,
        "or_high": or_high,
        "or_low": or_low,
        "ema_9": round(latest['EMA_9'], 2),
        "ema_21": round(latest['EMA_21'], 2),
        "bb_upper": round(latest['BB_Upper'], 2) if not pd.isna(latest['BB_Upper']) else None,
        "bb_lower": round(latest['BB_Lower'], 2) if not pd.isna(latest['BB_Lower']) else None,
        "rsi": round(latest['RSI'], 2) if not pd.isna(latest['RSI']) else 50,
        "atr": round(latest['ATR'], 2) if not pd.isna(latest['ATR']) else 10,
        "adx": round(latest['ADX'], 2) if not pd.isna(latest['ADX']) else 0,
        "vwap": round(latest['VWAP'], 2) if not pd.isna(latest['VWAP']) else None,
        "support": round(s1, 2),
        "resistance": round(r1, 2)
    }

def generate_signals(ticker="^NSEI"):
    from data_fetcher import fetch_advanced_oi, fetch_fii_dii, fetch_market_data, fetch_news, fetch_vix, calculate_vpvr
    from ml_engine import predict_breakout_probability
    
    # 1. Multi-Timeframe (MTF) Data Fetching
    df_15m = fetch_market_data(ticker, interval="15m")
    df_1h = fetch_market_data(ticker, interval="1h")
    
    headlines = fetch_news(ticker)
    sentiment_val = analyze_sentiment(headlines)
    
    tech_data = analyze_technicals(df_15m)
    if not tech_data:
        return {"error": "Failed to fetch market data"}
        
    # Calculate POC (Volume Profile)
    poc = calculate_vpvr(df_15m)
    
    # Check 1H Macro Trend
    macro_tech = analyze_technicals(df_1h) if not df_1h.empty else tech_data
    macro_trend = macro_tech.get('trend', 'Neutral')

    current_price = tech_data['current_price']
    
    # Fetch Advanced Options Data from Angel One API
    oi_metrics = fetch_advanced_oi(ticker, current_price)
    
    # Fetch VIX Data
    vix_data = fetch_vix()

    # Initialize Combo Scores
    smart_money_score = 50
    options_score = 50
    technical_score = 50
    sentiment_score = 50
    
    # -----------------------------------------------------
    # SMART MONEY SCORE (Institutions)
    # -----------------------------------------------------
    fii_dii_data = fetch_fii_dii()
    fii_net = 0
    dii_net = 0
    if fii_dii_data:
        for item in fii_dii_data:
            try:
                if item.get("category") == "FII/FPI":
                    fii_net = float(item.get("netValue", 0))
                elif item.get("category") == "DII":
                    dii_net = float(item.get("netValue", 0))
            except:
                pass
                
        if fii_net > 0 and dii_net > 0:
            smart_money_score += 25
        elif fii_net < 0 and dii_net < 0:
            smart_money_score -= 25
        
        if fii_net > 5000: smart_money_score += 15
        elif fii_net < -5000: smart_money_score -= 15
        if dii_net > 5000: smart_money_score += 10
        elif dii_net < -5000: smart_money_score -= 10

    # -----------------------------------------------------
    # SENTIMENT SCORE
    # -----------------------------------------------------
    event_warning = False
    event_keywords = ["rbi", "fed", "policy", "rate", "budget", "cpi", "inflation", "election", "war", "crash"]
    for headline in headlines:
        if any(keyword in headline.lower() for keyword in event_keywords):
            event_warning = True
            sentiment_score -= 30
            break
            
    sentiment_score += (sentiment_val * 0.4) # Scale sentiment (-100 to 100)
    
    if vix_data and 'pct_change' in vix_data:
        if vix_data['pct_change'] < -3.0:
            sentiment_score -= 15 # VIX Crashing, Option Premium Melting
        elif vix_data['pct_change'] > 3.0:
            sentiment_score += 10 # VIX Rising, Fear increasing
        
    # -----------------------------------------------------
    # OPTIONS SCORE
    # -----------------------------------------------------
    pcr = None
    vpcr = None
    if oi_metrics:
        pcr = oi_metrics.get('pcr')
        if pcr is not None:
            if pcr > 1.2: options_score += 25
            elif pcr > 1.0: options_score += 10
            elif pcr < 0.6: options_score -= 25
            elif pcr < 0.8: options_score -= 10
            
        # Max Pain Magnetism
        if oi_metrics.get('max_pain', 0) > 0:
            if current_price < (oi_metrics['max_pain'] - tech_data['atr']):
                options_score += 15
            elif current_price > (oi_metrics['max_pain'] + tech_data['atr']):
                options_score -= 15

        # VWAP & VPCR
        total_ce_vol = oi_metrics.get('total_ce_vol', 0)
        total_pe_vol = oi_metrics.get('total_pe_vol', 0)
        if total_ce_vol > 0:
            vpcr = round(total_pe_vol / total_ce_vol, 2)
            if vpcr > 1.2: options_score += 15
            elif vpcr < 0.8: options_score -= 15
            
        # Intraday OI Buildup (Change in OI)
        buildup = oi_metrics.get('buildup', {})
        ce_change = buildup.get('ce_change', 0)
        pe_change = buildup.get('pe_change', 0)
        if ce_change > pe_change * 1.5 and ce_change > 0:
            options_score -= 15 # Aggressive Call Writing (Bearish)
        elif pe_change > ce_change * 1.5 and pe_change > 0:
            options_score += 15 # Aggressive Put Writing (Bullish)
            
        # IV Skew
        greeks = oi_metrics.get('greeks', {})
        iv_skew = greeks.get('ce_iv', 0) - greeks.get('pe_iv', 0)
        if iv_skew > 3: options_score += 15
        elif iv_skew < -3: options_score -= 15

    # -----------------------------------------------------
    # TECHNICAL SCORE & PINE SCRIPT ORB LOGIC
    # -----------------------------------------------------
    if tech_data['trend'] == "Bullish": technical_score += 15
    elif tech_data['trend'] == "Bearish": technical_score -= 15

    if tech_data['momentum'] == "Strong Bullish": technical_score += 10
    elif tech_data['momentum'] == "Strong Bearish": technical_score -= 10
    
    if tech_data['bb_lower'] and current_price < tech_data['bb_lower']: technical_score += 10
    if tech_data['bb_upper'] and current_price > tech_data['bb_upper']: technical_score -= 10

    if poc is not None:
        if current_price > poc: technical_score += 10
        else: technical_score -= 10
        
    if tech_data['vwap'] is not None and not np.isnan(tech_data['vwap']):
        if current_price > tech_data['vwap']: technical_score += 10
        else: technical_score -= 10

    if tech_data['chart_pattern'] == "Bullish Engulfing": technical_score += 15
    elif tech_data['chart_pattern'] == "Hammer": technical_score += 10
    elif tech_data['chart_pattern'] == "Bearish Engulfing": technical_score -= 15

    # PINE SCRIPT ORB Integration
    if tech_data['orb_breakout'] == "Bullish": technical_score += 30
    elif tech_data['orb_breakout'] == "Bearish": technical_score -= 30

    ml_prob = predict_breakout_probability(tech_data)
    if ml_prob is not None:
        if ml_prob > 0.75: technical_score += 20
        elif ml_prob > 0.60: technical_score += 10
        elif ml_prob < 0.25: technical_score -= 20
        elif ml_prob < 0.40: technical_score -= 10

    # MTF Confluence Check
    if tech_data['trend'] == "Bullish" and macro_trend == "Bearish":
        technical_score -= 20 # Bull Trap
    elif tech_data['trend'] == "Bearish" and macro_trend == "Bullish":
        technical_score += 20 # Bear Trap
        
    market_condition = "Trending"
    if tech_data['adx'] < 20:
        technical_score -= 30 
        market_condition = "Choppy/Sideways"

    # Clamp scores
    smart_money_score = int(max(0, min(100, smart_money_score)))
    options_score = int(max(0, min(100, options_score)))
    technical_score = int(max(0, min(100, technical_score)))
    sentiment_score = int(max(0, min(100, sentiment_score)))

    # Calculate global confidence average
    confidence = int((smart_money_score + options_score + technical_score + sentiment_score) / 4)

    # -----------------------------------------------------
    # ACTION TRIGGER COMBO LOGIC
    # -----------------------------------------------------
    action = "WAIT"
    strike_type = None

    if smart_money_score >= 80 and options_score >= 60:
        action = "BUY"
        strike_type = "CE"
        confidence = max(confidence, 85) # Smart Money Override
    elif smart_money_score <= 20 and options_score <= 40:
        action = "SELL"
        strike_type = "PE"
        confidence = min(confidence, 15)
        
    elif technical_score >= 80 and options_score >= 55:
        action = "BUY"
        strike_type = "CE"
        confidence = max(confidence, 80) # Pine Script Breakout Override
    elif technical_score <= 20 and options_score <= 45:
        action = "SELL"
        strike_type = "PE"
        confidence = min(confidence, 20)
        
    elif confidence >= 75:
        action = "BUY"
        strike_type = "CE"
    elif confidence <= 25:
        action = "SELL"
        strike_type = "PE"

    import math
    atm_strike = round(current_price / 100) * 100
    strike_price = atm_strike
    atr = tech_data['atr']
    
    if action == "BUY":
        strike_price = math.floor(current_price / 100) * 100
        entry = current_price
        if oi_metrics and oi_metrics.get('highest_ce_strike') and oi_metrics.get('highest_ce_strike') > entry:
            target = oi_metrics['highest_ce_strike']
        else:
            target = entry + (atr * 3.0)
        if oi_metrics and oi_metrics.get('highest_pe_strike', 0) > 0 and oi_metrics.get('highest_pe_strike') < entry:
            stop_loss = max(oi_metrics['highest_pe_strike'], entry - (atr * 2.0))
        else:
            stop_loss = entry - (atr * 1.5)
            
    elif action == "SELL":
        strike_price = math.ceil(current_price / 100) * 100
        entry = current_price
        if oi_metrics and oi_metrics.get('highest_pe_strike', 0) > 0 and oi_metrics.get('highest_pe_strike') < entry:
            target = oi_metrics['highest_pe_strike']
        else:
            target = entry - (atr * 3.0) 
        if oi_metrics and oi_metrics.get('highest_ce_strike') and oi_metrics.get('highest_ce_strike') > entry:
            stop_loss = min(oi_metrics['highest_ce_strike'], entry + (atr * 2.0))
        else:
            stop_loss = entry + (atr * 1.5) 
    else:
        entry = stop_loss = target = 0

    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "ticker": ticker,
        "current_price": current_price,
        "sentiment_score_raw": sentiment_val,
        "tech_trend": f"{tech_data['trend']} ({tech_data['momentum']}) [{market_condition}]",
        "market_condition": market_condition,
        "rsi": tech_data['rsi'],
        
        "confidence_score": confidence,
        "smart_money_score": smart_money_score,
        "options_score": options_score,
        "technical_score": technical_score,
        "sentiment_score": sentiment_score,
        
        "pcr": pcr,
        "vpcr": vpcr,
        "max_pain": oi_metrics['max_pain'] if oi_metrics else None,
        "chart_pattern": tech_data['chart_pattern'],
        "orb_breakout": tech_data['orb_breakout'],
        
        "action": action,
        "strike_recommendation": f"{strike_price} {strike_type}" if strike_type else "None",
        "entry_level": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "support": tech_data['support'],
        "resistance": tech_data['resistance'],
        "recent_headlines": headlines[:3],
        "event_warning": event_warning,
        "fii_net": fii_net,
        "dii_net": dii_net
    }

if __name__ == "__main__":
    import json
    print(json.dumps(generate_signals(), indent=2))
