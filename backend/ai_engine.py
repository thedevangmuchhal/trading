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

# ---------------------------------------------------------------
# NEW: Stochastic RSI
# ---------------------------------------------------------------
def calculate_stochastic_rsi(rsi_series, period=14, smooth_k=3, smooth_d=3):
    """
    Stochastic RSI = (RSI - RSI_Low) / (RSI_High - RSI_Low)
    K% = SMA(StochRSI, smooth_k)
    D% = SMA(K%, smooth_d)
    Returns (K%, D%) series.
    """
    rsi_min = rsi_series.rolling(window=period).min()
    rsi_max = rsi_series.rolling(window=period).max()
    stoch_rsi = (rsi_series - rsi_min) / (rsi_max - rsi_min + 1e-10)
    k_line = stoch_rsi.rolling(window=smooth_k).mean() * 100
    d_line = k_line.rolling(window=smooth_d).mean()
    return k_line, d_line

# ---------------------------------------------------------------
# NEW: Supertrend Indicator (popular for Indian markets)
# ---------------------------------------------------------------
def calculate_supertrend(df, period=10, multiplier=3.0):
    """
    Supertrend uses ATR bands around HL2 midpoint.
    Returns a Series: positive value = bullish, negative = bearish.
    The returned value is the Supertrend line price itself.
    Also returns direction: 1 = bullish, -1 = bearish.
    """
    hl2 = (df['High'] + df['Low']) / 2
    
    # True Range and ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)  # 1 = bullish, -1 = bearish
    
    for i in range(1, len(df)):
        # Adjust bands based on previous values
        if lower_band.iloc[i] > lower_band.iloc[i-1] or df['Close'].iloc[i-1] < lower_band.iloc[i-1]:
            pass  # keep current lower band
        else:
            lower_band.iloc[i] = lower_band.iloc[i-1]
            
        if upper_band.iloc[i] < upper_band.iloc[i-1] or df['Close'].iloc[i-1] > upper_band.iloc[i-1]:
            pass  # keep current upper band
        else:
            upper_band.iloc[i] = upper_band.iloc[i-1]
        
        # Determine direction
        if supertrend.iloc[i-1] == upper_band.iloc[i-1]:
            if df['Close'].iloc[i] > upper_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
        else:
            if df['Close'].iloc[i] < lower_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
    
    return supertrend, direction

# ---------------------------------------------------------------
# NEW: RSI Divergence Detection
# ---------------------------------------------------------------
def detect_rsi_divergence(df, rsi_series, lookback=5):
    """
    Compares price slope vs RSI slope over the last `lookback` candles.
    Bullish divergence: price making lower lows, RSI making higher lows.
    Bearish divergence: price making higher highs, RSI making lower highs.
    Returns: 'bullish', 'bearish', or 'none'
    """
    if len(df) < lookback + 1 or len(rsi_series) < lookback + 1:
        return 'none'
    
    recent_prices = df['Close'].iloc[-lookback:]
    recent_rsi = rsi_series.iloc[-lookback:]
    
    # Skip if any NaN
    if recent_prices.isna().any() or recent_rsi.isna().any():
        return 'none'
    
    # Calculate slopes using linear regression approximation (simple: last - first)
    price_slope = recent_prices.iloc[-1] - recent_prices.iloc[0]
    rsi_slope = recent_rsi.iloc[-1] - recent_rsi.iloc[0]
    
    # Bullish divergence: price falling but RSI rising
    if price_slope < 0 and rsi_slope > 0:
        # Confirm with low comparisons
        price_low_recent = recent_prices.min()
        price_low_prev = df['Close'].iloc[-(lookback*2):-lookback].min() if len(df) >= lookback * 2 else price_low_recent
        rsi_at_price_low_recent = recent_rsi.min()
        rsi_at_price_low_prev = rsi_series.iloc[-(lookback*2):-lookback].min() if len(rsi_series) >= lookback * 2 else rsi_at_price_low_recent
        
        if price_low_recent < price_low_prev and rsi_at_price_low_recent > rsi_at_price_low_prev:
            return 'bullish'
    
    # Bearish divergence: price rising but RSI falling
    elif price_slope > 0 and rsi_slope < 0:
        price_high_recent = recent_prices.max()
        price_high_prev = df['Close'].iloc[-(lookback*2):-lookback].max() if len(df) >= lookback * 2 else price_high_recent
        rsi_at_price_high_recent = recent_rsi.max()
        rsi_at_price_high_prev = rsi_series.iloc[-(lookback*2):-lookback].max() if len(rsi_series) >= lookback * 2 else rsi_at_price_high_recent
        
        if price_high_recent > price_high_prev and rsi_at_price_high_recent < rsi_at_price_high_prev:
            return 'bearish'
    
    return 'none'

# ---------------------------------------------------------------
# NEW: Volume Surge Detection
# ---------------------------------------------------------------
def detect_volume_surge(df, avg_period=20, threshold=2.0):
    """
    Returns True if the latest candle's volume exceeds `threshold` x
    the `avg_period`-period average volume.
    Also returns the ratio for display.
    """
    if len(df) < avg_period + 1 or 'Volume' not in df.columns:
        return False, 1.0
    
    avg_vol = df['Volume'].iloc[-(avg_period+1):-1].mean()
    if avg_vol <= 0:
        return False, 1.0
    
    current_vol = df['Volume'].iloc[-1]
    ratio = current_vol / avg_vol
    return ratio >= threshold, round(ratio, 2)


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

    # NEW: Stochastic RSI
    stoch_k, stoch_d = calculate_stochastic_rsi(df['RSI'])
    df['StochRSI_K'] = stoch_k
    df['StochRSI_D'] = stoch_d

    # NEW: Supertrend
    supertrend_line, supertrend_dir = calculate_supertrend(df)
    df['Supertrend'] = supertrend_line
    df['Supertrend_Dir'] = supertrend_dir

    # NEW: RSI Divergence
    rsi_divergence = detect_rsi_divergence(df, df['RSI'])

    # NEW: Volume Surge
    volume_surge, volume_ratio = detect_volume_surge(df)

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

    # Supertrend direction for latest candle
    supertrend_direction = "Bullish" if latest['Supertrend_Dir'] == 1 else "Bearish"
    
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
        "resistance": round(r1, 2),
        # NEW fields
        "stoch_rsi_k": round(latest['StochRSI_K'], 2) if not pd.isna(latest['StochRSI_K']) else 50,
        "stoch_rsi_d": round(latest['StochRSI_D'], 2) if not pd.isna(latest['StochRSI_D']) else 50,
        "supertrend": round(latest['Supertrend'], 2) if not pd.isna(latest['Supertrend']) else None,
        "supertrend_direction": supertrend_direction,
        "rsi_divergence": rsi_divergence,
        "volume_surge": volume_surge,
        "volume_ratio": volume_ratio,
    }

def generate_signals(ticker="^NSEI"):
    from data_fetcher import fetch_advanced_oi, fetch_fii_dii, fetch_market_data, fetch_news, fetch_vix, calculate_vpvr
    from ml_engine import predict_breakout_probability
    
    # 1. Multi-Timeframe (MTF) Data Fetching
    df_15m = fetch_market_data(ticker, interval="15m")
    df_1h = fetch_market_data(ticker, interval="1h")
    df_4h = fetch_market_data(ticker, interval="1h", period="1mo")  # Approximate 4h via 1mo of 1h data
    
    headlines = fetch_news(ticker)
    sentiment_val = analyze_sentiment(headlines)
    
    tech_data = analyze_technicals(df_15m)
    if not tech_data:
        return {"error": "Failed to fetch market data"}
        
    # Calculate POC (Volume Profile)
    poc = calculate_vpvr(df_15m)
    
    # Check 1H Macro Trend
    macro_tech = analyze_technicals(df_1h) if not df_1h.empty else tech_data

    # NEW: Check 4H Higher-Timeframe Trend (resample 1h data to approximate 4h)
    htf_tech = None
    if not df_4h.empty and len(df_4h) >= 20:
        try:
            df_4h_resampled = df_4h.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            if not df_4h_resampled.empty:
                htf_tech = analyze_technicals(df_4h_resampled)
        except Exception:
            htf_tech = None

    macro_trend = macro_tech.get('trend', 'Neutral')
    htf_trend = htf_tech.get('trend', 'Neutral') if htf_tech else 'Neutral'

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
    
    # NEW: IV Crush Warning
    iv_crush_warning = False
    if vix_data and 'pct_change' in vix_data:
        if vix_data['pct_change'] < -5.0:
            iv_crush_warning = True
            sentiment_score -= 20  # Extra penalty for IV crush
        elif vix_data['pct_change'] < -3.0:
            sentiment_score -= 15 # VIX Crashing, Option Premium Melting
        elif vix_data['pct_change'] > 3.0:
            sentiment_score += 10 # VIX Rising, Fear increasing
        
    # -----------------------------------------------------
    # OPTIONS SCORE
    # -----------------------------------------------------
    pcr = None
    vpcr = None
    atm_ce_ltp = 0
    atm_pe_ltp = 0
    if oi_metrics:
        pcr = oi_metrics.get('pcr')
        atm_ce_ltp = oi_metrics.get('atm_ce_ltp', 0)
        atm_pe_ltp = oi_metrics.get('atm_pe_ltp', 0)
        
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

    # NEW: Supertrend contribution (replaces weak EMA overlap)
    if tech_data.get('supertrend_direction') == "Bullish":
        technical_score += 15
    elif tech_data.get('supertrend_direction') == "Bearish":
        technical_score -= 15

    # NEW: RSI Divergence
    rsi_divergence = tech_data.get('rsi_divergence', 'none')
    if rsi_divergence == 'bullish':
        technical_score += 15
    elif rsi_divergence == 'bearish':
        technical_score -= 15

    # NEW: Stochastic RSI crossover
    stoch_k = tech_data.get('stoch_rsi_k', 50)
    stoch_d = tech_data.get('stoch_rsi_d', 50)
    if stoch_k < 20 and stoch_k > stoch_d:
        technical_score += 10  # Oversold crossover (bullish)
    elif stoch_k > 80 and stoch_k < stoch_d:
        technical_score -= 10  # Overbought crossover (bearish)

    # NEW: Volume Surge confirmation
    if tech_data.get('volume_surge', False):
        # Surge on a bullish candle amplifies, on bearish candle dampens
        if tech_data['trend'] == "Bullish":
            technical_score += 10
        elif tech_data['trend'] == "Bearish":
            technical_score -= 10

    # ENHANCED MTF Confluence Check (now uses 3 timeframes: 15m, 1h, 4h)
    trends_15m = tech_data['trend']
    trends_1h = macro_trend
    trends_4h = htf_trend

    bullish_count = sum(1 for t in [trends_15m, trends_1h, trends_4h] if t == "Bullish")
    bearish_count = sum(1 for t in [trends_15m, trends_1h, trends_4h] if t == "Bearish")

    if bullish_count >= 3:
        technical_score += 25  # Full confluence bonus
    elif bullish_count == 2:
        technical_score += 10  # Partial confluence
    elif bearish_count >= 3:
        technical_score -= 25  # Full bearish confluence
    elif bearish_count == 2:
        technical_score -= 10

    # Trap detection from old MTF logic (still useful)
    if trends_15m == "Bullish" and trends_1h == "Bearish" and trends_4h == "Bearish":
        technical_score -= 20  # Bull Trap — lower TF against higher TFs
    elif trends_15m == "Bearish" and trends_1h == "Bullish" and trends_4h == "Bullish":
        technical_score += 20  # Bear Trap — pullback in uptrend
        
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

    # VIX data for frontend
    vix_current = vix_data.get('current', 0) if vix_data else 0
    vix_change = vix_data.get('pct_change', 0) if vix_data else 0

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
        "recent_headlines": headlines[:5],  # Expanded from 3 to 5
        "event_warning": event_warning,
        "fii_net": fii_net,
        "dii_net": dii_net,

        # NEW fields
        "rsi_divergence": rsi_divergence,
        "stoch_rsi_k": stoch_k,
        "stoch_rsi_d": stoch_d,
        "supertrend": tech_data.get('supertrend'),
        "supertrend_direction": tech_data.get('supertrend_direction', 'Neutral'),
        "volume_surge": tech_data.get('volume_surge', False),
        "volume_ratio": tech_data.get('volume_ratio', 1.0),
        "iv_crush_warning": iv_crush_warning,
        "atm_ce_ltp": atm_ce_ltp,
        "atm_pe_ltp": atm_pe_ltp,
        "vix": vix_current,
        "vix_change": round(vix_change, 2),
        "mtf_confluence": f"{bullish_count}B/{bearish_count}S (15m:{trends_15m} 1h:{trends_1h} 4h:{trends_4h})",
    }

if __name__ == "__main__":
    import json
    print(json.dumps(generate_signals(), indent=2))
