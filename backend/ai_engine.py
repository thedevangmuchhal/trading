import re
import math
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from data_fetcher import fetch_market_data, fetch_news

analyzer = SentimentIntensityAnalyzer()

# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────
def analyze_sentiment(headlines):
    if not headlines:
        return 0
    total = sum(analyzer.polarity_scores(t)['compound'] for t in headlines)
    return round((total / len(headlines)) * 100, 2)

# ─────────────────────────────────────────────────────────────────────────────
# CORE INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def calculate_rsi(data, window=14):
    delta = data.diff()
    up   = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up   = up.ewm(com=window - 1, adjust=False).mean()
    ema_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def calculate_atr(df, window=14):
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calculate_macd(data):
    ema12  = data.ewm(span=12, adjust=False).mean()
    ema26  = data.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return macd, signal, hist

def calculate_adx(df, window=14):
    hd = df['High'].diff()
    ld = df['Low'].diff()
    pos_dm = pd.Series(np.where((hd > ld) & (hd > 0), hd, 0.0), index=df.index)
    neg_dm = pd.Series(np.where((ld > hd) & (ld > 0), ld, 0.0), index=df.index)
    pos_dm = pos_dm.ewm(alpha=1 / window, adjust=False).mean()
    neg_dm = neg_dm.ewm(alpha=1 / window, adjust=False).mean()
    tr     = calculate_atr(df, window)
    pos_di = 100 * (pos_dm / tr)
    neg_di = 100 * (neg_dm / tr)
    dx     = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10))
    return dx.ewm(alpha=1 / window, adjust=False).mean()

def calculate_vwap(df):
    if df['Volume'].sum() == 0:
        return pd.Series(index=df.index, dtype=float)
    q = df['Volume']
    p = (df['High'] + df['Low'] + df['Close']) / 3
    return (p * q).cumsum() / q.cumsum()

def calculate_stochastic_rsi(rsi_series, period=14, smooth_k=3, smooth_d=3):
    rsi_min  = rsi_series.rolling(window=period).min()
    rsi_max  = rsi_series.rolling(window=period).max()
    stoch    = (rsi_series - rsi_min) / (rsi_max - rsi_min + 1e-10)
    k_line   = stoch.rolling(window=smooth_k).mean() * 100
    d_line   = k_line.rolling(window=smooth_d).mean()
    return k_line, d_line

# ─────────────────────────────────────────────────────────────────────────────
# FIXED SUPERTREND  — numpy-based, no pandas chained-assignment bug
# ─────────────────────────────────────────────────────────────────────────────
def calculate_supertrend(df, period=10, multiplier=3.0):
    hl2  = (df['High'] + df['Low']) / 2
    hl   = df['High'] - df['Low']
    hc   = (df['High'] - df['Close'].shift()).abs()
    lc   = (df['Low']  - df['Close'].shift()).abs()
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr  = tr.ewm(span=period, adjust=False).mean()

    basic_upper = (hl2 + multiplier * atr).values
    basic_lower = (hl2 - multiplier * atr).values
    close = df['Close'].values
    n = len(df)

    fu = np.zeros(n); fl = np.zeros(n)
    st = np.zeros(n); di = np.ones(n, dtype=int)

    fu[0] = basic_upper[0]; fl[0] = basic_lower[0]
    st[0] = basic_upper[0]; di[0] = -1

    for i in range(1, n):
        fu[i] = basic_upper[i] if (basic_upper[i] < fu[i-1] or close[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = basic_lower[i] if (basic_lower[i] > fl[i-1] or close[i-1] < fl[i-1]) else fl[i-1]

        if st[i-1] == fu[i-1]:
            if close[i] > fu[i]:  di[i] = 1;  st[i] = fl[i]
            else:                  di[i] = -1; st[i] = fu[i]
        else:
            if close[i] < fl[i]:  di[i] = -1; st[i] = fu[i]
            else:                  di[i] = 1;  st[i] = fl[i]

    return pd.Series(st, index=df.index), pd.Series(di, index=df.index)

# ─────────────────────────────────────────────────────────────────────────────
# RSI DIVERGENCE — Swing-Pivot Based (proper lower-low / higher-high detection)
# ─────────────────────────────────────────────────────────────────────────────
def detect_rsi_divergence(df, rsi_series, lookback=20, order=3):
    """Detects RSI divergence using swing pivot highs/lows.
    
    Bullish: Price makes lower low, RSI makes higher low.
    Bearish: Price makes higher high, RSI makes lower high.
    
    Uses scipy.signal.argrelextrema for clean pivot detection.
    Falls back to slope-based method if scipy unavailable.
    """
    if len(df) < lookback + order or len(rsi_series) < lookback + order:
        return 'none'
    
    try:
        from scipy.signal import argrelextrema
    except ImportError:
        # Fallback: simplified slope-based method
        return _rsi_divergence_slope_fallback(df, rsi_series, lookback)
    
    price_arr = df['Close'].iloc[-lookback:].values
    rsi_arr = rsi_series.iloc[-lookback:].values
    
    if any(np.isnan(price_arr)) or any(np.isnan(rsi_arr)):
        return 'none'
    
    # Find swing lows (local minima)
    low_idxs = argrelextrema(price_arr, np.less_equal, order=order)[0]
    # Find swing highs (local maxima)
    high_idxs = argrelextrema(price_arr, np.greater_equal, order=order)[0]
    
    # Bullish divergence: need at least 2 swing lows
    if len(low_idxs) >= 2:
        # Take the last two swing lows
        i1, i2 = low_idxs[-2], low_idxs[-1]
        # Price: lower low (or equal), RSI: higher low
        if price_arr[i2] <= price_arr[i1] and rsi_arr[i2] > rsi_arr[i1]:
            return 'bullish'
    
    # Bearish divergence: need at least 2 swing highs
    if len(high_idxs) >= 2:
        i1, i2 = high_idxs[-2], high_idxs[-1]
        # Price: higher high (or equal), RSI: lower high
        if price_arr[i2] >= price_arr[i1] and rsi_arr[i2] < rsi_arr[i1]:
            return 'bearish'
    
    return 'none'


def _rsi_divergence_slope_fallback(df, rsi_series, lookback=5):
    """Simplified slope-based RSI divergence (used if scipy unavailable)."""
    if len(df) < lookback + 1 or len(rsi_series) < lookback + 1:
        return 'none'
    rp = df['Close'].iloc[-lookback:]; rr = rsi_series.iloc[-lookback:]
    if rp.isna().any() or rr.isna().any():
        return 'none'
    ps = rp.iloc[-1] - rp.iloc[0]; rs = rr.iloc[-1] - rr.iloc[0]
    if ps < 0 and rs > 0:
        return 'bullish'
    elif ps > 0 and rs < 0:
        return 'bearish'
    return 'none'

# ─────────────────────────────────────────────────────────────────────────────
# VOLUME SURGE
# ─────────────────────────────────────────────────────────────────────────────
def detect_volume_surge(df, avg_period=20, threshold=2.0):
    if len(df) < avg_period + 1 or 'Volume' not in df.columns:
        return False, 1.0
    avg = df['Volume'].iloc[-(avg_period+1):-1].mean()
    if avg <= 0:
        return False, 1.0
    cur = df['Volume'].iloc[-1]
    ratio = cur / avg
    return ratio >= threshold, round(ratio, 2)

# ─────────────────────────────────────────────────────────────────────────────
# EMA STACK
# ─────────────────────────────────────────────────────────────────────────────
def calculate_ema_stack(df):
    e9   = df['Close'].ewm(span=9,   adjust=False).mean().iloc[-1]
    e21  = df['Close'].ewm(span=21,  adjust=False).mean().iloc[-1]
    e50  = df['Close'].ewm(span=50,  adjust=False).mean().iloc[-1]
    e200 = df['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
    if   e9 > e21 > e50 > e200: alignment = "Strong Bullish"
    elif e9 > e21 > e50:        alignment = "Bullish"
    elif e9 < e21 < e50 < e200: alignment = "Strong Bearish"
    elif e9 < e21 < e50:        alignment = "Bearish"
    elif e9 > e21:              alignment = "Mildly Bullish"
    else:                       alignment = "Neutral"
    return {
        "ema9":  round(e9, 2),  "ema21": round(e21, 2),
        "ema50": round(e50, 2), "ema200": round(e200, 2),
        "alignment": alignment
    }

# ─────────────────────────────────────────────────────────────────────────────
# FIBONACCI RETRACEMENT
# ─────────────────────────────────────────────────────────────────────────────
def calculate_fibonacci_levels(df, lookback=50):
    recent = df.tail(lookback)
    high = recent['High'].max()
    low  = recent['Low'].min()
    rng  = high - low
    if rng < 1:
        return {}
    return {
        "swing_high": round(high,              2),
        "fib_786":    round(low + rng * 0.786, 2),
        "fib_618":    round(low + rng * 0.618, 2),
        "fib_500":    round(low + rng * 0.500, 2),
        "fib_382":    round(low + rng * 0.382, 2),
        "fib_236":    round(low + rng * 0.236, 2),
        "swing_low":  round(low,               2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# CAMARILLA PIVOTS
# ─────────────────────────────────────────────────────────────────────────────
def calculate_camarilla(pdh, pdl, pdc):
    hl = pdh - pdl
    r  = 1.1
    return {
        "h4": round(pdc + hl * (r / 4),  2),
        "h3": round(pdc + hl * (r / 6),  2),
        "h2": round(pdc + hl * (r / 12), 2),
        "h1": round(pdc + hl * (r / 24), 2),
        "l1": round(pdc - hl * (r / 24), 2),
        "l2": round(pdc - hl * (r / 12), 2),
        "l3": round(pdc - hl * (r / 6),  2),
        "l4": round(pdc - hl * (r / 4),  2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# PREVIOUS-DAY LEVELS
# ─────────────────────────────────────────────────────────────────────────────
def get_prev_day_levels(df):
    try:
        daily = df.resample('D').agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        ).dropna()
        if len(daily) >= 2:
            prev = daily.iloc[-2]
            return {"pdh": round(prev['High'], 2),
                    "pdl": round(prev['Low'],  2),
                    "pdc": round(prev['Close'], 2)}
    except Exception:
        pass
    return {"pdh": None, "pdl": None, "pdc": None}

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED CANDLESTICK PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
def detect_advanced_patterns(df):
    if len(df) < 3:
        return "None"
    c  = df.iloc[-1]; p1 = df.iloc[-2]; p2 = df.iloc[-3]

    body   = abs(c['Close']  - c['Open'])
    body1  = abs(p1['Close'] - p1['Open'])
    body2  = abs(p2['Close'] - p2['Open'])
    fr     = c['High']  - c['Low']
    uw     = c['High']  - max(c['Close'],  c['Open'])
    lw     = min(c['Close'],  c['Open'])  - c['Low']
    uw1    = p1['High'] - max(p1['Close'], p1['Open'])
    lw1    = min(p1['Close'], p1['Open']) - p1['Low']

    if fr > 0 and body < 0.1 * fr:
        return "Doji"
    if lw > 2 * body and uw < 0.3 * body:
        return "Hammer" if c['Close'] >= c['Open'] else "Hanging Man"
    if uw > 2 * body and lw < 0.3 * body:
        return "Shooting Star" if c['Close'] < c['Open'] else "Inverted Hammer"
    if (p1['Close'] < p1['Open'] and c['Close'] > c['Open'] and
            c['Close'] > p1['Open'] and c['Open'] < p1['Close']):
        return "Bullish Engulfing"
    if (p1['Close'] > p1['Open'] and c['Close'] < c['Open'] and
            c['Close'] < p1['Open'] and c['Open'] > p1['Close']):
        return "Bearish Engulfing"
    if (p2['Close'] < p2['Open'] and body2 > body1 * 1.5 and
            abs(p1['Close'] - p1['Open']) < 0.3 * body2 and
            c['Close'] > c['Open'] and c['Close'] > (p2['Open'] + p2['Close']) / 2):
        return "Morning Star"
    if (p2['Close'] > p2['Open'] and body2 > body1 * 1.5 and
            abs(p1['Close'] - p1['Open']) < 0.3 * body2 and
            c['Close'] < c['Open'] and c['Close'] < (p2['Open'] + p2['Close']) / 2):
        return "Evening Star"
    if c['High'] < p1['High'] and c['Low'] > p1['Low']:
        return "Inside Bar"
    if uw > 3 * body and lw < 0.5 * body:
        return "Bearish Pin Bar"
    if lw > 3 * body and uw < 0.5 * body:
        return "Bullish Pin Bar"
    if (c['Close'] > c['Open'] and p1['Close'] > p1['Open'] and p2['Close'] > p2['Open'] and
            c['Close'] > p1['Close'] > p2['Close'] and c['Open'] > p1['Open']):
        return "Three White Soldiers"
    if (c['Close'] < c['Open'] and p1['Close'] < p1['Open'] and p2['Close'] < p2['Open'] and
            c['Close'] < p1['Close'] < p2['Close'] and c['Open'] < p1['Open']):
        return "Three Black Crows"
    return "None"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TECHNICAL ANALYSIS (for a given timeframe dataframe)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_technicals(df):
    if df.empty or len(df) < 30:
        return {}

    df = df.copy()
    df['EMA_9']   = df['Close'].ewm(span=9,   adjust=False).mean()
    df['EMA_21']  = df['Close'].ewm(span=21,  adjust=False).mean()
    df['EMA_50']  = df['Close'].ewm(span=50,  adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    df['BB_Mid']   = df['Close'].rolling(20).mean()
    bb_std         = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std

    df['ATR']  = calculate_atr(df, 14)
    df['RSI']  = calculate_rsi(df['Close'], 14)
    macd, sig, hist = calculate_macd(df['Close'])
    df['MACD'] = macd; df['MACD_Signal'] = sig; df['MACD_Hist'] = hist
    df['ADX']  = calculate_adx(df, 14)
    df['VWAP'] = calculate_vwap(df)

    sk, sd = calculate_stochastic_rsi(df['RSI'])
    df['StochRSI_K'] = sk; df['StochRSI_D'] = sd

    st_line, st_dir = calculate_supertrend(df)
    df['Supertrend'] = st_line; df['Supertrend_Dir'] = st_dir

    rsi_div   = detect_rsi_divergence(df, df['RSI'])
    vol_surge, vol_ratio = detect_volume_surge(df)

    lat   = df.iloc[-1]
    price = lat['Close']

    trend    = "Bullish" if lat['EMA_9'] > lat['EMA_21'] else ("Bearish" if lat['EMA_9'] < lat['EMA_21'] else "Neutral")
    momentum = ("Strong Bullish" if lat['MACD_Hist'] > 0 and lat['MACD'] > lat['MACD_Signal']
                 else ("Strong Bearish" if lat['MACD_Hist'] < 0 and lat['MACD'] < lat['MACD_Signal'] else "Neutral"))

    pattern   = detect_advanced_patterns(df)
    ema_stack = calculate_ema_stack(df)
    vwap_val  = lat['VWAP']
    vwap_cross = "Above" if price > vwap_val else "Below"

    # ORB
    orb_breakout = "None"; or_high = None; or_low = None
    try:
        today    = df.index[-1].date()
        today_df = df[df.index.date == today]
        or_df    = today_df.between_time("09:15", "10:00", inclusive="left")
        if not or_df.empty:
            or_high = or_df['High'].max(); or_low = or_df['Low'].min()
            if lat.name.time() >= pd.to_datetime("10:00").time():
                if   price > or_high: orb_breakout = "Bullish"
                elif price < or_low:  orb_breakout = "Bearish"
    except Exception:
        pass

    h = df['High'].max(); l = df['Low'].min()
    pivot = (h + l + price) / 3
    r1 = (2 * pivot) - l
    s1 = (2 * pivot) - h

    def _v(x): return round(x, 2) if x is not None and not pd.isna(x) else None

    return {
        "current_price":      round(price, 2),
        "trend":              trend,
        "momentum":           momentum,
        "chart_pattern":      pattern,
        "orb_breakout":       orb_breakout,
        "or_high":            or_high, "or_low": or_low,
        "ema_9":  _v(lat['EMA_9']),   "ema_21":  _v(lat['EMA_21']),
        "ema_50": _v(lat['EMA_50']),  "ema_200": _v(lat['EMA_200']),
        "bb_upper": _v(lat['BB_Upper']), "bb_lower": _v(lat['BB_Lower']),
        "rsi":  _v(lat['RSI'])  or 50,
        "atr":  _v(lat['ATR'])  or 10,
        "adx":  _v(lat['ADX'])  or 0,
        "macd": _v(lat['MACD']) or 0,
        "macd_signal": _v(lat['MACD_Signal']) or 0,
        "vwap":  _v(vwap_val),
        "vwap_cross": vwap_cross,
        "support":    round(s1, 2), "resistance": round(r1, 2),
        "stoch_rsi_k": _v(lat['StochRSI_K']) or 50,
        "stoch_rsi_d": _v(lat['StochRSI_D']) or 50,
        "supertrend": _v(lat['Supertrend']),
        "supertrend_direction": "Bullish" if lat['Supertrend_Dir'] == 1 else "Bearish",
        "rsi_divergence": rsi_div,
        "volume_surge":   vol_surge, "volume_ratio": vol_ratio,
        "ema_stack": ema_stack,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIGNAL GENERATOR  — 5-Timeframe Edition
# ─────────────────────────────────────────────────────────────────────────────
def generate_signals(ticker="^NSEI"):
    from data_fetcher import (fetch_advanced_oi, fetch_fii_dii,
                               fetch_market_data, fetch_news, fetch_vix,
                               calculate_vpvr)
    from ml_engine import predict_breakout_probability

    # ── 1. Fetch all 5 timeframes ──────────────────────────────────────────
    df_5m  = fetch_market_data(ticker, interval="5m",  period="2d")
    df_15m = fetch_market_data(ticker, interval="15m", period="5d")
    df_30m = fetch_market_data(ticker, interval="30m", period="30d")
    df_1h  = fetch_market_data(ticker, interval="1h",  period="60d")
    df_4h_raw = fetch_market_data(ticker, interval="1h", period="90d")

    headlines     = fetch_news(ticker)
    sentiment_val = analyze_sentiment(headlines)

    # Primary: 15m
    tech_data = analyze_technicals(df_15m)
    if not tech_data:
        return {"error": "Failed to fetch market data"}

    current_price = tech_data['current_price']

    # Analyze each TF
    tech_5m  = analyze_technicals(df_5m)  if not df_5m.empty  else {}
    tech_30m = analyze_technicals(df_30m) if not df_30m.empty else {}
    tech_1h  = analyze_technicals(df_1h)  if not df_1h.empty  else {}

    tech_4h = {}
    if not df_4h_raw.empty and len(df_4h_raw) >= 20:
        try:
            df_4h = df_4h_raw.resample('4h').agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            ).dropna()
            if not df_4h.empty:
                tech_4h = analyze_technicals(df_4h)
        except Exception:
            pass

    # TF trends
    t5m  = tech_5m.get('trend',  'Neutral')
    t15m = tech_data.get('trend', 'Neutral')
    t30m = tech_30m.get('trend', 'Neutral')
    t1h  = tech_1h.get('trend',  'Neutral')
    t4h  = tech_4h.get('trend',  'Neutral')

    all_trends    = [t5m, t15m, t30m, t1h, t4h]
    bullish_count = sum(1 for t in all_trends if t == "Bullish")
    bearish_count = sum(1 for t in all_trends if t == "Bearish")

    # ── 2. Market data helpers ─────────────────────────────────────────────
    poc        = calculate_vpvr(df_15m)
    oi_metrics = fetch_advanced_oi(ticker, current_price)
    vix_data   = fetch_vix()

    # PDH / PDL / Camarilla
    prev       = get_prev_day_levels(df_15m)
    pdh, pdl, pdc = prev.get('pdh'), prev.get('pdl'), prev.get('pdc')
    camarilla  = calculate_camarilla(pdh, pdl, pdc) if pdh and pdl and pdc else {}

    # Fibonacci
    fibonacci  = calculate_fibonacci_levels(df_15m)

    # Weekly levels
    weekly_high = weekly_low = None
    if not df_1h.empty:
        try:
            lw = df_1h.last('5D')
            weekly_high = round(lw['High'].max(), 2)
            weekly_low  = round(lw['Low'].min(),  2)
        except Exception:
            pass

    # ── 3. Score system ───────────────────────────────────────────────────
    smart_money_score = 50; options_score = 50
    technical_score   = 50; sentiment_score = 50

    # Smart Money (FII / DII)
    fii_dii_data = fetch_fii_dii()
    fii_net = 0; dii_net = 0
    if fii_dii_data:
        for item in fii_dii_data:
            try:
                cat = item.get("category", "")
                val = float(item.get("netValue", 0))
                if   cat == "FII/FPI": fii_net = val
                elif cat == "DII":     dii_net = val
            except:
                pass
        if fii_net > 0 and dii_net > 0:   smart_money_score += 25
        elif fii_net < 0 and dii_net < 0: smart_money_score -= 25
        if   fii_net >  5000: smart_money_score += 15
        elif fii_net < -5000: smart_money_score -= 15
        if   dii_net >  5000: smart_money_score += 10
        elif dii_net < -5000: smart_money_score -= 10

    # Sentiment
    event_warning = False
    event_phrases = ["rbi policy","rbi rate","rbi decision","monetary policy",
                     "fed rate","fed decision","fomc","union budget",
                     "cpi data","inflation surge","election result",
                     "circuit breaker","black swan","panic sell"]
    single_words  = ["crash","war","fomc","conflict","missile","attack"]
    for hl in headlines:
        hlow = hl.lower()
        hclean = re.sub(r'[^\w\s]', ' ', hlow)
        if any(p in hlow for p in event_phrases) or any(f" {w} " in f" {hclean} " for w in single_words):
            event_warning = True; break
    sentiment_score += sentiment_val * 0.4

    iv_crush_warning = False
    if vix_data:
        pct = vix_data.get('pct_change', 0)
        if   pct < -5.0: iv_crush_warning = True; sentiment_score -= 20
        elif pct < -3.0: sentiment_score -= 15
        elif pct >  3.0: sentiment_score += 10

    # Options
    pcr = None; vpcr = None; atm_ce_ltp = 0; atm_pe_ltp = 0
    if oi_metrics:
        pcr = oi_metrics.get('pcr')
        atm_ce_ltp = oi_metrics.get('atm_ce_ltp', 0)
        atm_pe_ltp = oi_metrics.get('atm_pe_ltp', 0)
        if pcr is not None:
            if   pcr > 1.2: options_score += 25
            elif pcr > 1.0: options_score += 10
            elif pcr < 0.6: options_score -= 25
            elif pcr < 0.8: options_score -= 10
        mp = oi_metrics.get('max_pain', 0)
        if mp > 0:
            if   current_price < (mp - tech_data['atr']): options_score += 15
            elif current_price > (mp + tech_data['atr']): options_score -= 15
        tv_ce = oi_metrics.get('total_ce_vol', 0)
        tv_pe = oi_metrics.get('total_pe_vol', 0)
        if tv_ce > 0:
            vpcr = round(tv_pe / tv_ce, 2)
            if   vpcr > 1.2: options_score += 15
            elif vpcr < 0.8: options_score -= 15
        bu = oi_metrics.get('buildup', {})
        cc = bu.get('ce_change', 0); pc = bu.get('pe_change', 0)
        if   cc > pc * 1.5 and cc > 0: options_score -= 15
        elif pc > cc * 1.5 and pc > 0: options_score += 15
        gr = oi_metrics.get('greeks', {})
        iv_skew = gr.get('ce_iv', 0) - gr.get('pe_iv', 0)
        if   iv_skew >  3: options_score += 15
        elif iv_skew < -3: options_score -= 15

    # ── Technical score ────────────────────────────────────────────────────
    if   tech_data['trend'] == "Bullish":      technical_score += 15
    elif tech_data['trend'] == "Bearish":      technical_score -= 15
    if   tech_data['momentum'] == "Strong Bullish": technical_score += 10
    elif tech_data['momentum'] == "Strong Bearish": technical_score -= 10

    if tech_data['bb_lower'] and current_price < tech_data['bb_lower']: technical_score += 10
    if tech_data['bb_upper'] and current_price > tech_data['bb_upper']: technical_score -= 10

    if poc:
        if current_price > poc: technical_score += 10
        else:                    technical_score -= 10

    if tech_data['vwap']:
        if current_price > tech_data['vwap']: technical_score += 10
        else:                                  technical_score -= 10

    BULL_PATTERNS = {"Bullish Engulfing","Hammer","Morning Star","Bullish Pin Bar","Three White Soldiers","Inverted Hammer"}
    BEAR_PATTERNS = {"Bearish Engulfing","Shooting Star","Evening Star","Bearish Pin Bar","Three Black Crows","Hanging Man"}
    pat = tech_data.get('chart_pattern', 'None')
    if pat in BULL_PATTERNS:  technical_score += 15
    elif pat in BEAR_PATTERNS: technical_score -= 15

    if   tech_data['orb_breakout'] == "Bullish": technical_score += 30
    elif tech_data['orb_breakout'] == "Bearish": technical_score -= 30

    ml_prob = predict_breakout_probability(tech_data)
    if ml_prob is not None:
        if   ml_prob > 0.75: technical_score += 20
        elif ml_prob > 0.60: technical_score += 10
        elif ml_prob < 0.25: technical_score -= 20
        elif ml_prob < 0.40: technical_score -= 10

    if   tech_data.get('supertrend_direction') == "Bullish": technical_score += 15
    elif tech_data.get('supertrend_direction') == "Bearish": technical_score -= 15

    rsi_div = tech_data.get('rsi_divergence', 'none')
    if   rsi_div == 'bullish': technical_score += 15
    elif rsi_div == 'bearish': technical_score -= 15

    stoch_k = tech_data.get('stoch_rsi_k', 50); stoch_d = tech_data.get('stoch_rsi_d', 50)
    if   stoch_k < 20 and stoch_k > stoch_d: technical_score += 10
    elif stoch_k > 80 and stoch_k < stoch_d: technical_score -= 10
    if   stoch_k < 5:  technical_score += 10
    elif stoch_k > 95: technical_score -= 10

    if tech_data.get('volume_surge', False):
        if   tech_data['trend'] == "Bullish": technical_score += 10
        elif tech_data['trend'] == "Bearish": technical_score -= 10

    ea = tech_data.get('ema_stack', {}).get('alignment', 'Neutral')
    if   ea == "Strong Bullish": technical_score += 15
    elif ea == "Bullish":        technical_score += 8
    elif ea == "Strong Bearish": technical_score -= 15
    elif ea == "Bearish":        technical_score -= 8

    if pdh and current_price > pdh: technical_score += 15
    if pdl and current_price < pdl: technical_score -= 15

    if tech_data.get('vwap_cross') == "Above": technical_score += 5
    else:                                        technical_score -= 5

    # 5-TF confluence
    if   bullish_count == 5: technical_score += 30
    elif bullish_count == 4: technical_score += 20
    elif bullish_count == 3: technical_score += 10
    elif bearish_count == 5: technical_score -= 30
    elif bearish_count == 4: technical_score -= 20
    elif bearish_count == 3: technical_score -= 10

    # Trap detection
    if t15m == "Bullish" and t1h == "Bearish" and t4h == "Bearish": technical_score -= 20
    elif t15m == "Bearish" and t1h == "Bullish" and t4h == "Bullish": technical_score += 20

    rsi_val = tech_data.get('rsi', 50)
    if rsi_val < 35 and bullish_count >= 3: technical_score += 15
    elif rsi_val > 65 and bearish_count >= 3: technical_score -= 15
    if rsi_val < 25 and stoch_k < 10 and bullish_count >= 3: technical_score += 20
    elif rsi_val > 75 and stoch_k > 90 and bearish_count >= 3: technical_score -= 20

    market_condition = "Trending"
    adx_val = tech_data['adx']
    if adx_val < 20:
        technical_score  -= 30
        market_condition = "Choppy/Sideways"

    # ── Clamp all scores ──────────────────────────────────────────────────
    smart_money_score = int(max(0, min(100, smart_money_score)))
    options_score     = int(max(0, min(100, options_score)))
    technical_score   = int(max(0, min(100, technical_score)))
    sentiment_score   = int(max(0, min(100, sentiment_score)))

    # ══════════════════════════════════════════════════════════════════════
    # IMPROVEMENT #1: ADAPTIVE PILLAR WEIGHTS
    # ══════════════════════════════════════════════════════════════════════
    has_oi = oi_metrics is not None and oi_metrics.get('pcr') is not None
    dte = oi_metrics.get('days_to_expiry', 5) if oi_metrics else 5
    is_expiry_day = dte <= 1

    if not has_oi:
        # No OI data — rely on technicals + smart money
        w_tech, w_opt, w_sm, w_sent = 0.45, 0.0, 0.30, 0.25
    elif is_expiry_day:
        # Expiry day — options flow dominates (gamma/theta effects)
        w_tech, w_opt, w_sm, w_sent = 0.20, 0.45, 0.20, 0.15
    elif adx_val >= 25:
        # Trending market — technicals dominate
        w_tech, w_opt, w_sm, w_sent = 0.40, 0.25, 0.25, 0.10
    elif adx_val < 20:
        # Choppy market — options flow + smart money more reliable
        w_tech, w_opt, w_sm, w_sent = 0.20, 0.35, 0.30, 0.15
    else:
        # Normal market (ADX 20-25)
        w_tech, w_opt, w_sm, w_sent = 0.30, 0.28, 0.27, 0.15

    # ══════════════════════════════════════════════════════════════════════
    # IMPROVEMENT #5: TIME-OF-DAY BIAS
    # ══════════════════════════════════════════════════════════════════════
    from datetime import datetime, timedelta
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        ist_hour = ist_now.hour
        ist_min = ist_now.minute
        ist_total_mins = ist_hour * 60 + ist_min
    except:
        ist_total_mins = 720  # Noon default

    time_of_day_penalty = 0
    orb_suppressed = False

    if 555 <= ist_total_mins < 585:
        # 9:15-9:45 IST — Opening noise, unreliable signals
        time_of_day_penalty = -15
        orb_suppressed = True  # ORB signal is noise in first 30 min
    elif 900 <= ist_total_mins <= 930:
        # 15:00-15:30 IST — Closing session, no new entries
        time_of_day_penalty = -20
    elif 840 <= ist_total_mins < 900:
        # 14:00-15:00 IST — Institutional hour, boost smart money
        w_sm = min(1.0, w_sm + 0.10)
        # Re-normalize weights
        w_total = w_tech + w_opt + w_sm + w_sent
        w_tech /= w_total; w_opt /= w_total; w_sm /= w_total; w_sent /= w_total

    # Suppress ORB contribution if in opening noise window
    if orb_suppressed and tech_data['orb_breakout'] != "None":
        # Undo the ORB score that was already added (±30)
        if tech_data['orb_breakout'] == "Bullish":
            technical_score -= 30
        elif tech_data['orb_breakout'] == "Bearish":
            technical_score += 30
        technical_score = int(max(0, min(100, technical_score)))

    # Compute weighted confidence
    confidence = int(
        technical_score * w_tech +
        options_score * w_opt +
        smart_money_score * w_sm +
        sentiment_score * w_sent
    )

    # Apply time-of-day penalty
    confidence = int(max(0, min(100, confidence + time_of_day_penalty)))

    # ══════════════════════════════════════════════════════════════════════
    # IMPROVEMENT #2: EXPIRY DAY SPECIAL LOGIC
    # ══════════════════════════════════════════════════════════════════════
    expiry_blocked = False
    if is_expiry_day and has_oi:
        max_pain = oi_metrics.get('max_pain', 0)
        # Gamma pin: if price is within 0.5% of Max Pain → strong WAIT bias
        if max_pain and abs(current_price - max_pain) / max_pain < 0.005:
            confidence = max(35, min(65, confidence))  # Clamp to WAIT zone
            expiry_blocked = True

        # Theta decay penalty after 1:30 PM
        if ist_total_mins >= 810:  # 13:30 IST
            confidence = int(confidence * 0.90)  # 10% penalty

        # Volume confirmation required on expiry
        if not tech_data.get('volume_surge', False):
            confidence = max(35, min(65, confidence))
            expiry_blocked = True

    # ══════════════════════════════════════════════════════════════════════
    # IMPROVEMENT #3: CAPITAL PRESERVATION RULES
    # ══════════════════════════════════════════════════════════════════════

    # --- Pillar Agreement Gate ---
    # Count how many pillars agree on direction
    bullish_pillars = sum([
        smart_money_score > 60,
        options_score > 60 if has_oi else True,  # Skip if no OI
        technical_score > 60,
        sentiment_score > 55,
    ])
    bearish_pillars = sum([
        smart_money_score < 40,
        options_score < 40 if has_oi else True,
        technical_score < 40,
        sentiment_score < 45,
    ])

    pillar_agreement = max(bullish_pillars, bearish_pillars)

    # --- Choppy Market Hard Block ---
    choppy_blocked = adx_val < 15  # No trend at all → no trade

    # ── Action logic (with preservation gates) ────────────────────────────
    action = "WAIT"; strike_type = None

    # Dynamic thresholds based on context
    buy_threshold = 80 if is_expiry_day else 70
    sell_threshold = 20 if is_expiry_day else 30

    if smart_money_score >= 80 and (options_score >= 60 or not has_oi):
        action = "BUY"; strike_type = "CE"; confidence = max(confidence, 80)
    elif smart_money_score <= 20 and (options_score <= 40 or not has_oi):
        action = "SELL"; strike_type = "PE"; confidence = min(confidence, 20)
    elif technical_score >= 75 and (options_score >= 55 or not has_oi):
        action = "BUY"; strike_type = "CE"; confidence = max(confidence, 75)
    elif technical_score <= 25 and (options_score <= 45 or not has_oi):
        action = "SELL"; strike_type = "PE"; confidence = min(confidence, 25)
    elif rsi_val < 35 and stoch_k < 20 and bullish_count >= 3 and smart_money_score >= 60:
        action = "BUY"; strike_type = "CE"; confidence = max(confidence, 65)
    elif rsi_val > 65 and stoch_k > 80 and bearish_count >= 3 and smart_money_score <= 40:
        action = "SELL"; strike_type = "PE"; confidence = min(confidence, 35)
    elif confidence >= buy_threshold:
        action = "BUY"; strike_type = "CE"
    elif confidence <= sell_threshold:
        action = "SELL"; strike_type = "PE"

    # ── GOLDEN SNIPER STRATEGY (High Win Rate / Reversal) ─────────────────
    strategy_type = "NORMAL"
    in_chop_zone = 660 <= ist_total_mins <= 810
    has_surge = tech_data.get('volume_surge', False)
    bb_lower = tech_data.get('bb_lower')
    bb_upper = tech_data.get('bb_upper')
    
    # Sniper BUY (Bounce off lower band + MTF aligned + surge)
    if not in_chop_zone and has_surge and bb_lower and current_price <= bb_lower * 1.002 and rsi_val < 35 and bullish_count >= 3:
        if not has_oi or options_score >= 40: # Not heavily bearish options chain
            action = "BUY"; strike_type = "CE"; confidence = 95
            strategy_type = "SNIPER"
            
    # Sniper SELL
    elif not in_chop_zone and has_surge and bb_upper and current_price >= bb_upper * 0.998 and rsi_val > 65 and bearish_count >= 3:
        if not has_oi or options_score <= 60: 
            action = "SELL"; strike_type = "PE"; confidence = 5
            strategy_type = "SNIPER"

    if action != "WAIT":
        if strategy_type == "NORMAL":
            # Gate 1: Pillar Agreement — at least 3 of 4 must agree
            if pillar_agreement < 3:
                action = "WAIT"
                strike_type = None

            # Gate 2: Choppy Market Hard Block (ADX < 15)
            elif choppy_blocked:
                action = "WAIT"
                strike_type = None

            # Gate 3: Expiry Day Blocks
            elif expiry_blocked:
                action = "WAIT"
                strike_type = None

        # Gate 4: Closing Session Block (15:00-15:30) — applies to ALL strategies
        if ist_total_mins >= 900:
            action = "WAIT"
            strike_type = None

    # ── Entry / Target / SL ───────────────────────────────────────────────
    atr = tech_data['atr']
    atm_strike = round(current_price / 100) * 100

    # ── Delta-based strike selection ──────────────────────────────────────
    strike_greeks = oi_metrics.get('strike_greeks', {}) if oi_metrics else {}
    dte = oi_metrics.get('days_to_expiry', 5) if oi_metrics else 5
    # Tighter delta for 0-1 DTE (scalping), wider for weekly
    target_delta = 0.45 if dte <= 1 else 0.40 if dte <= 3 else 0.35
    selected_strike = atm_strike
    selected_delta = 0
    selected_ltp = 0

    def find_strike_by_delta(greeks_dict, tgt_delta, opt_type):
        """Find the strike with delta closest to target."""
        best_strike = atm_strike
        best_diff = float('inf')
        best_delta = 0
        best_ltp = 0
        delta_key = f"{opt_type.lower()}_delta"
        ltp_key = f"{opt_type.lower()}_ltp"
        for s, g in greeks_dict.items():
            s_delta = abs(g.get(delta_key, 0))
            s_ltp = g.get(ltp_key, 0)
            if s_delta <= 0 or s_ltp <= 0:
                continue
            diff = abs(s_delta - tgt_delta)
            if diff < best_diff:
                best_diff = diff
                best_strike = s
                best_delta = g.get(delta_key, 0)
                best_ltp = s_ltp
        return best_strike, best_delta, best_ltp

    if action == "BUY":
        if strike_greeks:
            selected_strike, selected_delta, selected_ltp = find_strike_by_delta(strike_greeks, target_delta, "CE")
            sp = selected_strike
        else:
            sp = math.floor(current_price / 100) * 100
        entry = current_price
        
        if strategy_type == "SNIPER":
            target = entry + atr * 1.5
            stop_loss = entry - atr * 1.0
        else:
            target = (oi_metrics['highest_ce_strike']
                      if oi_metrics and oi_metrics.get('highest_ce_strike', 0) > entry
                      else entry + atr * 3.0)
            hp = oi_metrics.get('highest_pe_strike', 0) if oi_metrics else 0
            stop_loss = max(hp, entry - atr * 2.0) if hp > 0 and hp < entry else entry - atr * 1.5
            
    elif action == "SELL":
        if strike_greeks:
            selected_strike, selected_delta, selected_ltp = find_strike_by_delta(strike_greeks, target_delta, "PE")
            sp = selected_strike
        else:
            sp = math.ceil(current_price / 100) * 100
        entry = current_price
        
        if strategy_type == "SNIPER":
            target = entry - atr * 1.5
            stop_loss = entry + atr * 1.0
        else:
            hp = oi_metrics.get('highest_pe_strike', 0) if oi_metrics else 0
            target = hp if hp > 0 and hp < entry else entry - atr * 3.0
            hc = oi_metrics.get('highest_ce_strike', 0) if oi_metrics else 0
            stop_loss = min(hc, entry + atr * 2.0) if hc > entry else entry + atr * 1.5
            
    else:
        sp = atm_strike; entry = stop_loss = target = 0

    risk_reward = 0
    if action != "WAIT" and entry > 0 and stop_loss > 0 and target > 0:
        risk   = abs(entry - stop_loss)
        reward = abs(target - entry)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

    # ── Signal reasons (human-readable) ───────────────────────────────────
    reasons = []
    if strategy_type == "SNIPER" and action != "WAIT":
        reasons.append("🎯 [SNIPER] Golden Sniper Strategy Triggered")
    elif action != "WAIT":
        reasons.append("📊 [NORMAL] Standard Trend Following")

    if   bullish_count == 5: reasons.append(f"✅ 5/5 TF Bullish — Full Confluence")
    elif bullish_count == 4: reasons.append(f"✅ 4/5 TF Bullish ({t5m}·{t15m}·{t30m}·{t1h}·{t4h})")
    elif bullish_count == 3: reasons.append(f"🟡 3/5 TF Bullish")
    elif bearish_count == 5: reasons.append(f"✅ 5/5 TF Bearish — Full Confluence")
    elif bearish_count == 4: reasons.append(f"✅ 4/5 TF Bearish")
    elif bearish_count == 3: reasons.append(f"🟡 3/5 TF Bearish")

    if rsi_val < 30:  reasons.append(f"📉 RSI oversold ({rsi_val:.1f})")
    elif rsi_val > 70: reasons.append(f"📈 RSI overbought ({rsi_val:.1f})")
    if stoch_k < 20 and stoch_k > stoch_d: reasons.append(f"🔄 StochRSI oversold crossover K:{stoch_k:.1f}")
    elif stoch_k > 80 and stoch_k < stoch_d: reasons.append(f"🔄 StochRSI overbought crossover K:{stoch_k:.1f}")
    if rsi_div == 'bullish': reasons.append("📊 Bullish RSI divergence detected")
    elif rsi_div == 'bearish': reasons.append("📊 Bearish RSI divergence detected")
    if tech_data.get('supertrend_direction') == "Bullish": reasons.append("📈 Supertrend: Bullish (15m)")
    elif tech_data.get('supertrend_direction') == "Bearish": reasons.append("📉 Supertrend: Bearish (15m)")
    if tech_data.get('volume_surge'): reasons.append(f"🔊 Volume surge {tech_data['volume_ratio']}× avg")
    if pat != "None": reasons.append(f"🕯 Pattern: {pat}")
    if tech_data['orb_breakout'] == "Bullish": reasons.append("🚀 ORB: Bullish breakout above opening range")
    elif tech_data['orb_breakout'] == "Bearish": reasons.append("🔻 ORB: Bearish breakdown below opening range")
    if smart_money_score >= 75: reasons.append(f"🏦 FII/DII net buying (FII ₹{fii_net:,.0f} Cr)")
    elif smart_money_score <= 25: reasons.append(f"🏦 FII/DII net selling (FII ₹{fii_net:,.0f} Cr)")
    if pcr and pcr > 1.2: reasons.append(f"📊 PCR bullish {pcr} — put writing dominant")
    elif pcr and pcr < 0.8: reasons.append(f"📊 PCR bearish {pcr} — call writing dominant")
    if   ea == "Strong Bullish": reasons.append("📈 EMA Stack: Strong Bullish (9>21>50>200)")
    elif ea == "Bullish":        reasons.append("📈 EMA Stack: Bullish (9>21>50)")
    elif ea == "Strong Bearish": reasons.append("📉 EMA Stack: Strong Bearish (9<21<50<200)")
    elif ea == "Bearish":        reasons.append("📉 EMA Stack: Bearish")
    if pdh and current_price > pdh: reasons.append(f"🔓 PDH breakout: above ₹{pdh}")
    elif pdl and current_price < pdl: reasons.append(f"🔓 PDL breakdown: below ₹{pdl}")
    if iv_crush_warning: reasons.append(f"⚡ IV crush — VIX falling {vix_data.get('pct_change',0):.1f}%")
    if market_condition == "Choppy/Sideways": reasons.append("⚠️ ADX < 20 — choppy market, wait for trend")
    if not reasons: reasons.append("⏳ No strong confluence — waiting for a clean setup")

    # Enhanced signal strength (0-10)
    signal_strength = min(10, len([r for r in reasons if r.startswith(("✅","🚀","🏦"))]))
    if tech_data.get('volume_surge') and tech_data['trend'] != 'Neutral': signal_strength = min(10, signal_strength + 2)
    if ea in ('Strong Bullish', 'Strong Bearish'): signal_strength = min(10, signal_strength + 2)
    elif ea in ('Bullish', 'Bearish'): signal_strength = min(10, signal_strength + 1)
    if tech_data.get('vwap_cross') == 'Above' and tech_data['trend'] == 'Bullish': signal_strength = min(10, signal_strength + 1)
    elif tech_data.get('vwap_cross') == 'Below' and tech_data['trend'] == 'Bearish': signal_strength = min(10, signal_strength + 1)
    if pdh and current_price > pdh: signal_strength = min(10, signal_strength + 1)
    if pdl and current_price < pdl: signal_strength = min(10, signal_strength + 1)

    # ── Capital Preservation Reasons ──────────────────────────────────────
    if choppy_blocked: reasons.append("🛑 ADX < 15 — no trend, trades blocked")
    elif market_condition == "Choppy/Sideways": reasons.append("⚠️ ADX < 20 — choppy market, reduced weight")
    if is_expiry_day: reasons.append(f"📅 Expiry day (DTE={dte}) — tighter thresholds")
    if expiry_blocked: reasons.append("🎯 Near Max Pain / low volume — gamma pin risk")
    if orb_suppressed: reasons.append("⏰ 9:15-9:45 — ORB suppressed (opening noise)")
    if ist_total_mins >= 900 and action == "WAIT": reasons.append("🔒 15:00-15:30 — closing session, no new entries")
    if pillar_agreement < 3 and action == "WAIT":
        reasons.append(f"⚖️ Only {pillar_agreement}/4 pillars agree — need 3+ for trade")
    if 840 <= ist_total_mins < 900: reasons.append("🏦 14:00-15:00 — institutional hour, Smart Money boosted")
    # Weight regime indicator
    if adx_val >= 25: reasons.append(f"📊 Trending regime (ADX {adx_val:.0f}) — Tech weighted 40%")
    elif adx_val < 20: reasons.append(f"📊 Choppy regime (ADX {adx_val:.0f}) — Options weighted 35%")
    if not reasons: reasons.append("⏳ No strong confluence — waiting for a clean setup")

    # ── Conviction Filter (Gate 5) ────────────────────────────────────────
    # If signal strength is too low, force WAIT even if confidence threshold passed
    if action != "WAIT" and signal_strength < 4:
        action = "WAIT"
        strike_type = None
        reasons.append("🛡️ Conviction too low (strength < 4) — trade blocked")
        # Reset entry/target/SL to 0
        sp = atm_strike; entry = stop_loss = target = 0

    # Expected move range (ATR-based intraday range)
    expected_move_high = round(current_price + atr * 1.5, 2)
    expected_move_low  = round(current_price - atr * 1.5, 2)

    # ── MTF matrix for frontend ────────────────────────────────────────────
    def tf_sum(td):
        if not td: return {"trend": "N/A", "rsi": 50, "supertrend": "N/A", "stoch_k": 50}
        return {
            "trend":      td.get("trend", "Neutral"),
            "rsi":        td.get("rsi",   50),
            "supertrend": td.get("supertrend_direction", "Neutral"),
            "stoch_k":    td.get("stoch_rsi_k", 50),
        }
    mtf_matrix = {
        "5m": tf_sum(tech_5m), "15m": tf_sum(tech_data),
        "30m": tf_sum(tech_30m), "1h": tf_sum(tech_1h), "4h": tf_sum(tech_4h),
    }

    vix_now    = vix_data.get('current',    0) if vix_data else 0
    vix_change = vix_data.get('pct_change', 0) if vix_data else 0

    return {
        # ── Core ──────────────────────────────────────────────────────────
        "timestamp":    pd.Timestamp.now().isoformat(),
        "ticker":       ticker,
        "current_price": current_price,
        "sentiment_score_raw": sentiment_val,
        "tech_trend":   f"{tech_data['trend']} ({tech_data['momentum']}) [{market_condition}]",
        "market_condition": market_condition,
        "rsi":          tech_data['rsi'],
        # ── Scores ────────────────────────────────────────────────────────
        "confidence_score":   confidence,
        "smart_money_score":  smart_money_score,
        "options_score":      options_score,
        "technical_score":    technical_score,
        "sentiment_score":    int(max(0, min(100, sentiment_score))),
        # ── Options data ──────────────────────────────────────────────────
        "pcr":  pcr,  "vpcr": vpcr,
        "max_pain": oi_metrics['max_pain'] if oi_metrics else None,
        "expiry":   oi_metrics['expiry']   if oi_metrics else None,
        "oi_data":  oi_metrics['oi_data']  if oi_metrics else [],
        "buildup":  oi_metrics['buildup']  if oi_metrics else None,
        "greeks":   oi_metrics['greeks']   if oi_metrics else None,
        "total_ce_vol": oi_metrics['total_ce_vol'] if oi_metrics else None,
        "total_pe_vol": oi_metrics['total_pe_vol'] if oi_metrics else None,
        "atm_ce_ltp": atm_ce_ltp, "atm_pe_ltp": atm_pe_ltp,
        "atm_strike": atm_strike,
        # ── Action ────────────────────────────────────────────────────────
        "chart_pattern":  pat,
        "orb_breakout":   tech_data['orb_breakout'],
        "action":         action,
        "strategy_type":  strategy_type,
        "strike_recommendation": f"{sp} {strike_type}" + (f" Δ{abs(selected_delta):.2f}" if selected_delta else "") if strike_type else "None",
        "selected_option_ltp":   selected_ltp if selected_ltp > 0 else None,
        "entry_level":    round(entry, 2),
        "stop_loss":      round(stop_loss, 2),
        "target":         round(target, 2),
        "support":        tech_data['support'],
        "resistance":     tech_data['resistance'],
        "risk_reward":    risk_reward,
        # ── Headlines ─────────────────────────────────────────────────────
        "recent_headlines": headlines[:5],
        "event_warning":    event_warning,
        # ── Institutional ─────────────────────────────────────────────────
        "fii_net": fii_net, "dii_net": dii_net,
        # ── Indicators ────────────────────────────────────────────────────
        "rsi_divergence":       rsi_div,
        "stoch_rsi_k":          stoch_k,
        "stoch_rsi_d":          stoch_d,
        "supertrend":           tech_data.get('supertrend'),
        "supertrend_direction": tech_data.get('supertrend_direction', 'Neutral'),
        "volume_surge":         tech_data.get('volume_surge', False),
        "volume_ratio":         tech_data.get('volume_ratio', 1.0),
        "iv_crush_warning":     iv_crush_warning,
        "vix":        vix_now,
        "vix_change": round(vix_change, 2),
        # ── MTF ───────────────────────────────────────────────────────────
        "mtf_confluence": f"{bullish_count}B/{bearish_count}S (5m:{t5m} 15m:{t15m} 30m:{t30m} 1h:{t1h} 4h:{t4h})",
        "mtf_matrix":     mtf_matrix,
        # ── NEW advanced fields ───────────────────────────────────────────
        "fibonacci":        fibonacci,
        "pdh":              pdh,
        "pdl":              pdl,
        "pdc":              pdc,
        "weekly_high":      weekly_high,
        "weekly_low":       weekly_low,
        "camarilla":        camarilla,
        "ema_stack":        tech_data.get('ema_stack', {}),
        "vwap":             tech_data.get('vwap'),
        "vwap_cross":       tech_data.get('vwap_cross'),
        "signal_reasons":   reasons,
        "signal_strength":  signal_strength,
        "expected_move_high": expected_move_high,
        "expected_move_low":  expected_move_low,
    }

if __name__ == "__main__":
    import json
    print(json.dumps(generate_signals(), indent=2))
