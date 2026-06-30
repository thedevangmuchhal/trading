import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CACHE — backtest results cached for 6 hours (historical data, rarely changes)
# ─────────────────────────────────────────────────────────────────────────────
_backtest_cache = {}
_CACHE_TTL = 6 * 3600  # 6 hours


def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def run_backtest(ticker="^NSEI"):
    """
    Runs a 1-year daily backtest with ALL capital preservation rules.
    Returns a dict with performance metrics (cached for 6 hours).
    """
    global _backtest_cache

    # Check cache
    now_ts = datetime.now().timestamp()
    if ticker in _backtest_cache:
        cached = _backtest_cache[ticker]
        if (now_ts - cached['timestamp']) < _CACHE_TTL:
            return cached['data']

    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
    except Exception as e:
        return {"error": f"Failed to download data: {e}"}

    if df.empty:
        return {"error": "No data available for backtest"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df['Returns'] = df['Close'].pct_change()
    df['RSI'] = calculate_rsi(df)

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ADX for trend strength
    high = df['High']; low = df['Low']; close = df['Close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().abs().clip(lower=0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_series = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_series)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_series)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(span=14, adjust=False).mean()
    df['ATR'] = atr_series

    # EMA trend
    df['EMA_9']  = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # Stochastic RSI
    rsi_min = df['RSI'].rolling(14).min()
    rsi_max = df['RSI'].rolling(14).max()
    df['StochRSI_K'] = ((df['RSI'] - rsi_min) / (rsi_max - rsi_min) * 100).rolling(3).mean()

    # Supertrend (simplified)
    hl2 = (df['High'] + df['Low']) / 2
    df['ST_upper'] = hl2 + 2.0 * df['ATR']
    df['ST_lower'] = hl2 - 2.0 * df['ATR']
    df['Supertrend'] = 0  # 1=bullish, -1=bearish
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['ST_upper'].iloc[i-1]:
            df.iloc[i, df.columns.get_loc('Supertrend')] = 1
        elif df['Close'].iloc[i] < df['ST_lower'].iloc[i-1]:
            df.iloc[i, df.columns.get_loc('Supertrend')] = -1
        else:
            df.iloc[i, df.columns.get_loc('Supertrend')] = df['Supertrend'].iloc[i-1]

    df.dropna(inplace=True)

    initial_capital = 100000  # ₹1 Lakh
    capital = initial_capital
    trades = 0
    wins = 0
    losses = 0
    peak_capital = initial_capital
    max_drawdown = 0
    consecutive_wins = 0
    max_consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    trade_returns = []

    # ── Capital Preservation State ────────────────────────────────────────
    daily_loss = 0
    MAX_DAILY_LOSS = 5000
    consec_losses_today = 0
    blocked_until = 0  # index until which trading is blocked
    trades_today = 0
    last_day = None

    for i in range(1, len(df)):
        current_day = df.index[i].date()
        close_price = float(df['Close'].iloc[i-1])
        rsi = float(df['RSI'].iloc[i-1])
        macd = float(df['MACD'].iloc[i-1])
        signal = float(df['Signal_Line'].iloc[i-1])
        adx = float(df['ADX'].iloc[i-1])
        stoch_k = float(df['StochRSI_K'].iloc[i-1]) if not np.isnan(df['StochRSI_K'].iloc[i-1]) else 50
        supertrend = int(df['Supertrend'].iloc[i-1])
        ema9 = float(df['EMA_9'].iloc[i-1])
        ema21 = float(df['EMA_21'].iloc[i-1])
        ema50 = float(df['EMA_50'].iloc[i-1])

        # Reset daily counters
        if current_day != last_day:
            daily_loss = 0
            consec_losses_today = 0
            trades_today = 0
            blocked_until = 0
            last_day = current_day

        # ── Capital Preservation Gates ────────────────────────────────────
        # Gate: Max daily loss
        if daily_loss >= MAX_DAILY_LOSS:
            continue

        # Gate: Consecutive loss cooldown (skip 3 bars after 3 consecutive losses)
        if i < blocked_until:
            continue

        # Gate: Choppy market hard block (ADX < 15)
        if adx < 15:
            continue

        # ── Multi-Factor Confidence (Adaptive Weights) ────────────────────
        # Pillar 1: Technical score
        tech_score = 50
        if rsi > 55 and macd > signal: tech_score += 20
        elif rsi < 45 and macd < signal: tech_score -= 20
        if supertrend == 1: tech_score += 15
        elif supertrend == -1: tech_score -= 15
        if close_price > ema50: tech_score += 10
        else: tech_score -= 10
        if ema9 > ema21 > ema50: tech_score += 10
        elif ema9 < ema21 < ema50: tech_score -= 10
        if stoch_k < 20: tech_score += 10
        elif stoch_k > 80: tech_score -= 10
        tech_score = max(0, min(100, tech_score))

        # Pillar 2: Trend/Smart Money proxy (using EMA alignment as daily proxy)
        sm_score = 50
        if close_price > ema50: sm_score += 20
        if ema9 > ema21: sm_score += 15
        sm_score = max(0, min(100, sm_score))

        # Pillar 3: Sentiment proxy (momentum)
        sent_score = 50
        if rsi > 50: sent_score += 10
        if macd > 0: sent_score += 10
        sent_score = max(0, min(100, sent_score))

        # ── Adaptive Weights based on ADX ─────────────────────────────────
        if adx >= 25:
            w_tech, w_sm, w_sent = 0.50, 0.30, 0.20
        elif adx < 20:
            w_tech, w_sm, w_sent = 0.30, 0.40, 0.30
        else:
            w_tech, w_sm, w_sent = 0.40, 0.35, 0.25

        confidence = int(tech_score * w_tech + sm_score * w_sm + sent_score * w_sent)

        # ── Pillar Agreement Gate ─────────────────────────────────────────
        bullish_pillars = sum([tech_score > 60, sm_score > 60, sent_score > 55])
        bearish_pillars = sum([tech_score < 40, sm_score < 40, sent_score < 45])
        agreement = max(bullish_pillars, bearish_pillars)

        # Determine action
        action = "WAIT"
        if confidence >= 70 and agreement >= 2:
            action = "BUY"
        elif confidence <= 30 and agreement >= 2:
            action = "SELL"

        # ── Signal Strength (conviction filter) ───────────────────────────
        signal_strength = 0
        if abs(rsi - 50) > 15: signal_strength += 2
        if supertrend != 0: signal_strength += 2
        if adx > 25: signal_strength += 2
        if abs(macd) > abs(signal) * 0.5: signal_strength += 1
        if ema9 > ema21 > ema50 or ema9 < ema21 < ema50: signal_strength += 2

        # Conviction filter: skip weak signals
        if signal_strength < 4:
            action = "WAIT"

        # ── Execute Trade ─────────────────────────────────────────────────
        actual_return = float(df['Returns'].iloc[i])
        atr_val = float(df['ATR'].iloc[i-1])

        if action == "BUY":
            trades += 1
            trades_today += 1
            pnl = actual_return

            # Trailing SL simulation: if return is negative and exceeds 1.5 ATR%, stop out early
            max_loss_pct = (1.5 * atr_val / close_price) if close_price > 0 else 0.02
            if pnl < -max_loss_pct:
                pnl = -max_loss_pct  # Capped by trailing SL

            trade_returns.append(pnl)
            if pnl > 0:
                wins += 1
                consecutive_wins += 1
                consecutive_losses = 0
                consec_losses_today = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                losses += 1
                consecutive_losses += 1
                consec_losses_today += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                daily_loss += abs(pnl * capital)
                if consec_losses_today >= 3:
                    blocked_until = i + 3  # Block next 3 bars

            capital *= (1 + pnl)

        elif action == "SELL":
            trades += 1
            trades_today += 1
            pnl = -actual_return

            max_loss_pct = (1.5 * atr_val / close_price) if close_price > 0 else 0.02
            if pnl < -max_loss_pct:
                pnl = -max_loss_pct

            trade_returns.append(pnl)
            if pnl > 0:
                wins += 1
                consecutive_wins += 1
                consecutive_losses = 0
                consec_losses_today = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                losses += 1
                consecutive_losses += 1
                consec_losses_today += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                daily_loss += abs(pnl * capital)
                if consec_losses_today >= 3:
                    blocked_until = i + 3

            capital *= (1 + pnl)

        if capital > peak_capital:
            peak_capital = capital
        drawdown = (peak_capital - capital) / peak_capital
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    win_rate = round((wins / trades) * 100, 1) if trades > 0 else 0
    total_return = round(((capital - initial_capital) / initial_capital) * 100, 1)

    # Sharpe Ratio (annualized, ~252 trading days)
    if trade_returns:
        avg_ret = np.mean(trade_returns)
        std_ret = np.std(trade_returns) if len(trade_returns) > 1 else 1
        sharpe = round((avg_ret / std_ret) * np.sqrt(252), 2) if std_ret > 0 else 0
        avg_trade_pct = round(avg_ret * 100, 3)
    else:
        sharpe = 0
        avg_trade_pct = 0

    result = {
        "total_trades": trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "max_drawdown": round(max_drawdown * 100, 1),
        "ending_capital": round(capital, 2),
        "sharpe_ratio": sharpe,
        "avg_trade_pct": avg_trade_pct,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "period": "1 Year",
        "ticker": ticker,
        "rules_applied": [
            "Adaptive Weights (ADX-based)",
            "Pillar Agreement Gate (2/3)",
            "Choppy Market Block (ADX<15)",
            "Conviction Filter (strength≥4)",
            "Max Daily Loss ₹5K",
            "Consecutive Loss Cooldown (3)",
            "Trailing SL (1.5×ATR)"
        ],
    }

    # Cache the result
    _backtest_cache[ticker] = {
        "timestamp": now_ts,
        "data": result,
    }

    return result


if __name__ == "__main__":
    import json
    result = run_backtest()
    print(json.dumps(result, indent=2))
