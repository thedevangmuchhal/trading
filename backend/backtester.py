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
    Runs a 1-year daily backtest of the signal strategy.
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
    atr = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(span=14, adjust=False).mean()

    # EMA trend
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

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

    for i in range(1, len(df)):
        rsi = df['RSI'].iloc[i-1]
        macd = df['MACD'].iloc[i-1]
        signal = df['Signal_Line'].iloc[i-1]
        close_price = df['Close'].iloc[i-1]
        adx = df['ADX'].iloc[i-1]

        # Multi-factor signal (mimics ai_engine logic)
        trend_bullish = close_price > df['EMA_50'].iloc[i-1]
        confidence = 50

        if rsi > 55 and macd > signal and trend_bullish:
            confidence += 30
        if rsi < 45 and macd < signal and not trend_bullish:
            confidence -= 30
        if adx > 25:  # Strong trend confirmation
            if confidence > 50: confidence += 10
            elif confidence < 50: confidence -= 10

        action = "WAIT"
        if confidence >= 75:
            action = "BUY"
        elif confidence <= 25:
            action = "SELL"

        actual_return = float(df['Returns'].iloc[i])

        if action == "BUY":
            trades += 1
            trade_returns.append(actual_return)
            if actual_return > 0:
                wins += 1
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                losses += 1
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            capital *= (1 + actual_return)

        elif action == "SELL":
            trades += 1
            trade_returns.append(-actual_return)
            if actual_return < 0:
                wins += 1
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                losses += 1
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            capital *= (1 - actual_return)

        if capital > peak_capital:
            peak_capital = capital
        drawdown = (peak_capital - capital) / peak_capital
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    win_rate = round((wins / trades) * 100, 1) if trades > 0 else 0
    total_return = round(((capital - initial_capital) / initial_capital) * 100, 1)

    # Sharpe Ratio (annualized, assuming ~252 trading days)
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
