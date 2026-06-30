import pandas as pd
import yfinance as yf
import numpy as np

def run_golden_sniper_backtest():
    ticker = "^NSEI"
    # Get 60 days of 15m data (max allowed for 15m)
    df = yf.download(ticker, period="60d", interval="15m", progress=False)
    if df.empty:
        print("Failed to download data")
        return

    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze()

    # If volume is 0 or NaN (sometimes true for NSE indices on yfinance), 
    # we simulate volume surge using ATR surge instead.
    has_volume = volume.sum() > 0

    # 1. Bollinger Bands (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)

    # 2. RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. ATR for Surge/StopLoss
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['ATR'] = atr

    # 4. Surge Detection (Volume or ATR)
    if has_volume:
        vol_sma = volume.rolling(20).mean()
        df['Surge'] = volume > (1.5 * vol_sma)
    else:
        df['Surge'] = tr > (1.5 * atr)

    trades = 0
    wins = 0
    total_pnl_pct = 0
    
    in_trade = False
    entry_price = 0
    trade_type = ""
    stop_loss = 0
    take_profit = 0

    for i in range(20, len(df)):
        current_time = df.index[i].time()
        # Convert UTC to IST if needed, but yfinance usually returns local for NSE if timezone aware.
        # Assuming index is tz-aware or we just use hour/minute roughly.
        # Let's just check the hour to approximate the chop zone (11:00 to 13:30)
        hour = current_time.hour
        minute = current_time.minute
        time_mins = hour * 60 + minute
        
        # 11:00 AM = 660 mins, 1:30 PM = 810 mins
        in_chop_zone = (660 <= time_mins <= 810)

        c = float(close.iloc[i])
        h = float(high.iloc[i])
        l = float(low.iloc[i])
        r = float(df['RSI'].iloc[i-1])
        lb = float(lower_bb.iloc[i-1])
        ub = float(upper_bb.iloc[i-1])
        surge = bool(df['Surge'].iloc[i-1])
        atr_val = float(df['ATR'].iloc[i-1])

        if in_trade:
            # Move Stop to Breakeven if in profit
            if trade_type == "BUY" and h >= entry_price + (1.0 * atr_val):
                stop_loss = max(stop_loss, entry_price)
            elif trade_type == "SELL" and l <= entry_price - (1.0 * atr_val):
                stop_loss = min(stop_loss, entry_price)

            # Check exits
            if trade_type == "BUY":
                if h >= take_profit:
                    total_pnl_pct += (take_profit - entry_price) / entry_price
                    wins += 1
                    in_trade = False
                elif l <= stop_loss:
                    total_pnl_pct += (stop_loss - entry_price) / entry_price
                    if stop_loss > entry_price:
                        wins += 1 # Technical win
                    in_trade = False
            elif trade_type == "SELL":
                if l <= take_profit:
                    total_pnl_pct += (entry_price - take_profit) / entry_price
                    wins += 1
                    in_trade = False
                elif h >= stop_loss:
                    total_pnl_pct += (entry_price - stop_loss) / entry_price
                    if stop_loss < entry_price:
                        wins += 1 # Technical win
                    in_trade = False
            continue

        # Check Entries
        # Golden Sniper BUY: Not in chop zone, price touches lower BB, RSI oversold, with a surge
        if not in_chop_zone and l <= lb and r < 30 and surge:
            in_trade = True
            trade_type = "BUY"
            entry_price = c
            trades += 1
            # R:R = 1:1.5
            stop_loss = entry_price - (1.0 * atr_val)
            take_profit = entry_price + (1.5 * atr_val)

        # Golden Sniper SELL: Not in chop zone, price touches upper BB, RSI overbought, with a surge
        elif not in_chop_zone and h >= ub and r > 70 and surge:
            in_trade = True
            trade_type = "SELL"
            entry_price = c
            trades += 1
            # R:R = 1:1.5
            stop_loss = entry_price + (1.0 * atr_val)
            take_profit = entry_price - (1.5 * atr_val)

    win_rate = (wins / trades * 100) if trades > 0 else 0
    print(f"--- GOLDEN SNIPER (15m Intraday, 60 Days) ---")
    print(f"Total Trades: {trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total PNL (Unleveraged): {total_pnl_pct*100:.2f}%")
    print(f"Risk:Reward Ratio per trade: 1 : 1.5")

if __name__ == "__main__":
    run_golden_sniper_backtest()
