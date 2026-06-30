import pandas as pd
import yfinance as yf
import numpy as np

def run_scalper_backtest():
    ticker = "^NSEI"
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if df.empty:
        return

    close = df['Close'].squeeze()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Returns'] = close.pct_change()
    
    trades = 0
    wins = 0
    pnl = 0
    
    for i in range(20, len(df)):
        rsi = df['RSI'].iloc[i-1]
        ret = df['Returns'].iloc[i]
        
        # Mean Reversion: Buy when Oversold
        if rsi < 40:
            trades += 1
            # Take profit at +0.2%, Stop loss at -0.6%
            trade_pnl = ret
            if trade_pnl >= 0.002:
                trade_pnl = 0.002
            elif trade_pnl <= -0.006:
                trade_pnl = -0.006
                
            pnl += trade_pnl
            if trade_pnl > 0: wins += 1
            
        # Sell when Overbought
        elif rsi > 60:
            trades += 1
            trade_pnl = -ret
            if trade_pnl >= 0.002:
                trade_pnl = 0.002
            elif trade_pnl <= -0.006:
                trade_pnl = -0.006
                
            pnl += trade_pnl
            if trade_pnl > 0: wins += 1

    win_rate = (wins / trades * 100) if trades > 0 else 0
    print(f"Total Trades: {trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total PNL: {pnl*100:.2f}%")

run_scalper_backtest()
