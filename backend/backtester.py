import yfinance as yf
import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_backtest():
    print("Downloading 1 Year of Nifty Data for Backtesting...")
    df = yf.download("^NSEI", period="1y", interval="1d")
    
    if df.empty:
        print("Failed to download data.")
        return
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df['Returns'] = df['Close'].pct_change()
    df['RSI'] = calculate_rsi(df)
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df.dropna(inplace=True)
    
    initial_capital = 100000 # 1 Lakh INR
    capital = initial_capital
    position = 0
    trades = 0
    wins = 0
    peak_capital = initial_capital
    max_drawdown = 0
    
    # Simple simulated logic that mimics ai_engine technical portion
    for i in range(1, len(df)):
        rsi = df['RSI'].iloc[i-1]
        macd = df['MACD'].iloc[i-1]
        signal = df['Signal_Line'].iloc[i-1]
        close_price = df['Close'].iloc[i-1]
        
        # Calculate MTF trend proxy
        trend_bullish = close_price > df['Close'].rolling(50).mean().iloc[i-1]
        
        # Signal Generation
        action = "WAIT"
        confidence = 50
        
        if rsi > 55 and macd > signal and trend_bullish:
            confidence += 30
        if rsi < 45 and macd < signal and not trend_bullish:
            confidence -= 30
            
        if confidence >= 80:
            action = "BUY"
        elif confidence <= 20:
            action = "SELL"
            
        # Execute Trade (T+1 return)
        actual_return = df['Returns'].iloc[i]
        
        if action == "BUY":
            trades += 1
            if actual_return > 0: wins += 1
            capital *= (1 + actual_return)
        elif action == "SELL":
            trades += 1
            if actual_return < 0: wins += 1
            capital *= (1 - actual_return) # Shorting
            
        if capital > peak_capital:
            peak_capital = capital
        drawdown = (peak_capital - capital) / peak_capital
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            
    win_rate = (wins / trades) * 100 if trades > 0 else 0
    total_return = ((capital - initial_capital) / initial_capital) * 100
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS (1 YEAR - NIFTY FUTURES)")
    print("="*40)
    print(f"Total Trades Taken: {trades}")
    print(f"Win Rate:           {win_rate:.2f}%")
    print(f"Total Return:       {total_return:.2f}%")
    print(f"Max Drawdown:       {max_drawdown*100:.2f}%")
    print(f"Ending Capital:     ₹{capital:,.2f}")
    print("="*40)

if __name__ == "__main__":
    run_backtest()
