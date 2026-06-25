import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data():
    print("Fetching historical data for ^NSEI (15m interval)...")
    df = yf.download("^NSEI", period="60d", interval="15m")
    
    if df.empty:
        print("Error: No data fetched from yfinance.")
        return None

    # Handle multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    print("Calculating Features...")
    df['Returns'] = df['Close'].pct_change()
    df['RSI'] = calculate_rsi(df)
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ATR Approximation
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # Target: Will the price be higher in the next 4 periods (1 hour)?
    df['Future_Return'] = df['Close'].shift(-4) / df['Close'] - 1
    
    # Label: 1 if return > 0.1% (Bullish), 0 otherwise (Bearish/Sideways)
    df['Target'] = (df['Future_Return'] > 0.001).astype(int)
    
    df.dropna(inplace=True)
    return df

def train():
    df = prepare_data()
    if df is None: return
    
    features = ['RSI', 'MACD', 'Signal_Line', 'ATR', 'Returns']
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training RandomForest Model on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy on Test Set: {acc * 100:.2f}%")
    print(classification_report(y_test, preds))
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train()
