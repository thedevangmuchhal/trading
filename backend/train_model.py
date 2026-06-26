import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — 12 features matching ml_engine.py
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(rsi, period=14, smooth_k=3):
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    stoch = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
    return stoch.rolling(smooth_k).mean()

def calculate_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    plus_dm = high.diff(); minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(span=period, adjust=False).mean()

def calculate_supertrend_dir(df, period=10, multiplier=3.0):
    """Returns +1 for bullish, -1 for bearish Supertrend direction."""
    hl2 = (df['High'] + df['Low']) / 2
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

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
            if close[i] > fu[i]: di[i] = 1; st[i] = fl[i]
            else: di[i] = -1; st[i] = fu[i]
        else:
            if close[i] < fl[i]: di[i] = -1; st[i] = fu[i]
            else: di[i] = 1; st[i] = fl[i]

    return pd.Series(di, index=df.index)


def prepare_data():
    print("Fetching 60 days of ^NSEI 15m data for training...")
    df = yf.download("^NSEI", period="60d", interval="15m")

    if df.empty:
        print("Error: No data fetched from yfinance.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    print("Calculating 12 features...")

    # 1. RSI
    df['RSI'] = calculate_rsi(df)

    # 2-3. MACD + Signal Line
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # 5. Returns
    df['Returns'] = df['Close'].pct_change()

    # 6. Stochastic RSI K
    df['Stoch_RSI_K'] = calculate_stoch_rsi(df['RSI'])

    # 7. ADX
    df['ADX'] = calculate_adx(df)

    # 8. Volume Ratio (current / 20-period average)
    vol_avg = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / vol_avg.replace(0, 1)

    # 9. EMA 9/21 Cross (+1 if 9 > 21, -1 otherwise)
    ema9 = df['Close'].ewm(span=9, adjust=False).mean()
    ema21 = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_9_21_Cross'] = np.where(ema9 > ema21, 1, -1)

    # 10. Bollinger Band Position (0 to 1 scale)
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, 1)
    df['BB_Position'] = (df['Close'] - bb_lower) / bb_range

    # 11. VWAP Distance (% from VWAP)
    cum_vol = df['Volume'].cumsum()
    cum_pv = (df['Close'] * df['Volume']).cumsum()
    vwap = cum_pv / cum_vol.replace(0, 1)
    df['VWAP_Distance'] = ((df['Close'] - vwap) / df['Close'].replace(0, 1)) * 100

    # 12. Supertrend Direction (+1 or -1)
    df['Supertrend_Dir'] = calculate_supertrend_dir(df)

    # Target: Will price be higher in next 4 candles (1 hour)?
    df['Future_Return'] = df['Close'].shift(-4) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > 0.001).astype(int)

    df.dropna(inplace=True)
    return df


FEATURES = [
    'RSI', 'MACD', 'Signal_Line', 'ATR', 'Returns',
    'Stoch_RSI_K', 'ADX', 'Volume_Ratio', 'EMA_9_21_Cross',
    'BB_Position', 'VWAP_Distance', 'Supertrend_Dir'
]


def train():
    df = prepare_data()
    if df is None:
        return

    X = df[FEATURES]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"Training GradientBoosting Model on {len(X_train)} samples with {len(FEATURES)} features...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy on Test Set: {acc * 100:.2f}%")
    print(classification_report(y_test, preds))

    # Feature importance
    print("\nFeature Importance:")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:20s} {imp:.4f}")

    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
