import joblib
import os
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
_model = None

# Feature list MUST match train_model.py exactly
FEATURES = [
    'RSI', 'MACD', 'Signal_Line', 'ATR', 'Returns',
    'Stoch_RSI_K', 'ADX', 'Volume_Ratio', 'EMA_9_21_Cross',
    'BB_Position', 'VWAP_Distance', 'Supertrend_Dir'
]

def load_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading ML model: {e}")
    return _model

def predict_breakout_probability(tech_data):
    """
    Returns the probability (0.0 to 1.0) of a bullish breakout in the next hour.
    Uses 12 features from tech_data (matching train_model.py feature set).
    Gracefully handles missing features with sensible defaults.
    """
    model = load_model()
    if model is None:
        return None

    try:
        # EMA 9/21 Cross: +1 if ema_9 > ema_21, -1 otherwise
        ema9 = tech_data.get('ema_9', 0) or 0
        ema21 = tech_data.get('ema_21', 0) or 0
        ema_cross = 1 if ema9 > ema21 else -1

        # BB Position: (close - bb_lower) / (bb_upper - bb_lower)
        bb_upper = tech_data.get('bb_upper', 0) or 0
        bb_lower = tech_data.get('bb_lower', 0) or 0
        bb_range = bb_upper - bb_lower if bb_upper > bb_lower else 1
        current_price = tech_data.get('current_price', 0) or 0
        bb_position = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5

        # VWAP Distance: % from VWAP
        vwap = tech_data.get('vwap', 0) or 0
        vwap_distance = ((current_price - vwap) / current_price * 100) if current_price > 0 and vwap > 0 else 0

        # Supertrend Dir: +1 bullish, -1 bearish
        st_dir = 1 if tech_data.get('supertrend_direction') == 'Bullish' else -1

        # Volume ratio
        vol_ratio = tech_data.get('volume_ratio', 1.0) or 1.0

        features = pd.DataFrame([{
            'RSI':            tech_data.get('rsi', 50),
            'MACD':           tech_data.get('macd', 0),
            'Signal_Line':    tech_data.get('macd_signal', 0),
            'ATR':            tech_data.get('atr', 0),
            'Returns':        tech_data.get('returns', 0),
            'Stoch_RSI_K':    tech_data.get('stoch_rsi_k', 50),
            'ADX':            tech_data.get('adx', 0),
            'Volume_Ratio':   vol_ratio,
            'EMA_9_21_Cross': ema_cross,
            'BB_Position':    bb_position,
            'VWAP_Distance':  vwap_distance,
            'Supertrend_Dir': st_dir,
        }])

        # Ensure column order matches training
        features = features[FEATURES]

        probs = model.predict_proba(features)
        return probs[0][1]  # Probability of class 1 (Bullish)
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None
