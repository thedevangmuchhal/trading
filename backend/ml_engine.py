import joblib
import os
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
_model = None

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
    Expects tech_data dictionary with: RSI, MACD, Signal_Line, ATR, Returns.
    """
    model = load_model()
    if model is None:
        return None
        
    try:
        # Construct feature array exactly as trained: ['RSI', 'MACD', 'Signal_Line', 'ATR', 'Returns']
        features = pd.DataFrame([{
            'RSI': tech_data.get('rsi', 50),
            'MACD': tech_data.get('macd', 0),
            'Signal_Line': tech_data.get('macd_signal', 0),
            'ATR': tech_data.get('atr', 0),
            'Returns': tech_data.get('returns', 0)
        }])
        
        # predict_proba returns [[prob_0, prob_1]]
        probs = model.predict_proba(features)
        return probs[0][1] # Probability of class 1 (Bullish)
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None
