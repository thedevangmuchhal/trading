from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_engine import generate_signals

app = FastAPI(title="AI Trading API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/signal")
def get_signal(ticker: str = "^NSEI"):
    """
    Returns the latest AI-generated trading signal and market data.
    """
    data = generate_signals(ticker)
    return data

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
