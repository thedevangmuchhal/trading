"""
Signal Logger — Background service that monitors signal changes and logs
every trade entry/exit to a Google Sheet via Apps Script webhook.

Runs automatically during market hours (9:15-15:30 IST, Mon-Fri).
Auto-enters on BUY/SELL signal, auto-exits when signal flips.
"""

import threading
import time
import requests
import json
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

_POLL_INTERVAL = 60          # seconds between signal checks
_SHEET_URL = None            # Google Apps Script web app URL (set via API)
_LOGGER_THREAD = None
_LOGGER_RUNNING = False
_TICKER = "^NSEI"
_LOT_SIZE = 25               # NIFTY lot size

# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

_current_position = None     # {"action": "BUY"/"SELL", "entry_time": ..., ...}
_last_action = "WAIT"
_signal_log = []             # In-memory log for today
_daily_pnl = 0.0
_trade_count = 0
_lock = threading.Lock()


def _ist_now():
    """Get current IST time."""
    utc_now = datetime.utcnow()
    return utc_now + timedelta(hours=5, minutes=30)


def _is_market_hours():
    """Check if current IST time is within market hours (Mon-Fri, 9:15-15:30)."""
    ist = _ist_now()
    if ist.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    total_mins = ist.hour * 60 + ist.minute
    return 555 <= total_mins <= 930  # 9:15 to 15:30


def _post_to_sheet(row_data):
    """POST a row to Google Sheets via Apps Script webhook."""
    global _SHEET_URL
    if not _SHEET_URL:
        print("[SignalLogger] No sheet URL configured. Skipping POST.")
        return False
    try:
        # Inject date_sheet for date-wise tabs (e.g., "30062026")
        ist = _ist_now()
        row_data["date_sheet"] = ist.strftime("%d%m%Y")

        resp = requests.post(
            _SHEET_URL,
            json=row_data,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        if resp.status_code == 200:
            print(f"[SignalLogger] ✅ Logged to sheet: {row_data.get('trade', 'N/A')}")
            return True
        else:
            print(f"[SignalLogger] ❌ Sheet POST failed: {resp.status_code} — {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"[SignalLogger] ❌ Sheet POST error: {e}")
        return False


def _close_position(signal_data):
    """Close current open position and log exit."""
    global _current_position, _daily_pnl, _trade_count

    if not _current_position:
        return

    pos = _current_position
    exit_time = _ist_now().strftime("%H:%M:%S")
    exit_price = signal_data.get('current_price', 0)

    # Get current option premium for exit
    if pos["option_type"] == "CE":
        exit_premium = signal_data.get('atm_ce_ltp', 0)
    else:
        exit_premium = signal_data.get('atm_pe_ltp', 0)

    # Calculate P&L
    premium_diff = exit_premium - pos["entry_premium"]
    pnl_points = round(premium_diff, 2)
    pnl_rupees = round(premium_diff * _LOT_SIZE, 2)
    _daily_pnl += pnl_rupees
    _trade_count += 1

    # Build exit row
    row = {
        "type": "EXIT",
        "entry_time": pos["entry_time"],
        "spot_price": pos["entry_spot"],
        "trade": f"{pos['strike']} {pos['option_type']}",
        "entry_premium": pos["entry_premium"],
        "exit_time": exit_time,
        "exit_spot": exit_price,
        "exit_premium": exit_premium,
        "pnl_points": pnl_points,
        "pnl_rupees": pnl_rupees,
        "confidence": pos["confidence"],
        "signal_strength": pos["signal_strength"],
        "mtf": pos["mtf"],
        "reasons": pos.get("reasons", ""),
        "duration_mins": _calc_duration(pos["entry_time"], exit_time),
    }

    _signal_log.append(row)
    _post_to_sheet(row)
    _current_position = None
    print(f"[SignalLogger] 📤 Closed {pos['strike']} {pos['option_type']} — P&L: ₹{pnl_rupees}")


def _open_position(signal_data):
    """Open a new position based on signal."""
    global _current_position

    action = signal_data.get('action', 'WAIT')
    if action == 'WAIT':
        return

    entry_time = _ist_now().strftime("%H:%M:%S")
    spot = signal_data.get('current_price', 0)
    strike_rec = signal_data.get('strike_recommendation', '')
    confidence = signal_data.get('confidence', 50)
    signal_strength = signal_data.get('signal_strength', 0)
    mtf = signal_data.get('mtf_confluence', '')
    reasons = " | ".join(signal_data.get('signal_reasons', [])[:3])

    # Determine option type and strike
    if action == "BUY":
        option_type = "CE"
        entry_premium = signal_data.get('atm_ce_ltp', 0)
    else:
        option_type = "PE"
        entry_premium = signal_data.get('atm_pe_ltp', 0)

    # Parse strike from recommendation (e.g., "24100 CE Δ0.42")
    strike = signal_data.get('atm_strike', round(spot / 50) * 50)
    try:
        parts = strike_rec.split()
        if len(parts) >= 1:
            strike = float(parts[0])
    except:
        pass

    _current_position = {
        "action": action,
        "option_type": option_type,
        "strike": strike,
        "entry_time": entry_time,
        "entry_spot": spot,
        "entry_premium": entry_premium,
        "confidence": confidence,
        "signal_strength": signal_strength,
        "mtf": mtf,
        "reasons": reasons,
    }

    # Log entry row
    row = {
        "type": "ENTRY",
        "entry_time": entry_time,
        "spot_price": spot,
        "trade": f"{strike} {option_type}",
        "entry_premium": entry_premium,
        "exit_time": "",
        "exit_spot": "",
        "exit_premium": "",
        "pnl_points": "",
        "pnl_rupees": "",
        "confidence": confidence,
        "signal_strength": signal_strength,
        "mtf": mtf,
        "reasons": reasons,
        "duration_mins": "",
    }

    _signal_log.append(row)
    _post_to_sheet(row)
    print(f"[SignalLogger] 📥 Opened {strike} {option_type} @ ₹{entry_premium} (Conf: {confidence})")


def _calc_duration(entry_str, exit_str):
    """Calculate duration in minutes between HH:MM:SS strings."""
    try:
        fmt = "%H:%M:%S"
        e = datetime.strptime(entry_str, fmt)
        x = datetime.strptime(exit_str, fmt)
        return int((x - e).total_seconds() / 60)
    except:
        return 0


def _signal_loop():
    """Main loop: poll signals and detect changes."""
    global _last_action, _LOGGER_RUNNING, _daily_pnl, _trade_count

    from ai_engine import generate_signals

    print("[SignalLogger] 🚀 Logger started. Waiting for market hours...")

    # Reset daily state
    last_reset_date = None

    while _LOGGER_RUNNING:
        try:
            ist = _ist_now()

            # Reset daily counters at market open
            if ist.date() != last_reset_date and ist.hour == 9 and ist.minute >= 15:
                _daily_pnl = 0.0
                _trade_count = 0
                _last_action = "WAIT"
                last_reset_date = ist.date()
                print(f"[SignalLogger] 📅 New trading day: {ist.date()}")

            if not _is_market_hours():
                # Auto-close any open position at market close (15:30)
                if _current_position and ist.hour >= 15 and ist.minute >= 29:
                    try:
                        sig = generate_signals(_TICKER)
                        if not sig.get('error'):
                            _close_position(sig)
                            print("[SignalLogger] ⏰ Market closing — force-closed position")

                            # Post daily summary
                            _post_to_sheet({
                                "type": "DAILY_SUMMARY",
                                "entry_time": f"=== {ist.strftime('%Y-%m-%d')} SUMMARY ===",
                                "spot_price": "",
                                "trade": f"Total Trades: {_trade_count}",
                                "entry_premium": "",
                                "exit_time": "",
                                "exit_spot": "",
                                "exit_premium": "",
                                "pnl_points": "",
                                "pnl_rupees": _daily_pnl,
                                "confidence": "",
                                "signal_strength": "",
                                "mtf": "",
                                "reasons": f"Day P&L: ₹{_daily_pnl:,.2f}",
                                "duration_mins": "",
                            })
                    except Exception as e:
                        print(f"[SignalLogger] Error on market close: {e}")

                time.sleep(30)
                continue

            # ── Poll signal ──────────────────────────────────────────────
            sig = generate_signals(_TICKER)
            if sig.get('error'):
                print(f"[SignalLogger] ⚠ Signal error: {sig['error']}")
                time.sleep(_POLL_INTERVAL)
                continue

            new_action = sig.get('action', 'WAIT')

            with _lock:
                # Detect signal change
                if new_action != _last_action:
                    print(f"[SignalLogger] 🔄 Signal changed: {_last_action} → {new_action}")

                    # 1. Close existing position if any
                    if _current_position:
                        _close_position(sig)

                    # 2. Open new position if BUY or SELL
                    if new_action in ("BUY", "SELL"):
                        _open_position(sig)

                    _last_action = new_action

                # Even if signal didn't change, update the position's current premium
                elif _current_position:
                    if _current_position["option_type"] == "CE":
                        _current_position["_current_premium"] = sig.get('atm_ce_ltp', 0)
                    else:
                        _current_position["_current_premium"] = sig.get('atm_pe_ltp', 0)

        except Exception as e:
            print(f"[SignalLogger] ❌ Loop error: {e}")

        time.sleep(_POLL_INTERVAL)

    print("[SignalLogger] 🛑 Logger stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API (called from main.py)
# ─────────────────────────────────────────────────────────────────────────────

def start_logger(sheet_url: str, ticker: str = "^NSEI", lot_size: int = 25):
    """Start the signal logger background thread."""
    global _SHEET_URL, _TICKER, _LOT_SIZE, _LOGGER_THREAD, _LOGGER_RUNNING

    _SHEET_URL = sheet_url
    _TICKER = ticker
    _LOT_SIZE = lot_size
    _LOGGER_RUNNING = True

    if _LOGGER_THREAD and _LOGGER_THREAD.is_alive():
        return {"status": "already_running"}

    # Send a "START" heartbeat so the sheet is created immediately
    ist_time = _ist_now().strftime("%H:%M:%S")
    _post_to_sheet({
        "type": "START",
        "entry_time": ist_time,
        "trade": "Logger Started"
    })

    _LOGGER_THREAD = threading.Thread(target=_signal_loop, daemon=True)
    _LOGGER_THREAD.start()
    return {"status": "started", "sheet_url": sheet_url, "ticker": ticker}


def stop_logger():
    """Stop the signal logger."""
    global _LOGGER_RUNNING
    _LOGGER_RUNNING = False
    return {"status": "stopped"}


def get_logger_status():
    """Return current logger state."""
    return {
        "running": _LOGGER_RUNNING and _LOGGER_THREAD and _LOGGER_THREAD.is_alive(),
        "sheet_url": _SHEET_URL,
        "ticker": _TICKER,
        "lot_size": _LOT_SIZE,
        "current_position": {
            "trade": f"{_current_position['strike']} {_current_position['option_type']}" if _current_position else None,
            "entry_time": _current_position["entry_time"] if _current_position else None,
            "entry_premium": _current_position["entry_premium"] if _current_position else None,
        } if _current_position else None,
        "last_action": _last_action,
        "daily_pnl": round(_daily_pnl, 2),
        "trade_count": _trade_count,
        "today_log_count": len(_signal_log),
    }


def get_today_log():
    """Return all signal log entries for today."""
    return _signal_log.copy()
