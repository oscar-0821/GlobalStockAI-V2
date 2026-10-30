"""Currency conversion helpers. Market prices remain in their listing currency."""
from functools import lru_cache
import time
try:
    import yfinance as yf
except ImportError:
    yf = None

@lru_cache(maxsize=64)
def _cached_rate(source: str, target: str, bucket: int):
    if source == target:
        return 1.0
    if yf is None:
        return None
    pair = f"{source}{target}=X"
    try:
        hist = yf.Ticker(pair).history(period="5d", interval="1d", auto_adjust=False)
        if not hist.empty and float(hist["Close"].dropna().iloc[-1]) > 0:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    # Try inverse pair if direct quote is unavailable.
    try:
        inverse = yf.Ticker(f"{target}{source}=X").history(period="5d", interval="1d")
        value = float(inverse["Close"].dropna().iloc[-1])
        if value > 0:
            return 1.0 / value
    except Exception:
        pass
    return None

def get_fx_rate(source: str, target: str = "MYR"):
    source, target = (source or "").upper(), (target or "MYR").upper()
    return _cached_rate(source, target, int(time.time() // 3600))

def convert_amount(amount, source: str, target: str = "MYR"):
    rate = get_fx_rate(source, target)
    if rate is None:
        return {"success": False, "rate": None, "amount": None, "source_currency": source, "target_currency": target}
    return {"success": True, "rate": round(rate, 8), "amount": round(float(amount) * rate, 4), "source_currency": source, "target_currency": target}
