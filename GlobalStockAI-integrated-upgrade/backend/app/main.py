from importlib.util import find_spec
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .search import search_stocks
from .catalog import STOCKS
from .price_data import get_stock_history
from .indicators import calculate_indicators
from .market_data import get_stock_metadata
from .fx import get_fx_rate, convert_amount
from .risk import build_trade_plan, calculate_profit_plan

try:
    from .prediction import predict_stock
    PREDICTION_IMPORT_ERROR = None
except Exception as exc:  # keep health/search usable if optional ML dependency fails
    predict_stock = None
    PREDICTION_IMPORT_ERROR = str(exc)

app = FastAPI(title="GlobalStockAI API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

class ProfitPlanRequest(BaseModel):
    capital: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    risk_percent: float = Field(default=1.0, gt=0, le=100)
    fee_percent: float = Field(default=0.1, ge=0, le=20)
    fx_rate: float = Field(default=1.0, gt=0)

@app.get("/api/health")
def health():
    required = ["fastapi", "pandas", "numpy", "yfinance", "sklearn"]
    dependencies = {name: find_spec(name) is not None for name in required}
    ml_dependencies = {name: find_spec(name) is not None for name in ["xgboost", "lightgbm"]}
    ok = all(dependencies.values())
    return {"status": "ok" if ok else "degraded", "api": "online", "catalog_count": len(STOCKS), "catalog_mode": "demo", "dependencies": dependencies, "ml_dependencies": ml_dependencies, "prediction_available": predict_stock is not None, "prediction_import_error": PREDICTION_IMPORT_ERROR}

@app.get("/api/health/market-data")
def market_data_health():
    try:
        import yfinance as yf
        sample = yf.Ticker("^GSPC").history(period="5d", interval="1d")
        market_ok = not sample.empty and sample["Close"].dropna().size > 0
        return {"status": "ok" if market_ok else "degraded", "provider": "Yahoo Finance", "sample_symbol": "^GSPC", "records": int(len(sample)), "message": "Market data reachable" if market_ok else "Provider returned no recent data"}
    except Exception as exc:
        return {"status": "degraded", "provider": "Yahoo Finance", "message": str(exc)}

@app.get("/api/stocks/search")
def stock_search(q: str = Query(default="", max_length=120), limit: int = Query(default=10, ge=1, le=25)):
    return search_stocks(q, limit)

@app.get("/api/stocks/{ticker}/history")
def stock_history(ticker: str, period: str = Query(default="1y"), interval: str = Query(default="1d")):
    return get_stock_history(ticker=ticker, period=period, interval=interval)

@app.get("/api/stocks/{ticker}/indicators")
def stock_indicators(ticker: str, period: str = Query(default="1y")):
    return calculate_indicators(ticker=ticker, period=period)

@app.get("/api/stocks/{ticker}/prediction")
def stock_prediction(ticker: str, period: str = Query(default="5y")):
    if predict_stock is None:
        raise HTTPException(status_code=503, detail=f"Prediction engine unavailable: {PREDICTION_IMPORT_ERROR}")
    return predict_stock(ticker, period)

@app.get("/api/fx/rate")
def fx_rate(from_currency: str = Query(min_length=3, max_length=3), to_currency: str = Query(default="MYR", min_length=3, max_length=3)):
    rate = get_fx_rate(from_currency, to_currency)
    if rate is None:
        raise HTTPException(status_code=502, detail=f"Could not retrieve {from_currency}/{to_currency} exchange rate")
    return {"success": True, "from_currency": from_currency.upper(), "to_currency": to_currency.upper(), "rate": rate, "provider": "Yahoo Finance", "cache_window": "1 hour"}

@app.get("/api/fx/convert")
def fx_convert(amount: float = Query(ge=0), from_currency: str = Query(min_length=3, max_length=3), to_currency: str = Query(default="MYR", min_length=3, max_length=3)):
    result = convert_amount(amount, from_currency, to_currency)
    if not result["success"]:
        raise HTTPException(status_code=502, detail="Currency conversion rate unavailable")
    return result

@app.get("/api/stocks/{ticker}/trade-plan")
def stock_trade_plan(ticker: str, direction: str = Query(default="LONG", pattern="^(LONG|SHORT)$"), risk_reward: float = Query(default=2.0, ge=0.5, le=10)):
    data = calculate_indicators(ticker, "1y")
    if not data.get("success"):
        raise HTTPException(status_code=502, detail=data.get("message", "Indicator data unavailable"))
    atr = data.get("indicators", {}).get("ATR_14")
    if not atr:
        raise HTTPException(status_code=422, detail="Not enough data to calculate ATR-based trade levels")
    return {"success": True, "ticker": ticker.upper(), "currency": data.get("currency", "Unknown"), "current_price": data["latest_close"], "plan": build_trade_plan(data["latest_close"], atr, direction, risk_reward)}

@app.post("/api/profit-planner")
def profit_planner(request: ProfitPlanRequest):
    try:
        return {"success": True, **calculate_profit_plan(**request.model_dump())}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
