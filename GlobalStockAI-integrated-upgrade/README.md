# GlobalStockAI (integrated upgrade)

A research-oriented stock analysis starter. Historical prices and model calculations remain in each listing's native trading currency; the UI converts display values and MYR-denominated capital plans using a Yahoo Finance FX quote.

## Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: `http://127.0.0.1:8000/docs`

Important endpoints:
- `GET /api/health` — dependency and prediction-engine availability.
- `GET /api/health/market-data` — live Yahoo Finance connectivity check.
- `GET /api/stocks/search?q=AAPL` — identity-first stock search.
- `GET /api/stocks/{ticker}/history` — historical data in native currency.
- `GET /api/stocks/{ticker}/indicators` — technical indicators and listing currency.
- `GET /api/stocks/{ticker}/prediction` — 1-, 5- and 20-session forecasts, validation metrics and ATR risk scenarios.
- `GET /api/fx/rate?from_currency=USD&to_currency=MYR` — cached FX rate.
- `GET /api/stocks/{ticker}/trade-plan` — ATR-based long/short scenario.
- `POST /api/profit-planner` — risk-based share sizing and estimated profit/loss.

## Frontend

```bash
cd frontend
npm ci
npm run dev
```

The Vite development proxy should forward `/api` requests to the backend. Configure the proxy target in `frontend/vite.config.js` if your backend runs on a different port.

## Model and risk disclosures

- The regression and classification forecasts are experimental, not investment advice.
- The dashboard displays the forecasts together with held-out chronological validation metrics and baseline comparisons, including when the model does not beat the baseline.
- `BOTH_MODELS_BEAT_BASELINE`, `PARTIAL_EVIDENCE` and `LOW_RELIABILITY` are evidence labels, not probabilities of future success.
- Entry is the latest available close in the illustrative plan. Stop-loss and take-profit are ATR-distance scenarios; they are not claimed to be the mathematically optimal prices.
- Historical prices and risk levels are native-currency values. MYR equivalents use a current FX quote and can be unavailable if the provider has an outage.
- The profit planner is a simplified long-position estimator. It does not yet model exchange lot sizes, taxes, stamp duty, slippage, FX spread or all broker fee schedules. Verify local market rules before trading.
- The catalogue is a small starter catalogue, not a complete verified global security master. Provider search candidates must be checked against the exchange and company identity.

## Tests

```bash
cd backend
python -m pytest -q
```
