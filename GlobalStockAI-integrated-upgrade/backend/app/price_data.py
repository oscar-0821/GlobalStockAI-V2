
try:
    import yfinance as yf
except ImportError:
    yf = None
import pandas as pd


# Supported historical periods
ALLOWED_PERIODS = {
    "1mo", "3mo", "6mo", "1y",
    "2y", "5y", "10y", "max"
}

# Supported chart intervals
ALLOWED_INTERVALS = {
    "1d", "1wk", "1mo"
}


def get_stock_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
):
    """
    Retrieve historical stock prices from Yahoo Finance.

    Prices are returned in the stock's native trading currency.
    No MYR conversion is performed here.
    """

    ticker = (ticker or "").strip().upper()

    if not ticker:
        return {
            "success": False,
            "message": "A stock ticker is required.",
            "data": []
        }

    if period not in ALLOWED_PERIODS:
        return {
            "success": False,
            "message": "Unsupported historical period.",
            "data": []
        }

    if interval not in ALLOWED_INTERVALS:
        return {
            "success": False,
            "message": "Unsupported chart interval.",
            "data": []
        }

    if yf is None:
        return {"success": False, "ticker": ticker, "message": "yfinance is not installed.", "data": []}
    try:
        stock = yf.Ticker(ticker)

        history = stock.history(
            period=period,
            interval=interval,
            auto_adjust=True
        )

        if history.empty:
            return {
                "success": False,
                "ticker": ticker,
                "message": "No historical price data is available.",
                "data": []
            }

        history = history.reset_index()

        # Convert historical data into JSON-friendly records.
        records = []

        for _, row in history.iterrows():
            date_value = row.get("Date")

            if pd.isna(date_value):
                date_value = row.get("Datetime")

            if pd.isna(date_value):
                continue

            records.append({
                "date": str(date_value),
                "open": round(float(row["Open"]), 4)
                    if pd.notna(row["Open"]) else None,
                "high": round(float(row["High"]), 4)
                    if pd.notna(row["High"]) else None,
                "low": round(float(row["Low"]), 4)
                    if pd.notna(row["Low"]) else None,
                "close": round(float(row["Close"]), 4)
                    if pd.notna(row["Close"]) else None,
                "volume": int(row["Volume"])
                    if pd.notna(row["Volume"]) else None
            })

        currency = "Unknown"

        try:
            info = stock.get_info()
            currency = info.get("currency") or "Unknown"
        except Exception:
            pass

        return {
            "success": True,
            "ticker": ticker,
            "currency": currency,
            "period": period,
            "interval": interval,
            "adjusted_prices": True,
            "record_count": len(records),
            "data": records,
            "message": "Historical prices retrieved successfully."
        }

    except Exception as error:
        print(f"Historical price error for {ticker}: {error}")

        return {
            "success": False,
            "ticker": ticker,
            "message": "Unable to retrieve historical prices.",
            "error": str(error),
            "data": []
        }
