
try:
    import yfinance as yf
except ImportError:
    yf = None
import pandas as pd
from .market_data import get_stock_metadata


def calculate_indicators(ticker: str, period: str = "1y"):
    """
    Calculate technical indicators using adjusted historical prices.
    """

    ticker = (ticker or "").strip().upper()

    if not ticker:
        return {
            "success": False,
            "message": "A stock ticker is required."
        }

    allowed_periods = {
        "3mo", "6mo", "1y", "2y", "5y", "10y", "max"
    }

    if period not in allowed_periods:
        return {
            "success": False,
            "message": "Unsupported historical period."
        }

    if yf is None:
        return {"success": False, "ticker": ticker, "message": "yfinance is not installed."}
    try:
        stock = yf.Ticker(ticker)

        df = stock.history(
            period=period,
            interval="1d",
            auto_adjust=True
        )

        if df.empty:
            return {
                "success": False,
                "ticker": ticker,
                "message": "No historical data available."
            }

        close = df["Close"]

        # Moving averages
        df["SMA_20"] = close.rolling(window=20).mean()
        df["SMA_50"] = close.rolling(window=50).mean()

        # Exponential moving averages
        df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
        df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

        # RSI (14 periods)
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss

        df["RSI_14"] = 100 - (100 / (1 + rs))

        # Handle stocks with no losses or no gains.
        df.loc[
            (avg_loss == 0) & (avg_gain > 0),
            "RSI_14"
        ] = 100

        df.loc[
            (avg_loss == 0) & (avg_gain == 0),
            "RSI_14"
        ] = 50

        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]

        df["MACD_signal"] = df["MACD"].ewm(
            span=9,
            adjust=False
        ).mean()

        df["MACD_histogram"] = (
            df["MACD"] - df["MACD_signal"]
        )

        # Bollinger Bands
        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()

        df["BB_middle"] = rolling_mean
        df["BB_upper"] = rolling_mean + (2 * rolling_std)
        df["BB_lower"] = rolling_mean - (2 * rolling_std)

        # Average True Range (ATR 14)
        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - previous_close).abs(),
                (df["Low"] - previous_close).abs()
            ],
            axis=1
        ).max(axis=1)

        df["ATR_14"] = true_range.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()

        # Latest available indicator values
        latest = df.iloc[-1]

        indicator_columns = [
            "SMA_20",
            "SMA_50",
            "EMA_12",
            "EMA_26",
            "RSI_14",
            "MACD",
            "MACD_signal",
            "MACD_histogram",
            "BB_middle",
            "BB_upper",
            "BB_lower",
            "ATR_14"
        ]

        indicators = {}

        for column in indicator_columns:
            value = latest[column]

            indicators[column] = (
                round(float(value), 4)
                if pd.notna(value)
                else None
            )

        latest_date = df.index[-1]

        return {
            "success": True,
            "ticker": ticker,
            "currency": (get_stock_metadata(ticker).get("currency") or "Unknown"),
            "latest_date": str(latest_date),
            "latest_close": round(float(latest["Close"]), 4),
            "indicators": indicators,
            "record_count": len(df),
            "message": "Technical indicators calculated successfully."
        }

    except Exception as error:
        print(f"Indicator calculation error: {error}")

        return {
            "success": False,
            "ticker": ticker,
            "message": "Unable to calculate technical indicators.",
            "error": str(error)
        }
