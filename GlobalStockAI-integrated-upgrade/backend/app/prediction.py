
import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:
    yf = None

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from .market_data import get_stock_metadata
from .risk import build_trade_plan


HORIZONS = {
    "1_day": 1,
    "5_days": 5,
    "20_days": 20
}

NUMBER_OF_FOLDS = 5
INITIAL_TRAIN_FRACTION = 0.60

BENCHMARK_BY_EXCHANGE = {
    "Kuala Lumpur": "^KLSE",
    "SES": "^STI",
    "Singapore": "^STI",
    "HKSE": "^HSI",
    "Tokyo": "^N225",
    "XETRA": "^GDAXI",
    "NasdaqGS": "^GSPC",
    "Nasdaq": "^GSPC",
    "NMS": "^GSPC",
    "NYQ": "^GSPC",
    "NYSE": "^GSPC",
}


def get_market_benchmark(metadata):
    exchange = metadata.get("exchange", "Unknown")

    return BENCHMARK_BY_EXCHANGE.get(exchange)

def build_features(df, market_df=None):
    """Create features using historical information only."""

    data = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Historical returns
    for lag in [1, 2, 3, 5, 10, 20]:
        data[f"return_{lag}"] = close.pct_change(lag)

    # Market context
    if market_df is not None and not market_df.empty:
        market_close = market_df["Close"].reindex(
            df.index
        ).ffill()

        market_return = market_close.pct_change()

        data["market_return_1"] = market_return
        data["market_return_5"] = market_close.pct_change(5)
        data["relative_return_5"] = (
            close.pct_change(5) -
            market_close.pct_change(5)
        )
        
    # Moving average distances
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    data["sma20_distance"] = close / sma20 - 1
    data["sma50_distance"] = close / sma50 - 1

    # RSI
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
    data["rsi"] = 100 - (100 / (1 + rs))

    data.loc[
        (avg_loss == 0) & (avg_gain > 0), "rsi"
    ] = 100

    data.loc[
        (avg_loss == 0) & (avg_gain == 0), "rsi"
    ] = 50

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    data["macd_distance"] = macd / close
    data["macd_histogram"] = (macd - signal) / close

    # Volatility
    daily_return = close.pct_change()

    data["volatility_10"] = daily_return.rolling(10).std()
    data["volatility_20"] = daily_return.rolling(20).std()

    # Average True Range
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    data["atr_percentage"] = atr / close

    # Volume
    average_volume = volume.rolling(20).mean()
    data["volume_ratio"] = volume / average_volume

    return data.replace(
        [np.inf, -np.inf], np.nan
    )



def classify_direction(return_value, threshold):
    """
    Classify a future return using a volatility-adjusted threshold.

    UP: return exceeds the positive threshold.
    DOWN: return falls below the negative threshold.
    SIDEWAYS: return remains between both thresholds.
    """

    if pd.isna(return_value) or pd.isna(threshold):
        return np.nan

    if return_value > threshold:
        return 2

    elif return_value < -threshold:
        return 0

    return 1


def get_direction_model():
    """Create a classification model."""

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        max_features=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


def validate_direction_model(X, y, horizon):
    """
    Evaluate direction predictions using chronological holdout data.
    Purge overlapping future labels before training.
    """

    total = len(X)
    holdout_start = int(total * 0.80)

    # Prevent training labels from overlapping the test period.
    train_end = holdout_start - horizon

    if train_end < 50 or total - holdout_start < 10:
        return {
            "success": False,
            "reason": "Not enough data for reliable direction validation."
        }

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[holdout_start:]
    y_test = y.iloc[holdout_start:]

    if y_train.nunique() < 2:
        return {
            "success": False,
            "reason": "Training data contains fewer than two classes."
        }

    model = get_direction_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    model_accuracy = accuracy_score(y_test, predictions)
    model_balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )

    majority_class = y_train.mode().iloc[0]

    baseline_predictions = np.full(
        len(y_test), majority_class
    )

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )

    baseline_balanced_accuracy = balanced_accuracy_score(
        y_test, baseline_predictions
    )

    class_names = {
        0: "DOWN",
        1: "SIDEWAYS",
        2: "UP"
    }

    present_labels = sorted(
        set(y_test.astype(int)) |
        set(predictions.astype(int))
    )

    report = classification_report(
        y_test,
        predictions,
        labels=present_labels,
        target_names=[
            class_names[label]
            for label in present_labels
        ],
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )

    return {
        "success": True,

        "accuracy_percent": round(
            model_accuracy * 100, 2
        ),

        "balanced_accuracy_percent": round(
            model_balanced_accuracy * 100, 2
        ),

        "baseline_accuracy_percent": round(
            baseline_accuracy * 100, 2
        ),

        "baseline_balanced_accuracy_percent": round(
            baseline_balanced_accuracy * 100, 2
        ),

        "beats_baseline": (
            model_accuracy > baseline_accuracy
            and
            model_balanced_accuracy > baseline_balanced_accuracy
        ),

        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "holdout_records": len(y_test)
    }

def get_models():
    """Create fresh models for each training period."""

    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            verbosity=-1,
            random_state=42,
            n_jobs=-1
        )
    }

def walk_forward_validation(X, y, horizon):
    """Expanding-window validation with a final untouched chronological holdout.

    A failed model/fold is recorded and does not prevent other models from
    being evaluated. A zero-return benchmark is always included.
    """
    total = len(X)
    if total < 300:
        return {"success": False, "message": "Insufficient observations for validation."}
    holdout_start = int(total * 0.80)
    initial_train_end = int(holdout_start * INITIAL_TRAIN_FRACTION)
    remaining = holdout_start - initial_train_end
    if remaining < NUMBER_OF_FOLDS * 10:
        return {"success": False, "message": "Insufficient observations for model selection."}
    fold_size = max(1, remaining // NUMBER_OF_FOLDS)
    model_names = list(get_models().keys())
    fold_actual = []
    fold_predictions = {name: [] for name in model_names}
    fold_errors = {name: [] for name in model_names}
    for fold in range(NUMBER_OF_FOLDS):
        test_start = initial_train_end + fold * fold_size
        test_end = holdout_start if fold == NUMBER_OF_FOLDS - 1 else min(holdout_start, test_start + fold_size)
        train_end = test_start - horizon
        if train_end < 50 or test_end <= test_start:
            continue
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
        fold_actual.extend(y_test.tolist())
        for name, model in get_models().items():
            try:
                model.fit(X_train, y_train)
                fold_predictions[name].extend(model.predict(X_test).tolist())
            except Exception as exc:
                fold_errors[name].append(f"fold {fold + 1}: {type(exc).__name__}: {exc}")
    comparisons = {}
    actual = np.asarray(fold_actual, dtype=float)
    for name, values in fold_predictions.items():
        predicted = np.asarray(values, dtype=float)
        if len(predicted) != len(actual) or not len(actual) or fold_errors[name]:
            continue
        comparisons[name] = {
            "mae_percent": round(float(mean_absolute_error(actual, predicted) * 100), 3),
            "directional_accuracy_percent": round(float(np.mean(np.sign(actual) == np.sign(predicted)) * 100), 2)
        }
    if not comparisons:
        return {"success": False, "message": "All candidate models failed during walk-forward validation.", "model_errors": fold_errors}
    best_name = min(comparisons, key=lambda name: comparisons[name]["mae_percent"])
    train_end = holdout_start - horizon
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_holdout, y_holdout = X.iloc[holdout_start:], y.iloc[holdout_start:]
    if X_holdout.empty:
        return {"success": False, "message": "Final holdout is empty."}
    final_model = get_models()[best_name]
    try:
        final_model.fit(X_train, y_train)
        holdout_pred = final_model.predict(X_holdout)
    except Exception as exc:
        return {"success": False, "message": f"Final model fit failed: {exc}", "model_errors": fold_errors}
    holdout_mae = float(mean_absolute_error(y_holdout, holdout_pred))
    baseline_mae = float(mean_absolute_error(y_holdout, np.zeros(len(y_holdout))))
    direction_accuracy = float(np.mean(np.sign(y_holdout) == np.sign(holdout_pred)) * 100)
    return {
        "success": True, "folds": NUMBER_OF_FOLDS, "observations": len(y_holdout),
        "selection_observations": len(actual), "baseline_mae_percent": round(baseline_mae * 100, 3),
        "holdout_mae_percent": round(holdout_mae * 100, 3),
        "holdout_directional_accuracy_percent": round(direction_accuracy, 2),
        "model_comparison": comparisons, "best_model": best_name,
        "best_model_beats_baseline": bool(holdout_mae < baseline_mae),
        "model_errors": fold_errors
    }


def predict_stock(ticker: str, period: str = "5y"):
    """
    Compare regression and classification models.

    Regression estimates future price movement.
    Classification estimates UP, DOWN or SIDEWAYS probabilities.

    Predictions are experimental estimates, not guaranteed
    trading outcomes.
    """

    ticker = (ticker or "").strip().upper()

    if not ticker:
        return {
            "success": False,
            "message": "A stock ticker is required."
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

        if df.empty or len(df) < 300:
            return {
                "success": False,
                "ticker": ticker,
                "message": "Insufficient historical data."
            }

        df = df.dropna(
            subset=["Close", "High", "Low", "Volume"]
        )

        # Select a benchmark from the actual listing exchange. If no reliable
        # mapping exists, omit market-relative features rather than use SPY
        # indiscriminately for every international listing.
        metadata = get_stock_metadata(ticker)
        market_ticker = get_market_benchmark(metadata)
        market_df = pd.DataFrame()
        if market_ticker:
            try:
                market_df = yf.Ticker(market_ticker).history(
                    period=period, interval="1d", auto_adjust=True
                )
            except Exception:
                market_df = pd.DataFrame()

        # Generate stock features with market context.
        features = build_features(
            df,
            market_df=market_df
        )

        clean_features = features.replace([np.inf, -np.inf], np.nan).dropna()
        if clean_features.empty:
            return {"success": False, "ticker": ticker, "message": "No complete feature row is available."}
        latest_features = clean_features.iloc[[-1]]

        if latest_features.isna().any().any():
            return {
                "success": False,
                "ticker": ticker,
                "message": (
                    "Latest price observation has "
                    "incomplete features."
                )
            }

        latest_price = float(df["Close"].iloc[-1])

        results = {}

        for horizon_name, horizon in HORIZONS.items():

            target = (
                df["Close"].shift(-horizon) / df["Close"]
            ) - 1

            # Prepare historical training data.
            dataset = features.copy()
            dataset["target"] = target

            dataset = dataset.replace(
                [np.inf, -np.inf], np.nan
            )

            # Keep only observations with known future returns.
            training_dataset = dataset.dropna()

            if len(training_dataset) < 300:
                results[horizon_name] = {
                    "success": False,
                    "message": (
                        "Insufficient training observations."
                    )
                }
                continue

            X = training_dataset.drop(columns=["target"])
            y = training_dataset["target"]

            # ----------------------------------
            # ENGINE A: PRICE MOVEMENT REGRESSION
            # ----------------------------------

            validation = walk_forward_validation(
                X,
                y,
                horizon
            )

            if not validation["success"]:
                results[horizon_name] = validation
                continue

            best_model_name = validation["best_model"]

            # Train the selected model using all available
            # historical features with known future targets.
            final_model = get_models()[best_model_name]

            final_model.fit(X, y)

            # Predict using the latest available features.
            predicted_return = float(
                final_model.predict(latest_features)[0]
            )

            predicted_price = (
                latest_price * (1 + predicted_return)
            )

            regression_beats_baseline = validation[
                "best_model_beats_baseline"
            ]

            # ----------------------------------
            # ENGINE B: DIRECTION CLASSIFICATION
            # ----------------------------------

            # Calculate historical volatility using information
            # available at each prediction date.
            daily_returns = df["Close"].pct_change()

            historical_volatility = (
                daily_returns.rolling(20).std()
                * np.sqrt(horizon)
            )

            # Use half of the expected horizon volatility
            # as the classification threshold.
            direction_threshold = (
                historical_volatility.reindex(X.index) * 0.5
            )

            direction_y = pd.Series(
                [
                    classify_direction(return_value, threshold)
                    for return_value, threshold in zip(
                        y,
                        direction_threshold
                    )
                ],
                index=X.index,
                dtype="float"
            )

            # Remove observations without a valid threshold.
            valid_directions = direction_y.notna()

            X_direction = X.loc[valid_directions]
            y_direction = direction_y.loc[valid_directions]

            direction_validation = None

            direction_probabilities = {
                "DOWN": 0.0,
                "SIDEWAYS": 0.0,
                "UP": 0.0
            }

            raw_direction = "UNKNOWN"
            direction_beats_baseline = False

            try:
                direction_validation = (
                    validate_direction_model(
                        X_direction,
                        y_direction,
                        horizon
                    )
                )

                if direction_validation["success"]:
                    direction_model = get_direction_model()

                    direction_model.fit(
                        X_direction,
                        y_direction
                    )

                    probabilities = (
                        direction_model.predict_proba(
                            latest_features
                        )[0]
                    )

                    class_names = {
                        0: "DOWN",
                        1: "SIDEWAYS",
                        2: "UP"
                    }

                    for class_id, probability in zip(
                        direction_model.classes_,
                        probabilities
                    ):
                        direction_name = class_names.get(
                            int(class_id)
                        )

                        if direction_name:
                            direction_probabilities[
                                direction_name
                            ] = round(
                                float(probability) * 100,
                                2
                            )

                    predicted_class = int(
                        direction_model.predict(
                            latest_features
                        )[0]
                    )

                    raw_direction = class_names.get(
                        predicted_class,
                        "UNKNOWN"
                    )

                    direction_beats_baseline = (
                        direction_validation["beats_baseline"]
                    )

            except Exception as direction_error:
                print(
                    f"Direction model error for "
                    f"{ticker} ({horizon_name}): "
                    f"{direction_error}"
                )

                direction_validation = {
                    "success": False,
                    "message": str(direction_error)
                }

            # Only show a reliable direction signal when
            # the classifier beats its validation baseline.
            if direction_beats_baseline:
                predicted_direction = raw_direction
            else:
                predicted_direction = "NO RELIABLE SIGNAL"

            # ATR-based scenario levels are risk controls, not model price targets.
            prev_close = df["Close"].shift(1)
            tr = pd.concat([(df["High"] - df["Low"]), (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
            atr_latest = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1]
            trade_plan = build_trade_plan(latest_price, float(atr_latest), "LONG", 2.0, 1.5) if pd.notna(atr_latest) and atr_latest > 0 else None
            regression_reliable = bool(regression_beats_baseline)
            direction_reliable = bool(direction_beats_baseline)
            if regression_reliable and direction_reliable:
                reliability_label = "BOTH_MODELS_BEAT_BASELINE"
            elif regression_reliable or direction_reliable:
                reliability_label = "PARTIAL_EVIDENCE"
            else:
                reliability_label = "LOW_RELIABILITY"

            results[horizon_name] = {
                "success": True,
                "horizon_days": horizon,

                "current_price": round(
                    latest_price, 4
                ),

                "predicted_return_percent": round(
                    predicted_return * 100, 3
                ),

                "predicted_price": round(
                    predicted_price, 4
                ),

                "predicted_direction": predicted_direction,
                "raw_direction_prediction": raw_direction,

                "direction_probabilities": (
                    direction_probabilities
                ),

                "selected_model": best_model_name,

                "regression_reliable": (
                    regression_beats_baseline
                ),

                "direction_reliable": direction_reliable,
                "reliability_label": reliability_label,
                "reliability_note": "Both forecasts and validation metrics are shown; a baseline win is historical evidence, not a guarantee.",
                "trade_plan": trade_plan,

                "regression_validation": validation,

                "direction_validation": (
                    direction_validation
                )
            }

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
            "exchange": metadata.get("exchange", "Unknown"),
            "benchmark": market_ticker,
            "latest_date": str(df.index[-1]),
            "prediction_method": (
                "Regression and direction classification"
            ),
            "training_period": period,
            "results": results,
            "warning": (
                "Forecasts are experimental estimates. "
                "Historical validation does not guarantee "
                "future accuracy or trading profits."
            )
        }

    except Exception as error:
        print(
            f"Prediction error for {ticker}: {error}"
        )

        return {
            "success": False,
            "ticker": ticker,
            "message": (
                "Unable to generate stock prediction."
            ),
            "error": str(error)
        }
