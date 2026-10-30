"""Transparent ATR-based trade levels and position/profit planning."""

def build_trade_plan(price: float, atr: float, direction: str = "LONG", risk_reward: float = 2.0, atr_multiple: float = 1.5):
    if price <= 0 or atr <= 0:
        raise ValueError("Price and ATR must be positive.")
    direction = direction.upper()
    if direction not in {"LONG", "SHORT"}:
        direction = "LONG"
    risk_distance = atr * atr_multiple
    if direction == "LONG":
        entry, stop, target = price, price - risk_distance, price + risk_distance * risk_reward
    else:
        entry, stop, target = price, price + risk_distance, price - risk_distance * risk_reward
    return {"direction": direction, "entry_price": round(entry, 4), "stop_loss": round(stop, 4), "take_profit": round(target, 4), "risk_per_share": round(abs(entry-stop), 4), "reward_per_share": round(abs(target-entry), 4), "risk_reward_ratio": round(risk_reward, 2), "method": "ATR-based scenario; not a guaranteed optimal entry"}

def calculate_profit_plan(capital: float, entry_price: float, stop_loss: float, take_profit: float, risk_percent: float = 1.0, fee_percent: float = 0.1, fx_rate: float = 1.0):
    values = [capital, entry_price, stop_loss, take_profit, risk_percent, fee_percent, fx_rate]
    if any(float(v) < 0 for v in values) or capital <= 0 or entry_price <= 0 or fx_rate <= 0:
        raise ValueError("Capital, entry price and FX rate must be positive; percentages cannot be negative.")
    risk_budget = capital * risk_percent / 100
    per_share_risk = abs(entry_price - stop_loss)
    if per_share_risk <= 0:
        raise ValueError("Entry and stop-loss must be different.")
    qty_by_risk = int(risk_budget // per_share_risk)
    qty_by_capital = int(capital // entry_price)
    quantity = max(0, min(qty_by_risk, qty_by_capital))
    invested = quantity * entry_price
    gross_profit = quantity * (take_profit-entry_price)
    gross_loss = quantity * (stop_loss-entry_price)
    estimated_fees = invested * fee_percent / 100 * 2
    net_profit = gross_profit - estimated_fees
    net_loss = gross_loss - estimated_fees
    return {"quantity": quantity, "capital_used": round(invested*fx_rate,2), "capital_used_native": round(invested,2), "risk_budget": round(risk_budget*fx_rate,2), "gross_profit_native": round(gross_profit,2), "estimated_round_trip_fees_native": round(estimated_fees,2), "net_profit_native": round(net_profit,2), "net_loss_at_stop_native": round(net_loss,2), "net_profit_myr": round(net_profit*fx_rate,2), "net_loss_at_stop_myr": round(net_loss*fx_rate,2), "unused_capital_myr": round((capital-invested)*fx_rate,2), "currency_display": "MYR", "assumptions": "Long-only sizing; excludes tax, slippage, lot-size rules and FX spread. Negative/zero share quantities mean the risk budget is too small."}
