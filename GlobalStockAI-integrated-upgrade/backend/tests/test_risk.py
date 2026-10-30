import pytest
from app.risk import build_trade_plan, calculate_profit_plan

def test_atr_trade_plan_long_levels():
    plan = build_trade_plan(100, 2, "LONG", 2, 1.5)
    assert plan["stop_loss"] < plan["entry_price"] < plan["take_profit"]
    assert plan["risk_reward_ratio"] == 2

def test_atr_trade_plan_short_levels():
    plan = build_trade_plan(100, 2, "SHORT", 2, 1.5)
    assert plan["take_profit"] < plan["entry_price"] < plan["stop_loss"]

def test_profit_planner_sizes_by_risk_and_capital():
    result = calculate_profit_plan(10000, 100, 97, 106, 1, 0.1, 4.5)
    assert result["quantity"] == 33
    assert result["capital_used"] == 14850
    assert result["net_profit_myr"] > 0

def test_profit_planner_rejects_same_entry_and_stop():
    with pytest.raises(ValueError):
        calculate_profit_plan(1000, 10, 10, 12)
