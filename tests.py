import numpy as np

from alpaca_paper import AlpacaPosition, plan_pair_orders
from data import PriceSeries, align_price_series
from strategy import RegimeSwitchingPairsTrader, StrategyConfig, make_synthetic_pair


def test_demo_runs():
    a, b, _ = make_synthetic_pair(n=400)
    result = RegimeSwitchingPairsTrader(StrategyConfig()).run(a, b)
    assert len(result.equity) == 400
    assert np.isfinite(result.equity).all()
    assert np.isfinite(result.hedge_ratios).all()
    assert np.isfinite(result.zscores).all()


def test_break_detector_de_risks_transition():
    a, b, _ = make_synthetic_pair(n=700)
    result = RegimeSwitchingPairsTrader(StrategyConfig()).run(a, b)
    transition_positions = np.abs(result.positions[540:590])
    calm_positions = np.abs(result.positions[650:690])
    assert np.mean(transition_positions) <= max(0.05, np.mean(calm_positions) + 0.25)


def test_price_alignment():
    dates_a = [f"2024-{day:03d}" for day in range(1, 121)]
    dates_b = [f"2024-{day:03d}" for day in range(11, 131)]
    a = PriceSeries("AAA", dates_a, np.arange(120.0))
    b = PriceSeries("BBB", dates_b, np.arange(120.0) + 200.0)
    dates, prices = align_price_series([a, b])
    assert len(dates) == 110
    assert dates[0] == "2024-011"
    assert prices["AAA"][0] == 10.0
    assert prices["BBB"][0] == 200.0


def test_alpaca_order_planner():
    orders = plan_pair_orders(
        ["AAA", "BBB"],
        {"AAA": 0.20, "BBB": -0.10},
        account_equity=100000.0,
        current_positions={"AAA": AlpacaPosition("AAA", 10.0, 1000.0)},
        reference_prices={"AAA": 50.0, "BBB": 25.0},
        capital_fraction=0.10,
        min_order_value=10.0,
    )
    assert len(orders) == 2
    assert orders[0].symbol == "AAA"
    assert orders[0].side == "buy"
    assert round(orders[0].qty, 6) == 20.0
    assert orders[1].symbol == "BBB"
    assert orders[1].side == "sell"
    assert round(orders[1].qty, 6) == 40.0


if __name__ == "__main__":
    test_demo_runs()
    test_break_detector_de_risks_transition()
    test_price_alignment()
    test_alpaca_order_planner()
    print("tests passed")
