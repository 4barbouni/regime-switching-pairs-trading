import argparse
from pathlib import Path

from alpaca_paper import AlpacaPaperClient, plan_pair_orders
from data import align_price_series, fetch_daily_prices
from research import single_digit_return_configs
from strategy import REGIMES, RegimeSwitchingPairsTrader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="BMY,LLY", help="Pair symbols, for example DAL,UAL")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260610")
    parser.add_argument("--capital-fraction", type=float, default=0.10)
    parser.add_argument("--min-order-value", type=float, default=25.0)
    parser.add_argument("--signal-only", action="store_true", help="Compute the signal without connecting to Alpaca")
    parser.add_argument("--execute", action="store_true", help="Submit orders to Alpaca paper trading")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cache = here / "data_cache"
    symbol_a, symbol_b = [item.strip().upper() for item in args.pair.split(",", 1)]

    series_a = fetch_daily_prices(symbol_a, args.start, args.end, str(cache))
    series_b = fetch_daily_prices(symbol_b, args.start, args.end, str(cache))
    dates, prices = align_price_series([series_a, series_b])

    config = single_digit_return_configs()[0]
    result = RegimeSwitchingPairsTrader(config).run(prices[symbol_a], prices[symbol_b])
    target_position = float(result.positions[-1])
    hedge_ratio = float(result.hedge_ratios[-1])
    gross = abs(1.0) + abs(hedge_ratio)
    weights = {
        symbol_a: target_position / gross,
        symbol_b: -target_position * hedge_ratio / gross,
    }
    reference_prices = {
        symbol_a: float(prices[symbol_a][-1]),
        symbol_b: float(prices[symbol_b][-1]),
    }

    dominant_regime = REGIMES[int(result.regime_probabilities[-1].argmax())]
    print(f"Pair: {symbol_a}/{symbol_b}")
    print(f"Latest data date: {dates[-1]}")
    print(f"Target pair position: {target_position:.6f}")
    print(f"Hedge ratio: {hedge_ratio:.6f}")
    print(f"Z-score: {float(result.zscores[-1]):.6f}")
    print(f"Break probability: {float(result.break_probabilities[-1]):.6f}")
    print(f"Dominant regime: {dominant_regime}")
    print(f"Target weights: {symbol_a}={weights[symbol_a]:.6f}, {symbol_b}={weights[symbol_b]:.6f}")
    if args.signal_only:
        print("Signal-only mode. No Alpaca connection attempted.")
        return

    try:
        client = AlpacaPaperClient()
    except RuntimeError as exc:
        print(exc)
        print("No orders submitted.")
        return
    account = client.account()
    positions = client.positions()
    orders = plan_pair_orders(
        [symbol_a, symbol_b],
        weights,
        account.equity,
        positions,
        reference_prices,
        args.capital_fraction,
        args.min_order_value,
    )

    print(f"Paper account equity: {account.equity:.2f}")
    print(f"Capital fraction: {args.capital_fraction:.2%}")
    if not orders:
        print("No paper orders needed; current Alpaca positions are close to target.")
        return

    print("\nPlanned paper orders")
    for order in orders:
        print(
            f"{order.side.upper()} {order.qty:.4f} {order.symbol} "
            f"delta=${order.delta_value:.2f} target=${order.target_value:.2f} "
            f"current=${order.current_value:.2f}"
        )

    if not args.execute:
        print("\nDry run only. Add --execute to submit these orders to Alpaca paper trading.")
        return

    print("\nSubmitting paper orders...")
    for order in orders:
        response = client.submit_market_order(order)
        print(f"{order.symbol}: submitted order id {response.get('id', 'unknown')}")


if __name__ == "__main__":
    main()
