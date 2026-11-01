import argparse
from pathlib import Path

from data import align_price_series, fetch_daily_prices
from research import high_conviction_configs
from strategy import RegimeSwitchingPairsTrader, StrategyConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="KO,PEP", help="Pair symbols, for example XOM,CVX")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260610")
    parser.add_argument("--profile", choices=["balanced", "high"], default="high")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cache = here / "data_cache"

    symbol_a, symbol_b = [item.strip().upper() for item in args.pair.split(",", 1)]
    start = args.start
    end = args.end

    try:
        series_a = fetch_daily_prices(symbol_a, start, end, str(cache))
        series_b = fetch_daily_prices(symbol_b, start, end, str(cache))
    except RuntimeError as exc:
        print(exc)
        print("Place provider cache files in data_cache/ or rerun with network/DNS enabled.")
        return
    dates, prices = align_price_series([series_a, series_b])

    if args.profile == "high":
        config = high_conviction_configs()[1]
    else:
        config = StrategyConfig(trading_cost_bps=3.0, target_vol_annual=0.10, max_leverage=2.0)
    result = RegimeSwitchingPairsTrader(config).run(prices[symbol_a], prices[symbol_b])

    stem = f"real_{symbol_a}_{symbol_b}"
    result.write_csvs(
        str(here / f"{stem}_equity_curve.csv"),
        str(here / f"{stem}_trades.csv"),
    )
    result.write_summary(str(here / f"{stem}_summary.json"))

    with open(here / f"{stem}_dates.csv", "w", encoding="utf-8") as f:
        f.write("t,date\n")
        for i, date in enumerate(dates):
            f.write(f"{i},{date}\n")

    print(f"Real-data run complete for {symbol_a}/{symbol_b}")
    print(f"observations: {len(dates)}")
    for key, value in result.summary.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
