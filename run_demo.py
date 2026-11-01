from pathlib import Path

from strategy import RegimeSwitchingPairsTrader, StrategyConfig, make_synthetic_pair


def main() -> None:
    here = Path(__file__).resolve().parent
    price_a, price_b, _ = make_synthetic_pair()

    config = StrategyConfig()
    trader = RegimeSwitchingPairsTrader(config)
    result = trader.run(price_a, price_b)

    result.write_csvs(
        str(here / "demo_equity_curve.csv"),
        str(here / "demo_trades.csv"),
    )
    result.write_summary(str(here / "demo_summary.json"))

    print("Regime-switching pairs trading demo complete")
    for key, value in result.summary.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()

