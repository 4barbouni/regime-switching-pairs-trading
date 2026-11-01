import argparse
import csv
import json
from pathlib import Path

from data import fetch_daily_prices
from research import (
    DEFAULT_PAIRS,
    combine_selected_portfolio,
    evaluate_pair,
    high_conviction_configs,
    single_digit_return_configs,
)
from visualization import write_algorithm_3d_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-vol", type=float, default=0.12)
    parser.add_argument("--max-portfolio-leverage", type=float, default=8.0)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--risk-throttle", action="store_true")
    parser.add_argument("--mode", choices=["balanced", "single-digit-return"], default="balanced")
    parser.add_argument("--objective", choices=["sharpe", "return"], default="sharpe")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cache = here / "data_cache"
    start = "20180101"
    end = "20260610"

    configs = single_digit_return_configs() if args.mode == "single-digit-return" else high_conviction_configs()
    print(f"Scanning {len(DEFAULT_PAIRS)} candidate pairs...")

    results = []
    for symbol_a, symbol_b in DEFAULT_PAIRS:
        pair = f"{symbol_a}/{symbol_b}"
        try:
            series = {
                symbol_a: fetch_daily_prices(symbol_a, start, end, str(cache)),
                symbol_b: fetch_daily_prices(symbol_b, start, end, str(cache)),
            }
            result = evaluate_pair(symbol_a, symbol_b, series, configs)
        except Exception as exc:
            print(f"Skipped {pair}: {exc}")
            continue
        results.append(result)
        test = result["test"].summary
        quality = result["quality"]
        print(
            f"{pair}: test_return={test['annual_return']:.4f} "
            f"test_sharpe={test['sharpe']:.2f} "
            f"drawdown={test['max_drawdown']:.4f} "
            f"half_life={quality['half_life']:.1f}"
        )

    if not results:
        print("No pairs completed.")
        return

    portfolio = combine_selected_portfolio(
        results,
        top_n=args.top_n,
        target_vol_annual=args.target_vol,
        max_portfolio_leverage=args.max_portfolio_leverage,
        risk_throttle=args.risk_throttle,
        objective=args.objective,
    )
    write_scan_outputs(here, results, portfolio)
    model_path = write_algorithm_3d_model(here, results, portfolio)

    print("\nSelected portfolio")
    for item in portfolio["selected"]:
        print(f"- {item['pair']}")
    for key, value in portfolio["summary"].items():
        print(f"{key}: {value:.6f}")
    print(f"3d_model: {model_path.name}")


def write_scan_outputs(here: Path, results, portfolio) -> None:
    with open(here / "pair_scan_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair",
                "train_return",
                "train_sharpe",
                "validation_return",
                "validation_sharpe",
                "test_return",
                "test_sharpe",
                "test_drawdown",
                "test_final_equity",
                "test_trades",
                "quality_score",
                "return_correlation",
                "half_life",
                "residual_volatility",
                "target_vol",
                "max_leverage",
                "break_limit",
            ]
        )
        for item in sorted(results, key=lambda x: x["test"].summary["sharpe"], reverse=True):
            train = item["train"].summary
            validation = item["validation"].summary
            test = item["test"].summary
            quality = item["quality"]
            config = item["config"]
            writer.writerow(
                [
                    item["pair"],
                    f"{train['annual_return']:.10f}",
                    f"{train['sharpe']:.10f}",
                    f"{validation['annual_return']:.10f}",
                    f"{validation['sharpe']:.10f}",
                    f"{test['annual_return']:.10f}",
                    f"{test['sharpe']:.10f}",
                    f"{test['max_drawdown']:.10f}",
                    f"{test['final_equity']:.10f}",
                    f"{test['trade_count']:.0f}",
                    f"{quality['quality_score']:.10f}",
                    f"{quality['return_correlation']:.10f}",
                    f"{quality['half_life']:.10f}",
                    f"{quality['residual_volatility']:.10f}",
                    f"{config.target_vol_annual:.6f}",
                    f"{config.max_leverage:.6f}",
                    f"{config.break_probability_limit:.6f}",
                ]
            )

    with open(here / "pair_scan_portfolio_equity.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "return", "equity"])
        for date, ret, equity in zip(portfolio["dates"], portfolio["returns"], portfolio["equity"]):
            writer.writerow([date, f"{ret:.10f}", f"{equity:.10f}"])

    with open(here / "pair_scan_portfolio_weights.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair", "weight"])
        for item, weight in zip(portfolio["selected"], portfolio["weights"]):
            writer.writerow([item["pair"], f"{weight:.10f}"])

    payload = {
        "portfolio_summary": portfolio["summary"],
        "selected_pairs": [item["pair"] for item in portfolio["selected"]],
        "weights": {item["pair"]: float(weight) for item, weight in zip(portfolio["selected"], portfolio["weights"])},
    }
    with open(here / "pair_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
