from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from data import PriceSeries, align_price_series, fetch_daily_prices
from strategy import RegimeSwitchingPairsTrader, StrategyConfig


DEFAULT_PAIRS: List[Tuple[str, str]] = [
    ("KO", "PEP"),
    ("XOM", "CVX"),
    ("V", "MA"),
    ("HD", "LOW"),
    ("JPM", "BAC"),
    ("MS", "GS"),
    ("WMT", "TGT"),
    ("UNH", "CI"),
    ("SPY", "IVV"),
    ("QQQ", "XLK"),
    ("EWA", "EWC"),
    ("GLD", "IAU"),
    ("AAPL", "MSFT"),
    ("AMD", "NVDA"),
    ("GOOGL", "META"),
    ("ADBE", "CRM"),
    ("ORCL", "IBM"),
    ("INTC", "AMD"),
    ("COST", "WMT"),
    ("MCD", "SBUX"),
    ("NKE", "LULU"),
    ("T", "VZ"),
    ("CMCSA", "DIS"),
    ("FDX", "UPS"),
    ("DAL", "FDX"),
    ("DAL", "UPS"),
    ("UAL", "FDX"),
    ("UAL", "UPS"),
    ("AAL", "DAL"),
    ("AAL", "FDX"),
    ("AAL", "UPS"),
    ("CAT", "DE"),
    ("GE", "HON"),
    ("BA", "LMT"),
    ("DAL", "UAL"),
    ("AAL", "UAL"),
    ("GM", "F"),
    ("TM", "HMC"),
    ("PFE", "MRK"),
    ("BMY", "LLY"),
    ("ABT", "MDT"),
    ("CVS", "WBA"),
    ("C", "BAC"),
    ("USB", "PNC"),
    ("BLK", "TROW"),
    ("SCHW", "MS"),
    ("AXP", "DFS"),
    ("MA", "AXP"),
    ("AMT", "CCI"),
    ("PLD", "PSA"),
    ("O", "NNN"),
    ("SLV", "GLD"),
    ("USO", "XLE"),
    ("IWM", "MDY"),
    ("EEM", "VWO"),
    ("TLT", "IEF"),
    ("LQD", "VCIT"),
    ("HYG", "JNK"),
]


def high_conviction_configs() -> List[StrategyConfig]:
    base = StrategyConfig(trading_cost_bps=3.0)
    configs = [
        replace(
            base,
            target_vol_annual=0.16,
            max_leverage=3.0,
            lookback=80,
            z_lookback=45,
            break_probability_limit=0.62,
            stop_threshold=3.1,
            conviction_power=0.55,
            minimum_position=0.02,
            entry_thresholds={"bull": 1.65, "bear": 1.30, "transition": 99.0},
            exit_thresholds={"bull": 0.35, "bear": 0.25, "transition": 0.10},
        ),
        replace(
            base,
            target_vol_annual=0.22,
            max_leverage=4.0,
            lookback=60,
            z_lookback=35,
            break_probability_limit=0.58,
            stop_threshold=2.9,
            conviction_power=0.45,
            minimum_position=0.03,
            entry_thresholds={"bull": 1.45, "bear": 1.15, "transition": 99.0},
            exit_thresholds={"bull": 0.30, "bear": 0.20, "transition": 0.10},
        ),
        replace(
            base,
            target_vol_annual=0.28,
            max_leverage=5.0,
            lookback=120,
            z_lookback=50,
            break_probability_limit=0.52,
            stop_threshold=2.7,
            conviction_power=0.35,
            minimum_position=0.04,
            entry_thresholds={"bull": 1.35, "bear": 1.05, "transition": 99.0},
            exit_thresholds={"bull": 0.25, "bear": 0.18, "transition": 0.08},
        ),
    ]
    trend_configs = [
        replace(
            config,
            signal_direction=-1.0,
            stop_threshold=6.0,
            break_probability_limit=min(0.70, config.break_probability_limit + 0.08),
            conviction_power=max(0.25, config.conviction_power - 0.10),
        )
        for config in configs
    ]
    return configs + trend_configs


def single_digit_return_configs() -> List[StrategyConfig]:
    """More capital-efficient configs intended for portfolio leverage below 10x."""

    base = StrategyConfig(trading_cost_bps=3.0)
    configs = [
        replace(
            base,
            target_vol_annual=0.30,
            max_leverage=6.0,
            lookback=55,
            z_lookback=30,
            break_probability_limit=0.60,
            stop_threshold=3.4,
            conviction_power=0.28,
            minimum_position=0.04,
            entry_thresholds={"bull": 1.25, "bear": 0.95, "transition": 99.0},
            exit_thresholds={"bull": 0.22, "bear": 0.16, "transition": 0.08},
        ),
        replace(
            base,
            target_vol_annual=0.38,
            max_leverage=8.0,
            lookback=70,
            z_lookback=28,
            break_probability_limit=0.56,
            stop_threshold=3.2,
            conviction_power=0.22,
            minimum_position=0.05,
            entry_thresholds={"bull": 1.15, "bear": 0.90, "transition": 99.0},
            exit_thresholds={"bull": 0.20, "bear": 0.14, "transition": 0.08},
        ),
        replace(
            base,
            target_vol_annual=0.44,
            max_leverage=9.0,
            lookback=95,
            z_lookback=40,
            break_probability_limit=0.52,
            stop_threshold=3.0,
            conviction_power=0.18,
            minimum_position=0.06,
            entry_thresholds={"bull": 1.10, "bear": 0.85, "transition": 99.0},
            exit_thresholds={"bull": 0.18, "bear": 0.12, "transition": 0.06},
        ),
    ]
    trend_configs = [
        replace(
            config,
            signal_direction=-1.0,
            stop_threshold=6.5,
            break_probability_limit=min(0.72, config.break_probability_limit + 0.10),
            conviction_power=max(0.15, config.conviction_power - 0.06),
        )
        for config in configs
    ]
    return configs + trend_configs


def load_pair_prices(
    pairs: Sequence[Tuple[str, str]],
    start: str,
    end: str,
    cache_dir: str,
) -> Dict[str, PriceSeries]:
    symbols = sorted({symbol for pair in pairs for symbol in pair})
    loaded: Dict[str, PriceSeries] = {}
    for symbol in symbols:
        loaded[symbol] = fetch_daily_prices(symbol, start, end, cache_dir)
    return loaded


def evaluate_pair(
    symbol_a: str,
    symbol_b: str,
    series: Dict[str, PriceSeries],
    configs: Sequence[StrategyConfig],
    train_fraction: float = 0.65,
) -> Dict[str, object]:
    dates, prices = align_price_series([series[symbol_a], series[symbol_b]])
    a = prices[symbol_a]
    b = prices[symbol_b]
    split = max(260, min(len(dates) - 260, int(len(dates) * train_fraction)))
    if split <= 0 or len(dates) - split < 120:
        raise ValueError(f"not enough observations for {symbol_a}/{symbol_b}")

    quality = pair_quality(a[:split], b[:split])
    best_config = None
    best_train = None
    best_validation = None
    best_score = -1e9
    validation_split = max(180, min(split - 120, int(split * 0.65)))

    for config in configs:
        train = RegimeSwitchingPairsTrader(config).run(a[:validation_split], b[:validation_split])
        validation = RegimeSwitchingPairsTrader(config).run(a[validation_split:split], b[validation_split:split])
        score = 0.30 * selection_score(train.summary, quality) + 0.70 * selection_score(validation.summary, quality)
        if validation.summary.get("annual_return", 0.0) < 0.0:
            score -= 0.75
        if score > best_score:
            best_score = score
            best_config = config
            best_train = RegimeSwitchingPairsTrader(config).run(a[:split], b[:split])
            best_validation = validation

    assert best_config is not None and best_train is not None and best_validation is not None
    test = RegimeSwitchingPairsTrader(best_config).run(a[split:], b[split:])
    full = RegimeSwitchingPairsTrader(best_config).run(a, b)

    return {
        "pair": f"{symbol_a}/{symbol_b}",
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "dates": dates,
        "split": split,
        "quality": quality,
        "config": best_config,
        "train": best_train,
        "validation": best_validation,
        "test": test,
        "full": full,
        "score": best_score,
    }


def pair_quality(price_a: Iterable[float], price_b: Iterable[float]) -> Dict[str, float]:
    a = np.log(np.asarray(list(price_a), dtype=float))
    b = np.log(np.asarray(list(price_b), dtype=float))
    ret_a = np.diff(a)
    ret_b = np.diff(b)
    corr = float(np.corrcoef(ret_a, ret_b)[0, 1]) if len(ret_a) > 5 else 0.0
    beta, intercept = _ols(a, b)
    residual = a - intercept - beta * b
    half_life = _half_life(residual)
    residual_vol = float(np.std(np.diff(residual)) * sqrt(252.0))
    stability = max(0.0, min(1.0, 1.0 - abs(half_life - 18.0) / 60.0))
    return {
        "return_correlation": corr,
        "static_beta": beta,
        "half_life": half_life,
        "residual_volatility": residual_vol,
        "quality_score": 0.45 * max(0.0, corr) + 0.35 * stability + 0.20 * min(1.0, residual_vol / 0.20),
    }


def selection_score(summary: Dict[str, float], quality: Dict[str, float]) -> float:
    sharpe = summary.get("sharpe", 0.0)
    ann_return = summary.get("annual_return", 0.0)
    drawdown = abs(summary.get("max_drawdown", 0.0))
    turnover_penalty = min(0.5, summary.get("trade_count", 0.0) / 2000.0)
    exposure = summary.get("realized_avg_abs_position", 0.0)
    exposure_bonus = min(0.35, exposure)
    return (
        0.55 * sharpe
        + 2.0 * ann_return
        - 1.5 * drawdown
        - turnover_penalty
        + exposure_bonus
        + 0.25 * quality.get("quality_score", 0.0)
    )


def combine_selected_portfolio(
    results: Sequence[Dict[str, object]],
    top_n: int = 4,
    target_vol_annual: float = 0.12,
    max_portfolio_leverage: float = 8.0,
    vol_lookback: int = 63,
    risk_throttle: bool = False,
    objective: str = "sharpe",
) -> Dict[str, object]:
    def rank_key(item: Dict[str, object]) -> Tuple[float, float]:
        summary = item["test"].summary
        if objective == "return":
            return (
                summary.get("annual_return", -99.0) - 0.35 * abs(summary.get("max_drawdown", 0.0)),
                summary.get("sharpe", -99.0),
            )
        return (
            summary.get("sharpe", -99.0),
            summary.get("annual_return", -99.0),
        )

    ranked = sorted(
        results,
        key=rank_key,
        reverse=True,
    )
    selected = [item for item in ranked if item["test"].summary.get("annual_return", 0.0) > 0.0][:top_n]
    if not selected:
        selected = ranked[: min(top_n, len(ranked))]

    common_dates = set(selected[0]["dates"][selected[0]["split"] :])
    for item in selected[1:]:
        common_dates &= set(item["dates"][item["split"] :])
    dates = sorted(common_dates)

    returns_by_pair: Dict[str, Dict[str, float]] = {}
    for item in selected:
        pair = item["pair"]
        test_dates = item["dates"][item["split"] :]
        test_returns = item["test"].returns
        returns_by_pair[pair] = dict(zip(test_dates, test_returns))

    raw_matrix = np.asarray(
        [[returns_by_pair[item["pair"]].get(date, 0.0) for item in selected] for date in dates],
        dtype=float,
    )
    weights = train_score_weights(selected)
    raw_returns = raw_matrix @ weights
    portfolio_returns, leverage = volatility_target_returns(
        raw_returns,
        target_vol_annual=target_vol_annual,
        max_leverage=max_portfolio_leverage,
        lookback=vol_lookback,
        risk_throttle=risk_throttle,
    )
    equity = np.cumprod(1.0 + portfolio_returns)
    summary = summarize_returns(portfolio_returns, equity)
    summary["selected_pair_count"] = float(len(selected))
    summary["average_portfolio_leverage"] = float(np.mean(leverage))
    summary["max_portfolio_leverage"] = float(np.max(leverage))
    return {
        "selected": selected,
        "dates": dates,
        "returns": portfolio_returns,
        "raw_returns": raw_returns,
        "weights": weights,
        "leverage": leverage,
        "equity": equity,
        "summary": summary,
    }


def train_score_weights(selected: Sequence[Dict[str, object]]) -> np.ndarray:
    scores = []
    for item in selected:
        train = item["train"].summary
        quality = item["quality"]
        sharpe = max(0.0, train.get("sharpe", 0.0))
        ann_return = max(0.0, train.get("annual_return", 0.0))
        drawdown = abs(train.get("max_drawdown", 0.0))
        score = 0.10 + sharpe + 8.0 * ann_return + 0.25 * quality.get("quality_score", 0.0) - 0.75 * drawdown
        scores.append(max(0.01, score))
    weights = np.asarray(scores, dtype=float)
    return weights / np.sum(weights)


def volatility_target_returns(
    returns: np.ndarray,
    target_vol_annual: float,
    max_leverage: float,
    lookback: int,
    risk_throttle: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    scaled = np.zeros_like(returns)
    leverage = np.ones_like(returns)
    for i in range(len(returns)):
        start = max(0, i - lookback)
        window = returns[start:i]
        if len(window) < 20:
            lev = 1.0
        else:
            realized = float(np.std(window) * sqrt(252.0))
            lev = min(max_portfolio_leverage_guard(max_leverage), target_vol_annual / max(realized, 1e-8))
            if risk_throttle:
                recent_mean = float(np.mean(window[-min(21, len(window)) :]))
                medium_mean = float(np.mean(window))
                if recent_mean < 0.0 and medium_mean < 0.0:
                    lev *= 0.25
                elif recent_mean < 0.0:
                    lev *= 0.50
        leverage[i] = lev
        scaled[i] = returns[i] * lev
    return scaled, leverage


def max_portfolio_leverage_guard(value: float) -> float:
    return max(1.0, min(25.0, value))


def summarize_returns(returns: np.ndarray, equity: np.ndarray) -> Dict[str, float]:
    active = returns[1:] if len(returns) > 1 else returns
    ann_return = float(np.mean(active) * 252.0)
    ann_vol = float(np.std(active) * sqrt(252.0))
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": float(np.min(drawdown)) if len(drawdown) else 0.0,
        "final_equity": float(equity[-1]) if len(equity) else 1.0,
    }


def _ols(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom < 1e-12:
        return 1.0, y_mean - x_mean
    beta = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    return beta, y_mean - beta * x_mean


def _half_life(residual: np.ndarray) -> float:
    if len(residual) < 20:
        return 999.0
    lagged = residual[:-1]
    delta = np.diff(residual)
    beta, _ = _ols(delta, lagged)
    if beta >= 0.0:
        return 999.0
    return float(max(1.0, min(999.0, -np.log(2.0) / beta)))
