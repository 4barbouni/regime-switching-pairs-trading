from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from typing import Dict, Iterable, List, Optional, Tuple

import json
import numpy as np


REGIMES = ("bull", "bear", "transition")


@dataclass
class StrategyConfig:
    lookback: int = 90
    min_lookback: int = 45
    z_lookback: int = 60
    vol_lookback: int = 30
    calibration_lookback: int = 252
    beta_smoothing: float = 0.15
    target_vol_annual: float = 0.12
    max_leverage: float = 2.0
    conviction_power: float = 0.75
    minimum_position: float = 0.0
    signal_direction: float = 1.0
    min_entry_threshold: float = 1.10
    max_entry_threshold: float = 2.50
    entry_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"bull": 2.0, "bear": 1.5, "transition": 99.0}
    )
    exit_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"bull": 0.45, "bear": 0.35, "transition": 0.15}
    )
    stop_threshold: float = 3.5
    break_probability_limit: float = 0.75
    trading_cost_bps: float = 2.0
    random_seed: int = 7


@dataclass
class BacktestResult:
    returns: np.ndarray
    equity: np.ndarray
    positions: np.ndarray
    hedge_ratios: np.ndarray
    zscores: np.ndarray
    regime_probabilities: np.ndarray
    break_probabilities: np.ndarray
    trades: List[Dict[str, float]]
    summary: Dict[str, float]

    def write_csvs(self, equity_path: str, trades_path: str) -> None:
        with open(equity_path, "w", encoding="utf-8") as f:
            f.write("t,return,equity,position,hedge_ratio,zscore,break_probability,bull,bear,transition\n")
            for i in range(len(self.equity)):
                probs = self.regime_probabilities[i]
                f.write(
                    f"{i},{self.returns[i]:.10f},{self.equity[i]:.10f},"
                    f"{self.positions[i]:.6f},{self.hedge_ratios[i]:.6f},"
                    f"{self.zscores[i]:.6f},{self.break_probabilities[i]:.6f},"
                    f"{probs[0]:.6f},{probs[1]:.6f},{probs[2]:.6f}\n"
                )

        with open(trades_path, "w", encoding="utf-8") as f:
            f.write("t,action,position,zscore,break_probability,dominant_regime\n")
            for trade in self.trades:
                f.write(
                    f"{trade['t']},{trade['action']},{trade['position']:.6f},"
                    f"{trade['zscore']:.6f},{trade['break_probability']:.6f},"
                    f"{trade['dominant_regime']}\n"
                )

    def write_summary(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, sort_keys=True)


def _safe_std(x: np.ndarray) -> float:
    value = float(np.std(x))
    return value if value > 1e-10 else 1e-10


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _rolling_ols_beta(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom < 1e-12:
        return 1.0, y_mean - x_mean
    beta = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = y_mean - beta * x_mean
    return beta, intercept


class ContinuousJumpRegimeModel:
    """Continuous jump filter that produces bull/bear/transition probabilities."""

    def infer(self, pair_returns: np.ndarray, t: int, vol_lookback: int) -> np.ndarray:
        start = max(1, t - vol_lookback)
        window = pair_returns[start : t + 1]
        if len(window) < 5:
            return np.array([0.34, 0.33, 0.33])

        recent = float(window[-1])
        drift = float(np.mean(window))
        vol = _safe_std(window)
        shock = abs(recent - drift) / vol
        jump_prob = _normal_cdf(shock - 2.25)
        trend_score = drift / (vol / sqrt(max(len(window), 1)))

        bull = 1.1 * trend_score - 1.0 * jump_prob
        bear = -1.1 * trend_score + 0.35 * jump_prob
        transition = 2.2 * jump_prob + 0.35 * abs(trend_score)
        return _softmax(np.array([bull, bear, transition]))


class XGBoostRegimeForecaster:
    """
    Optional XGBoost forecasting adapter.

    Pass a fitted classifier with `predict_proba` to use a real XGBoost model. The
    fallback keeps the demo runnable in minimal environments and preserves the same
    probabilities interface for the trading engine.
    """

    def __init__(self, model=None):
        self.model = model
        self.fallback = MarkovRegimeForecaster()

    def forecast(self, current_probabilities: np.ndarray, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            return self.fallback.forecast(current_probabilities, features)
        x = np.concatenate([current_probabilities, features]).reshape(1, -1)
        probabilities = np.asarray(self.model.predict_proba(x)[0], dtype=float)
        if probabilities.shape[0] != 3:
            raise ValueError("regime forecaster must return 3 probabilities")
        blended = 0.35 * current_probabilities + 0.65 * probabilities
        return blended / np.sum(blended)


class MarkovRegimeForecaster:
    """
    Forecasting layer.

    In production, replace this with XGBoost trained on lagged jump probabilities,
    spread features, cross-volatility, and macro state. The interface is intentionally
    small: consume current probabilities and return next-step probabilities.
    """

    transition_matrix = np.array(
        [
            [0.90, 0.04, 0.06],
            [0.05, 0.88, 0.07],
            [0.24, 0.24, 0.52],
        ]
    )

    def forecast(self, current_probabilities: np.ndarray, features: np.ndarray) -> np.ndarray:
        forecast = current_probabilities @ self.transition_matrix
        feature_tilt = np.array([features[0], -features[0], abs(features[1])])
        return _softmax(np.log(forecast + 1e-12) + 0.20 * feature_tilt)


class RegimeWeightedCointegration:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.last_beta = 1.0

    def estimate(
        self,
        log_a: np.ndarray,
        log_b: np.ndarray,
        regime_probabilities: np.ndarray,
        t: int,
    ) -> Tuple[float, float, float]:
        cfg = self.config
        start = max(0, t - cfg.lookback + 1)
        y = log_a[start : t + 1]
        x = log_b[start : t + 1]
        probs = regime_probabilities[start : t + 1]

        if len(y) < cfg.min_lookback:
            return self.last_beta, 0.0, 1.0

        dominant = int(np.argmax(regime_probabilities[t]))
        weights = probs[:, dominant]
        weights = 0.15 + weights / max(float(np.max(weights)), 1e-8)
        wx = np.average(x, weights=weights)
        wy = np.average(y, weights=weights)
        denom = float(np.sum(weights * (x - wx) ** 2))
        if denom < 1e-12:
            beta = self.last_beta
            intercept = wy - beta * wx
        else:
            beta = float(np.sum(weights * (x - wx) * (y - wy)) / denom)
            intercept = float(wy - beta * wx)

        beta = (1.0 - cfg.beta_smoothing) * self.last_beta + cfg.beta_smoothing * beta
        self.last_beta = beta

        residual = y - intercept - beta * x
        error_correction = self._error_correction_strength(residual)
        return beta, intercept, error_correction

    @staticmethod
    def _error_correction_strength(residual: np.ndarray) -> float:
        if len(residual) < 12:
            return 0.0
        lagged = residual[:-1]
        delta = np.diff(residual)
        beta, _ = _rolling_ols_beta(delta, lagged)
        return max(0.0, min(1.0, -beta))


class CointegrationBreakDetector:
    def __init__(self):
        self.reference_mean = 0.0
        self.cusum = 0.0

    def probability(
        self,
        residual_window: np.ndarray,
        beta_window: np.ndarray,
        error_correction: float,
    ) -> float:
        if len(residual_window) < 20:
            return 0.5

        recent = residual_window[-1]
        scale = _safe_std(residual_window)
        self.reference_mean = 0.98 * self.reference_mean + 0.02 * float(np.mean(residual_window))
        self.cusum = max(0.0, 0.94 * self.cusum + (abs(recent - self.reference_mean) / scale - 1.45))
        cusum_score = min(1.0, self.cusum / 10.0)

        beta_drift = 0.0
        if len(beta_window) >= 10:
            beta_drift = abs(float(beta_window[-1] - np.median(beta_window))) / (_safe_std(beta_window) + 0.05)
            beta_drift = min(1.0, beta_drift / 3.0)

        mean_shift = 0.0
        variance_shift = 0.0
        if len(residual_window) >= 40:
            first = residual_window[: len(residual_window) // 2]
            second = residual_window[len(residual_window) // 2 :]
            pooled = sqrt(_safe_std(first) ** 2 / len(first) + _safe_std(second) ** 2 / len(second))
            mean_shift = min(1.0, abs(float(np.mean(second) - np.mean(first))) / (3.0 * pooled + 1e-12))
            variance_ratio = max(_safe_std(first), _safe_std(second)) / min(_safe_std(first), _safe_std(second))
            variance_shift = min(1.0, max(0.0, variance_ratio - 1.5) / 2.5)

        weak_error_correction = max(0.0, 1.0 - min(1.0, 5.0 * error_correction))
        score = (
            0.30 * cusum_score
            + 0.24 * beta_drift
            + 0.20 * mean_shift
            + 0.14 * variance_shift
            + 0.12 * weak_error_correction
        )
        return float(max(0.0, min(1.0, score)))


class RegimeSwitchingPairsTrader:
    def __init__(self, config: Optional[StrategyConfig] = None, regime_forecaster_model=None):
        self.config = config or StrategyConfig()
        self.jump_model = ContinuousJumpRegimeModel()
        self.forecaster = XGBoostRegimeForecaster(regime_forecaster_model)
        self.coint = RegimeWeightedCointegration(self.config)
        self.break_detector = CointegrationBreakDetector()

    def run(self, price_a: Iterable[float], price_b: Iterable[float]) -> BacktestResult:
        a = np.asarray(list(price_a), dtype=float)
        b = np.asarray(list(price_b), dtype=float)
        if len(a) != len(b) or len(a) < self.config.min_lookback + 5:
            raise ValueError("price_a and price_b must be same-length arrays with enough observations")
        if np.any(a <= 0) or np.any(b <= 0):
            raise ValueError("prices must be positive")

        n = len(a)
        log_a = np.log(a)
        log_b = np.log(b)
        pair_returns = 0.5 * (np.diff(log_a, prepend=log_a[0]) + np.diff(log_b, prepend=log_b[0]))

        returns = np.zeros(n)
        equity = np.ones(n)
        positions = np.zeros(n)
        hedge_ratios = np.ones(n)
        zscores = np.zeros(n)
        regime_probs = np.zeros((n, 3))
        regime_probs[0] = np.array([0.34, 0.33, 0.33])
        break_probs = np.ones(n)
        residuals = np.zeros(n)
        trades: List[Dict[str, float]] = []

        current_position = 0.0
        previous_weight_a = 0.0
        previous_weight_b = 0.0

        for t in range(1, n):
            ret_a = log_a[t] - log_a[t - 1]
            ret_b = log_b[t] - log_b[t - 1]
            beta_for_return = hedge_ratios[t - 1] if hedge_ratios[t - 1] != 0.0 else 1.0
            gross_for_return = abs(1.0) + abs(beta_for_return)
            held_weight_a = current_position / gross_for_return
            held_weight_b = -current_position * beta_for_return / gross_for_return

            jump_prob = self.jump_model.infer(pair_returns, t, self.config.vol_lookback)
            feature_window = pair_returns[max(1, t - self.config.vol_lookback) : t + 1]
            features = np.array([
                float(np.mean(feature_window) / (_safe_std(feature_window) + 1e-12)),
                float(pair_returns[t] / (_safe_std(feature_window) + 1e-12)),
            ])
            regime_probs[t] = self.forecaster.forecast(jump_prob, features)

            beta, intercept, ec_strength = self.coint.estimate(log_a, log_b, regime_probs, t)
            hedge_ratios[t] = beta
            residuals[t] = log_a[t] - intercept - beta * log_b[t]

            z_start = max(0, t - self.config.z_lookback + 1)
            spread_window = residuals[z_start : t + 1]
            zscores[t] = (residuals[t] - float(np.mean(spread_window))) / _safe_std(spread_window)

            break_probs[t] = self.break_detector.probability(
                spread_window,
                hedge_ratios[max(0, t - self.config.lookback + 1) : t + 1],
                ec_strength,
            )

            dominant = REGIMES[int(np.argmax(regime_probs[t]))]
            target_position = self._target_position(
                current_position,
                zscores[t],
                dominant,
                break_probs[t],
                regime_probs[t],
                np.diff(residuals[max(0, t - self.config.vol_lookback) : t + 1]),
                self._adaptive_entry_threshold(zscores, regime_probs, t, dominant),
            )

            if target_position != current_position:
                trades.append(
                    {
                        "t": float(t),
                        "action": "enter" if current_position == 0.0 and target_position != 0.0 else "adjust_or_exit",
                        "position": float(target_position),
                        "zscore": float(zscores[t]),
                        "break_probability": float(break_probs[t]),
                        "dominant_regime": dominant,
                    }
                )

            current_position = target_position
            positions[t] = current_position

            gross = abs(1.0) + abs(beta)
            weight_a = current_position / gross
            weight_b = -current_position * beta / gross
            turnover = abs(weight_a - previous_weight_a) + abs(weight_b - previous_weight_b)
            cost = turnover * self.config.trading_cost_bps / 10000.0
            returns[t] = held_weight_a * ret_a + held_weight_b * ret_b - cost
            equity[t] = equity[t - 1] * (1.0 + returns[t])
            previous_weight_a = weight_a
            previous_weight_b = weight_b

        summary = self._summary(returns, equity, trades, break_probs, regime_probs)
        summary.update(self._diagnostics(returns, positions, break_probs, regime_probs))
        return BacktestResult(
            returns=returns,
            equity=equity,
            positions=positions,
            hedge_ratios=hedge_ratios,
            zscores=zscores,
            regime_probabilities=regime_probs,
            break_probabilities=break_probs,
            trades=trades,
            summary=summary,
        )

    def _target_position(
        self,
        current_position: float,
        zscore: float,
        dominant_regime: str,
        break_probability: float,
        probabilities: np.ndarray,
        recent_returns: np.ndarray,
        adaptive_entry: float,
    ) -> float:
        cfg = self.config
        entry = adaptive_entry
        exit_level = cfg.exit_thresholds[dominant_regime]

        if break_probability > cfg.break_probability_limit or dominant_regime == "transition":
            return 0.0

        size = self._position_size(probabilities, recent_returns, break_probability)
        if current_position != 0.0:
            if abs(zscore) <= exit_level or abs(zscore) >= cfg.stop_threshold:
                return 0.0
            return (1.0 if current_position > 0 else -1.0) * size

        if abs(zscore) < entry or abs(zscore) >= cfg.stop_threshold:
            return 0.0

        direction = (-1.0 if zscore > 0 else 1.0) * cfg.signal_direction
        conviction = self._signal_conviction(abs(zscore), entry, cfg.stop_threshold)
        if size * conviction < cfg.minimum_position:
            return 0.0
        return direction * size * conviction

    def _position_size(
        self,
        probabilities: np.ndarray,
        recent_returns: np.ndarray,
        break_probability: float,
    ) -> float:
        cfg = self.config
        vol = _safe_std(recent_returns) * sqrt(252.0)
        vol_scale = min(cfg.max_leverage, cfg.target_vol_annual / max(vol, 1e-6))
        regime_confidence = float(np.max(probabilities))
        break_scale = max(0.0, 1.0 - 0.75 * break_probability)
        return min(cfg.max_leverage, vol_scale * regime_confidence * break_scale)

    def _signal_conviction(self, abs_zscore: float, entry: float, stop: float) -> float:
        cfg = self.config
        width = max(stop - entry, 1e-6)
        raw = max(0.0, min(1.0, (abs_zscore - entry) / width))
        return max(0.15, raw ** cfg.conviction_power)

    def _adaptive_entry_threshold(
        self,
        zscores: np.ndarray,
        regime_probabilities: np.ndarray,
        t: int,
        dominant_regime: str,
    ) -> float:
        cfg = self.config
        base = cfg.entry_thresholds[dominant_regime]
        if dominant_regime == "transition" or t < cfg.calibration_lookback:
            return base

        start = max(0, t - cfg.calibration_lookback)
        dominant_history = np.argmax(regime_probabilities[start:t], axis=1)
        regime_id = REGIMES.index(dominant_regime)
        candidates = np.abs(zscores[start:t][dominant_history == regime_id])
        candidates = candidates[np.isfinite(candidates)]
        candidates = candidates[candidates > 0.1]
        if len(candidates) < 30:
            return base

        calibrated = float(np.quantile(candidates, 0.86 if dominant_regime == "bear" else 0.90))
        return max(cfg.min_entry_threshold, min(cfg.max_entry_threshold, 0.50 * base + 0.50 * calibrated))

    @staticmethod
    def _summary(
        returns: np.ndarray,
        equity: np.ndarray,
        trades: List[Dict[str, float]],
        break_probs: np.ndarray,
        regime_probs: np.ndarray,
    ) -> Dict[str, float]:
        active_returns = returns[1:]
        ann_return = float(np.mean(active_returns) * 252.0)
        ann_vol = float(_safe_std(active_returns) * sqrt(252.0))
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        downside = active_returns[active_returns < 0.0]
        downside_vol = float(_safe_std(downside) * sqrt(252.0)) if len(downside) else 0.0
        sortino = ann_return / downside_vol if downside_vol > 0 else 0.0
        calmar = ann_return / abs(float(np.min(drawdown))) if np.min(drawdown) < 0 else 0.0
        dominant_counts = np.bincount(np.argmax(regime_probs, axis=1), minlength=3)
        return {
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "max_drawdown": float(np.min(drawdown)),
            "final_equity": float(equity[-1]),
            "trade_count": float(len(trades)),
            "average_break_probability": float(np.mean(break_probs)),
            "bull_fraction": float(dominant_counts[0] / len(regime_probs)),
            "bear_fraction": float(dominant_counts[1] / len(regime_probs)),
            "transition_fraction": float(dominant_counts[2] / len(regime_probs)),
        }

    @staticmethod
    def _diagnostics(
        returns: np.ndarray,
        positions: np.ndarray,
        break_probs: np.ndarray,
        regime_probs: np.ndarray,
    ) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {}
        dominant = np.argmax(regime_probs, axis=1)
        for i, name in enumerate(REGIMES):
            mask = dominant == i
            if np.any(mask):
                r = returns[mask]
                diagnostics[f"{name}_annual_return"] = float(np.mean(r) * 252.0)
                diagnostics[f"{name}_avg_abs_position"] = float(np.mean(np.abs(positions[mask])))

        buckets = {
            "low_break": break_probs < 0.35,
            "medium_break": (break_probs >= 0.35) & (break_probs < 0.70),
            "high_break": break_probs >= 0.70,
        }
        for name, mask in buckets.items():
            if np.any(mask):
                diagnostics[f"{name}_annual_return"] = float(np.mean(returns[mask]) * 252.0)
                diagnostics[f"{name}_avg_abs_position"] = float(np.mean(np.abs(positions[mask])))
        diagnostics["realized_avg_abs_position"] = float(np.mean(np.abs(positions)))
        return diagnostics


def make_synthetic_pair(n: int = 900, seed: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    regimes = np.zeros(n, dtype=int)
    regimes[280:520] = 1
    regimes[520:620] = 2
    regimes[620:] = 0

    base = np.zeros(n)
    partner = np.zeros(n)
    spread = np.zeros(n)
    beta_by_regime = {0: 0.85, 1: 1.18, 2: 0.35}
    vol_by_regime = {0: 0.010, 1: 0.018, 2: 0.026}
    mean_reversion = {0: 0.09, 1: 0.15, 2: 0.01}

    base[0] = 100.0
    partner[0] = 95.0
    for t in range(1, n):
        r = regimes[t]
        market_ret = 0.00035 + rng.normal(0, vol_by_regime[r])
        base[t] = base[t - 1] * np.exp(market_ret)
        spread[t] = (
            (1.0 - mean_reversion[r]) * spread[t - 1]
            + rng.normal(0, 0.011 + 0.009 * (r == 2))
            + (0.055 if 540 < t < 575 else 0.0)
        )
        log_partner = (np.log(base[t]) - spread[t]) / beta_by_regime[r]
        partner[t] = np.exp(log_partner)

    return base, partner, regimes
