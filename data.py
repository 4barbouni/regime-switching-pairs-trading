from __future__ import annotations

import csv
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class PriceSeries:
    symbol: str
    dates: List[str]
    close: np.ndarray


def fetch_stooq_daily(symbol: str, start: str, end: str, cache_dir: str) -> PriceSeries:
    """
    Fetch daily US equity prices from Stooq and cache the CSV.

    Dates use YYYYMMDD format. Symbol examples: KO, PEP, XOM, CVX.
    """

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    stooq_symbol = f"{symbol.lower()}.us"
    file_path = cache_path / f"{symbol.upper()}_{start}_{end}_stooq.csv"

    if not file_path.exists():
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d&d1={start}&d2={end}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not download {symbol} from Stooq. Original error: {exc}"
            ) from exc
        file_path.write_text(payload, encoding="utf-8")

    return read_stooq_csv(symbol, str(file_path))


def fetch_yahoo_daily(symbol: str, start: str, end: str, cache_dir: str) -> PriceSeries:
    """
    Fetch daily prices from Yahoo's chart endpoint and cache the JSON.

    Dates use YYYYMMDD format. The end date is treated as inclusive.
    """

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    file_path = cache_path / f"{symbol.upper()}_{start}_{end}_yahoo.json"

    if not file_path.exists():
        period1 = _date_to_epoch(start)
        period2 = _date_to_epoch(end) + 24 * 60 * 60
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
            f"?period1={period1}&period2={period2}&interval=1d&events=history"
        )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not download {symbol} from Yahoo Finance. Original error: {exc}"
            ) from exc
        file_path.write_text(payload, encoding="utf-8")

    return read_yahoo_json(symbol, str(file_path))


def fetch_daily_prices(symbol: str, start: str, end: str, cache_dir: str) -> PriceSeries:
    """
    Fetch daily prices with provider fallback.

    Stooq is tried first because it returns simple CSVs. Yahoo is used as a fallback
    for symbols/date ranges that Stooq rejects.
    """

    errors: List[str] = []
    for fetcher in (fetch_stooq_daily, fetch_yahoo_daily):
        try:
            return fetcher(symbol, start, end, cache_dir)
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))
    raise RuntimeError(f"Could not download {symbol} from any provider. " + " | ".join(errors))


def read_stooq_csv(symbol: str, path: str) -> PriceSeries:
    text = Path(path).read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "Close" not in rows[0]:
        raise ValueError(f"{path} does not look like a Stooq daily CSV")

    dates: List[str] = []
    close: List[float] = []
    for row in rows:
        value = row.get("Close", "")
        if value:
            dates.append(row["Date"])
            close.append(float(value))
    return PriceSeries(symbol=symbol.upper(), dates=dates, close=np.asarray(close, dtype=float))


def read_yahoo_json(symbol: str, path: str) -> PriceSeries:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo returned an error for {symbol}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError(f"{path} does not contain Yahoo chart results")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    if not timestamps or not closes:
        raise ValueError(f"{path} does not contain daily close prices")

    dates: List[str] = []
    close: List[float] = []
    for ts, value in zip(timestamps, closes):
        if value is None:
            continue
        dates.append(datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d"))
        close.append(float(value))
    return PriceSeries(symbol=symbol.upper(), dates=dates, close=np.asarray(close, dtype=float))


def _date_to_epoch(value: str) -> int:
    dt = datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def align_price_series(series: Iterable[PriceSeries]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    items = list(series)
    if len(items) < 2:
        raise ValueError("at least two series are required")

    common_dates = set(items[0].dates)
    for item in items[1:]:
        common_dates &= set(item.dates)
    dates = sorted(common_dates)
    if len(dates) < 100:
        raise ValueError("not enough overlapping dates")

    aligned: Dict[str, np.ndarray] = {}
    for item in items:
        by_date = dict(zip(item.dates, item.close))
        aligned[item.symbol] = np.asarray([by_date[d] for d in dates], dtype=float)
    return dates, aligned
