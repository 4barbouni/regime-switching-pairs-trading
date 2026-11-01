from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List


PAPER_BASE_URL = "https://paper-api.alpaca.markets"


@dataclass
class AlpacaAccount:
    equity: float
    buying_power: float


@dataclass
class AlpacaPosition:
    symbol: str
    qty: float
    market_value: float


@dataclass
class PlannedOrder:
    symbol: str
    side: str
    qty: float
    target_value: float
    current_value: float
    delta_value: float
    reference_price: float


class AlpacaPaperClient:
    def __init__(
        self,
        key_id: str | None = None,
        secret_key: str | None = None,
        base_url: str = PAPER_BASE_URL,
    ):
        self.key_id = key_id or os.environ.get("ALPACA_API_KEY_ID")
        self.secret_key = secret_key or os.environ.get("ALPACA_API_SECRET_KEY")
        self.base_url = base_url.rstrip("/")
        if not self.key_id or not self.secret_key:
            raise RuntimeError(
                "Missing Alpaca paper credentials. Set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY in your environment."
            )

    def account(self) -> AlpacaAccount:
        payload = self._request("GET", "/v2/account")
        return AlpacaAccount(
            equity=float(payload["equity"]),
            buying_power=float(payload["buying_power"]),
        )

    def positions(self) -> Dict[str, AlpacaPosition]:
        payload = self._request("GET", "/v2/positions")
        positions: Dict[str, AlpacaPosition] = {}
        for item in payload:
            symbol = item["symbol"].upper()
            positions[symbol] = AlpacaPosition(
                symbol=symbol,
                qty=float(item["qty"]),
                market_value=float(item["market_value"]),
            )
        return positions

    def submit_market_order(self, order: PlannedOrder) -> Dict[str, object]:
        body = {
            "symbol": order.symbol,
            "qty": f"{order.qty:.6f}",
            "side": order.side,
            "type": "market",
            "time_in_force": "day",
        }
        return self._request("POST", "/v2/orders", body)

    def _request(self, method: str, path: str, body: Dict[str, object] | None = None):
        url = self.base_url + path
        data = None
        headers = {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Alpaca paper API: {exc}") from exc
        return json.loads(text) if text else {}


def plan_pair_orders(
    symbols: Iterable[str],
    target_weights: Dict[str, float],
    account_equity: float,
    current_positions: Dict[str, AlpacaPosition],
    reference_prices: Dict[str, float],
    capital_fraction: float,
    min_order_value: float,
) -> List[PlannedOrder]:
    orders: List[PlannedOrder] = []
    for symbol in symbols:
        symbol = symbol.upper()
        target_value = account_equity * capital_fraction * target_weights.get(symbol, 0.0)
        current_value = current_positions.get(symbol, AlpacaPosition(symbol, 0.0, 0.0)).market_value
        delta = target_value - current_value
        if abs(delta) < min_order_value:
            continue
        price = reference_prices[symbol]
        if price <= 0.0:
            raise ValueError(f"reference price must be positive for {symbol}")
        orders.append(
            PlannedOrder(
                symbol=symbol,
                side="buy" if delta > 0.0 else "sell",
                qty=abs(delta) / price,
                target_value=target_value,
                current_value=current_value,
                delta_value=delta,
                reference_price=price,
            )
        )
    return orders

