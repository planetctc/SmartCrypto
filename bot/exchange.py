"""Brokers: BitgetBroker (demo or live via ccxt) and PaperBroker (local simulation on live prices).

Both expose the same small interface so the live engine does not care which one it drives.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import ccxt
import pandas as pd

from .config import Config
from .risk import liquidation_price
from .strategy import TF_MS

log = logging.getLogger("bot.exchange")


@dataclass
class Position:
    symbol: str
    side: str          # "long" / "short"
    qty: float
    entry: float
    liq_price: float
    unrealized: float
    margin: float


def candles_to_df(rows: list[list], tf: str, now_ms: int) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df[df["ts"] + TF_MS[tf] <= now_ms]  # drop the bar that is still forming
    return df.drop_duplicates("ts").reset_index(drop=True)


def retry(fn, *args, tries: int = 4, **kwargs):
    """Retry network hiccups with backoff. Exchange rejections are raised immediately."""
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            if i == tries - 1:
                raise
            wait = 2 ** (i + 1)
            log.warning("network error %s, retry in %ss", e, wait)
            time.sleep(wait)


class BitgetBroker:
    """Talks to Bitget USDT-M perpetuals. demo=True uses Bitget demo trading (fake money)."""

    def __init__(self, cfg: Config, demo: bool):
        key = os.environ.get("BITGET_API_KEY")
        secret = os.environ.get("BITGET_API_SECRET")
        password = os.environ.get("BITGET_API_PASSPHRASE")
        if not (key and secret and password):
            raise SystemExit("缺少环境变量 BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE")
        self.cfg = cfg
        self.ex = ccxt.bitget({
            "apiKey": key, "secret": secret, "password": password,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if demo:
            self.ex.set_sandbox_mode(True)  # sends PAPTRADING: 1 -> Bitget demo account
        self.ex.load_markets()

    # ----- setup -----
    def setup_symbol(self, symbol: str) -> None:
        try:
            self.ex.set_position_mode(False, symbol)  # one-way mode
        except ccxt.ExchangeError as e:
            log.info("position mode unchanged for %s: %s", symbol, e)
        try:
            self.ex.set_margin_mode("isolated", symbol)
        except ccxt.ExchangeError as e:
            log.info("margin mode unchanged for %s: %s", symbol, e)
        for hold in ("long", "short"):
            retry(self.ex.set_leverage, self.cfg.leverage, symbol, {"holdSide": hold})

    # ----- market data -----
    def ohlcv(self, symbol: str, tf: str, limit: int) -> pd.DataFrame:
        rows = retry(self.ex.fetch_ohlcv, symbol, tf, None, limit)
        return candles_to_df(rows, tf, self.now_ms())

    def price(self, symbol: str) -> float:
        return float(retry(self.ex.fetch_ticker, symbol)["last"])

    def now_ms(self) -> int:
        return self.ex.milliseconds()

    # ----- account -----
    def balance(self) -> tuple[float, float]:
        """(equity incl. unrealized PnL, free balance) in USDT."""
        b = retry(self.ex.fetch_balance, {"type": "swap"})
        usdt = b.get("USDT", {})
        free = float(usdt.get("free") or 0)
        total = float(usdt.get("total") or 0)
        upnl = sum(p.unrealized for p in self.positions().values())
        info_equity = None
        for item in b.get("info", []) if isinstance(b.get("info"), list) else []:
            if item.get("marginCoin") == "USDT" and item.get("accountEquity"):
                info_equity = float(item["accountEquity"])
        return (info_equity if info_equity is not None else total + upnl), free

    def positions(self) -> dict[str, Position]:
        out = {}
        for p in retry(self.ex.fetch_positions, self.cfg.symbols):
            qty = float(p.get("contracts") or 0)
            if qty <= 0:
                continue
            out[p["symbol"]] = Position(
                symbol=p["symbol"], side=p["side"], qty=qty,
                entry=float(p.get("entryPrice") or 0),
                liq_price=float(p.get("liquidationPrice") or 0),
                unrealized=float(p.get("unrealizedPnl") or 0),
                margin=float(p.get("initialMargin") or p.get("collateral") or 0),
            )
        return out

    # ----- orders -----
    def round_qty(self, symbol: str, qty: float) -> float:
        q = float(self.ex.amount_to_precision(symbol, qty))
        min_amt = (self.ex.market(symbol)["limits"]["amount"]["min"] or 0)
        return q if q >= min_amt else 0.0

    def open(self, symbol: str, side: str, qty: float, stop: float, tp: float) -> float:
        """Market entry with stop-loss and take-profit attached on the exchange, so they
        still fire if this program crashes or loses connection."""
        params = {
            "marginMode": "isolated",
            "hedged": False,
            "stopLoss": {"triggerPrice": float(self.ex.price_to_precision(symbol, stop))},
            "takeProfit": {"triggerPrice": float(self.ex.price_to_precision(symbol, tp))},
        }
        order = self.ex.create_order(symbol, "market", "buy" if side == "long" else "sell",
                                     qty, None, params)
        fill = order.get("average") or order.get("price")
        if not fill:
            pos = self.positions().get(symbol)
            fill = pos.entry if pos else self.price(symbol)
        return float(fill)

    def close(self, symbol: str, pos: Position) -> float:
        side = "sell" if pos.side == "long" else "buy"
        order = self.ex.create_order(symbol, "market", side, pos.qty, None,
                                     {"marginMode": "isolated", "hedged": False, "reduceOnly": True})
        self.cancel_triggers(symbol)
        return float(order.get("average") or order.get("price") or self.price(symbol))

    def cancel_triggers(self, symbol: str) -> None:
        for plan in ("profit_loss", "normal_plan"):
            try:
                self.ex.cancel_all_orders(symbol, {"trigger": True, "planType": plan})
            except ccxt.ExchangeError as e:
                log.debug("cancel %s triggers on %s: %s", plan, symbol, e)


class PaperBroker:
    """Simulated account on real public prices. No API key needed, nothing is sent to Bitget.
    Hard stop / take-profit / liquidation are checked against the last price each loop."""

    def __init__(self, cfg: Config, start_balance: float = 100.0):
        self.cfg = cfg
        self.ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self.ex.load_markets()
        self.cash = start_balance
        self.pos: dict[str, Position] = {}
        self.orders: dict[str, tuple[float, float]] = {}  # symbol -> (stop, tp)
        self.closed_by_exchange: list[tuple[str, float, str]] = []

    def setup_symbol(self, symbol: str) -> None:
        pass

    def ohlcv(self, symbol, tf, limit):
        return candles_to_df(retry(self.ex.fetch_ohlcv, symbol, tf, None, limit), tf, self.now_ms())

    def price(self, symbol):
        return float(retry(self.ex.fetch_ticker, symbol)["last"])

    def now_ms(self):
        return self.ex.milliseconds()

    def _fee(self, notional):
        return notional * self.cfg.fees.taker

    def _slip(self, price, side, opening):
        s = self.cfg.fees.slippage
        worse_up = (side == "long") == opening
        return price * (1 + s) if worse_up else price * (1 - s)

    def tick(self) -> None:
        """Fire simulated exchange-side stop / take-profit / liquidation."""
        for sym, p in list(self.pos.items()):
            px = self.price(sym)
            stop, tp = self.orders[sym]
            sign = 1 if p.side == "long" else -1
            if sign * (px - p.liq_price) <= 0:
                self.cash -= p.margin
                del self.pos[sym], self.orders[sym]
                self.closed_by_exchange.append((sym, -p.margin, "liquidated"))
            elif sign * (px - stop) <= 0:
                self.closed_by_exchange.append((sym, self._settle(sym, stop), "stop-loss"))
            elif sign * (px - tp) >= 0:
                self.closed_by_exchange.append((sym, self._settle(sym, tp), "take-profit"))

    def _settle(self, sym, px) -> float:
        p = self.pos.pop(sym)
        self.orders.pop(sym)
        fill = self._slip(px, p.side, opening=False)
        sign = 1 if p.side == "long" else -1
        pnl = sign * (fill - p.entry) * p.qty - self._fee(fill * p.qty)
        self.cash += pnl
        return pnl

    def balance(self):
        upnl = 0.0
        for sym, p in self.pos.items():
            sign = 1 if p.side == "long" else -1
            upnl += sign * (self.price(sym) - p.entry) * p.qty
        used = sum(p.margin for p in self.pos.values())
        return self.cash + upnl, self.cash - used

    def positions(self):
        return dict(self.pos)

    def round_qty(self, symbol, qty):
        q = float(self.ex.amount_to_precision(symbol, qty))
        return q if q >= (self.ex.market(symbol)["limits"]["amount"]["min"] or 0) else 0.0

    def open(self, symbol, side, qty, stop, tp):
        fill = self._slip(self.price(symbol), side, opening=True)
        notional = fill * qty
        self.cash -= self._fee(notional)
        liq = liquidation_price(fill, side, self.cfg.leverage, self.cfg.risk.maintenance_margin_rate)
        self.pos[symbol] = Position(symbol, side, qty, fill, liq, 0.0, notional / self.cfg.leverage)
        self.orders[symbol] = (stop, tp)
        return fill

    def close(self, symbol, pos):
        px = self.price(symbol)
        self._settle(symbol, px)
        return self._slip(px, pos.side, opening=False)

    def cancel_triggers(self, symbol):
        pass
