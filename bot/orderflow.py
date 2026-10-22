"""Record Bitget order book and trades (盘口) for short-term research.

Candles only keep open/high/low/close/volume. What a fast bot can see that a person can't is
the live book: how much is resting on each side and who is hitting whom right now. Bitget has
no history of that, so this records it from public data (no API key) to CSV, one row per coin
per poll, for later testing of second-to-minute signals.
"""
from __future__ import annotations

import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt

log = logging.getLogger("bot.orderflow")

COLUMNS = ["ts", "bid", "ask", "bid_qty5", "ask_qty5", "bid_qty15", "ask_qty15",
           "buy_qty", "sell_qty", "trades", "last", "maybe_missed"]
TRADES_LIMIT = 100


def live_flow(ex, symbol: str) -> dict:
    """Right-now order flow for one coin: resting size on the 5 best levels each side and the
    aggressor volume of the last TRADES_LIMIT trades (who is hitting whom)."""
    book = ex.fetch_order_book(symbol, 15)
    trades = ex.fetch_trades(symbol, None, TRADES_LIMIT)
    return {"bid": book["bids"][0][0] if book["bids"] else 0.0,
            "ask": book["asks"][0][0] if book["asks"] else 0.0,
            "bid_qty5": sum(a for _, a, *_ in book["bids"][:5]),
            "ask_qty5": sum(a for _, a, *_ in book["asks"][:5]),
            "buy_qty": sum(t["amount"] for t in trades if t["side"] == "buy"),
            "sell_qty": sum(t["amount"] for t in trades if t["side"] == "sell")}


def flow_agrees(side: str, f: dict, min_share: float = 0.5,
                max_wall: float = 0.3) -> tuple[bool, float, float]:
    """(ok, share of recent taker volume on our side, book imbalance toward our side).

    A breakout that is real is being pushed by takers on its side right now; one where the
    other side is doing the hitting, or a thick wall sits in the way, is the kind that fails.
    """
    sign = 1 if side == "long" else -1
    taker = f["buy_qty"] + f["sell_qty"]
    share = (f["buy_qty"] if sign > 0 else f["sell_qty"]) / taker if taker else 0.5
    book = f["bid_qty5"] + f["ask_qty5"]
    imb = sign * (f["bid_qty5"] - f["ask_qty5"]) / book if book else 0.0
    return share >= min_share and imb > -max_wall, share, imb


class Recorder:
    def __init__(self, ex, symbols: list[str], out_dir: str | Path):
        self.ex, self.symbols = ex, symbols
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.seen: dict[str, set] = {s: set() for s in symbols}   # trade ids already counted
        self.since: dict[str, int | None] = {s: None for s in symbols}
        self.rows: dict[str, int] = {s: 0 for s in symbols}

    def path(self, symbol: str, ts: int) -> Path:
        day = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y%m%d")
        return self.out / f"{symbol.split('/')[0]}_{day}.csv"

    def snapshot(self, symbol: str) -> dict | None:
        book = self.ex.fetch_order_book(symbol, 15)
        trades = self.ex.fetch_trades(symbol, None, TRADES_LIMIT)
        if not book["bids"] or not book["asks"]:
            return None
        ts = int(book.get("timestamp") or self.ex.milliseconds())
        first = self.since[symbol] is None
        # the first poll only learns which trades already happened
        new = [] if first else [t for t in trades if t["id"] not in self.seen[symbol]]
        self.seen[symbol] = {t["id"] for t in trades}
        self.since[symbol] = ts
        qty = lambda side, n: sum(a for _, a, *_ in book[side][:n])  # noqa: E731
        return {"ts": ts, "bid": book["bids"][0][0], "ask": book["asks"][0][0],
                "bid_qty5": qty("bids", 5), "ask_qty5": qty("asks", 5),
                "bid_qty15": qty("bids", 15), "ask_qty15": qty("asks", 15),
                "buy_qty": sum(t["amount"] for t in new if t["side"] == "buy"),
                "sell_qty": sum(t["amount"] for t in new if t["side"] == "sell"),
                "trades": len(new),
                "last": trades[-1]["price"] if trades else (book["bids"][0][0] + book["asks"][0][0]) / 2,
                # every trade in the reply was new: more may have happened than we saw
                "maybe_missed": int(not first and len(trades) >= TRADES_LIMIT and len(new) == len(trades))}

    def poll(self) -> None:
        for sym in self.symbols:
            row = self.snapshot(sym)
            if row is None:
                continue
            p = self.path(sym, row["ts"])
            new_file = not p.exists()
            with open(p, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, COLUMNS)
                if new_file:
                    w.writeheader()
                w.writerow(row)
            self.rows[sym] += 1


def record(symbols: list[str], data_dir: str, interval: float = 1.0) -> None:
    """Poll until stopped. Status for the control panel goes to data/record_status.txt."""
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    symbols = [s for s in symbols if s in ex.markets]  # skip coins Bitget has no futures for
    rec = Recorder(ex, symbols, Path(data_dir) / "orderflow")
    status = Path(data_dir) / "record_status.txt"
    started = time.time()
    while True:
        t0 = time.time()
        try:
            rec.poll()
        except ccxt.NetworkError as e:  # includes 429 too-many-requests
            log.warning("order book poll: %s; waiting 5s", type(e).__name__)
            time.sleep(5)
            continue
        hours = (time.time() - started) / 3600
        status.write_text(f"记录盘口中 {hours:.1f} 小时，{max(rec.rows.values())} 行/币",
                          encoding="utf-8")
        time.sleep(max(0.0, interval - (time.time() - t0)))
