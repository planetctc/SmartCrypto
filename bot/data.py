"""Download historical candles to CSV for backtesting."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import ccxt
import pandas as pd

from .strategy import TF_MS

log = logging.getLogger("bot.data")


def csv_path(data_dir: str, exchange: str, symbol: str, tf: str) -> Path:
    name = symbol.replace("/", "").replace(":", "_")
    return Path(data_dir) / f"{exchange}_{name}_{tf}.csv"


def download(symbol: str, tf: str, days: int, exchange: str = "bitget",
             data_dir: str = "data") -> Path:
    ex = getattr(ccxt, exchange)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    step = TF_MS[tf]
    now = ex.milliseconds()
    since = now - days * 86_400_000
    out = csv_path(data_dir, exchange, symbol, tf)
    old = pd.read_csv(out) if out.exists() else None
    if old is not None and len(old) and old["ts"].iloc[0] <= since + 86_400_000:
        since = int(old["ts"].iloc[-1]) + step  # already have the start: only fetch what is new
    else:
        old = None
    rows: list[list] = []
    while since < now - step:
        batch = ex.fetch_ohlcv(symbol, tf, since, 200)
        if not batch:
            # some exchanges only serve recent bars on the default endpoint; skip ahead
            since += 200 * step
            continue
        rows.extend(batch)
        nxt = batch[-1][0] + step
        if nxt <= since:
            break
        since = nxt
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if old is not None:
        df = pd.concat([old, df])
    df = df.drop_duplicates("ts").sort_values("ts")
    df = df[(df["ts"] + step <= now) & (df["ts"] >= now - days * 86_400_000)]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("saved %d bars to %s", len(df), out)
    if len(df) and df["ts"].iloc[0] > now - days * 86_400_000 + 2 * 86_400_000:
        log.warning("%s only returned data from %s; try --exchange binanceusdm for longer history",
                    exchange, pd.to_datetime(df["ts"].iloc[0], unit="ms"))
    return out


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
