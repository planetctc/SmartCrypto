"""盘口模式: trade the order-flow signal that beat the cost in 全面分析 (2026-10-23, 6.8 h, 9 coins):
30 s of taker flow at least 50% one way, volume at least 4x its usual, book not against it,
hold 10 minutes. Selection part +0.056%/trade, unseen part +0.050%, maker round trip 0.04%.

The signal is computed with exactly the analysis code (flowtest.features) on the rows the order
book recorder writes, so live trading sees the same numbers the test did. If no recorder is
running, the bot starts one itself.
"""
from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .flowtest import features

log = logging.getLogger("bot.flow")

KEEP_MS = 3_600_000 + 120_000   # the burst baseline is the last hour
FRESH_MS = 10_000               # rows older than this: recorder not running
TAIL_BYTES = 800_000            # about 1.5 hours of one coin's rows


class FlowFeed:
    """Reads new rows of data/orderflow/COIN_day.csv as the recorder appends them."""

    def __init__(self, data_dir: str | Path, symbols: list[str]):
        self.dir = Path(data_dir) / "orderflow"
        self.symbols = symbols
        self.pos: dict[Path, int] = {}
        self.cols: dict[Path, list[str]] = {}
        self.rows: dict[str, list[dict]] = {s: [] for s in symbols}
        self.started_recorder = False

    def _path(self, sym: str, ts_ms: int) -> Path:
        day = datetime.fromtimestamp(ts_ms / 1000, timezone.utc).strftime("%Y%m%d")
        return self.dir / f"{sym.split('/')[0]}_{day}.csv"

    def _read_new(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path, "rb") as f:
            if path not in self.pos:
                # first look at a big file: header plus only the last ~1.5 hours of rows
                header = f.readline()
                self.cols[path] = next(csv.reader([header.decode("utf-8", "replace").strip()]))
                size = path.stat().st_size
                start = max(len(header), size - TAIL_BYTES)
                f.seek(start)
                if start > len(header):
                    f.readline()  # skip the partial line
                self.pos[path] = f.tell()
            f.seek(self.pos[path])
            chunk = f.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return []
        self.pos[path] += end + 1
        lines = chunk[:end].decode("utf-8", "replace").splitlines()
        out = []
        for row in csv.reader(lines):
            if len(row) != len(self.cols[path]):
                continue
            try:
                out.append({k: float(v) if v not in ("", None) else 0.0
                            for k, v in zip(self.cols[path], row)})
            except ValueError:
                continue
        return out

    def update(self, now_ms: int) -> None:
        for sym in self.symbols:
            new = []
            for t in {now_ms - 86_400_000, now_ms}:  # yesterday's file too, around midnight
                new += self._read_new(self._path(sym, t))
            rows = self.rows[sym] + new
            self.rows[sym] = [r for r in rows if r["ts"] >= now_ms - KEEP_MS]

    def ensure_recorder(self, now_ms: int, symbols: list[str]) -> None:
        """Coins with no fresh rows (记录盘口 not running for them): record them in this
        process, once."""
        if self.started_recorder:
            return
        self.first_ms = getattr(self, "first_ms", now_ms)
        if now_ms - self.first_ms < 15_000:  # give a recorder started at the same time a moment
            return
        missing = [s for s in symbols
                   if not (self.rows.get(s) and self.rows[s][-1]["ts"] >= now_ms - FRESH_MS)]
        if not missing:
            return
        from .orderflow import record
        self.started_recorder = True
        log.info("FLOWFEED recording order book inside the bot for %s", " ".join(missing))
        threading.Thread(target=record, args=(missing, str(self.dir.parent)), daemon=True).start()

    def trend(self, sym: str, minutes: int, now_ms: int) -> int:
        """+1 / -1: mid up / down over the last `minutes`; 0 without that much history."""
        rows = self.rows.get(sym) or []
        if not rows or rows[0]["ts"] > now_ms - minutes * 60_000 * 0.9:
            return 0
        then = next(r for r in rows if r["ts"] >= now_ms - minutes * 60_000)
        a, b = (then["bid"] + then["ask"]) / 2, (rows[-1]["bid"] + rows[-1]["ask"]) / 2
        return 1 if b > a else -1 if b < a else 0

    def latest(self, sym: str, window: int) -> dict | None:
        """flowtest.features on the last hour, last row; None if stale or too little data."""
        rows = self.rows.get(sym) or []
        if len(rows) < 60:
            return None
        df = pd.DataFrame(rows).sort_values("ts").drop_duplicates("ts")
        last = features(df, window).iloc[-1]
        return last.to_dict()


def flow_side(f: dict, threshold: float, burst: float) -> str | None:
    """Same rule as flowtest.trades: one-sided taker flow, book not against it, volume burst."""
    if f["burst"] < burst:
        return None
    if f["flow"] >= threshold and f["tilt"] >= 0:
        return "long"
    if f["flow"] <= -threshold and f["tilt"] <= 0:
        return "short"
    return None
