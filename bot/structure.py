"""大结构: read the chart the way the analyst notes do, on 1h / 4h / daily candles.

The notes that called BTC right (2026-10-22) looked at: the nearest resistance and support
(swing highs/lows on 1h and 4h), a false breakout above an old high and the volume reclaim
after it, round numbers, the 1h and 4h channel direction, 4h MACD above or below zero, and
the daily close against the Bollinger middle band and the box. This computes the same things
from closed candles only, plus one score from -5 (everything bearish) to +5 (everything bullish).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .strategy import ema

PIVOT = 3            # a swing high is higher than the 3 bars on each side
RECLAIM_BARS = 24    # look for a false breakout / reclaim in the last 24 hours
VOL_MULT = 1.5       # a reclaim counts when its bar has 1.5x the average volume


def _pivots(df: pd.DataFrame, n: int = PIVOT) -> tuple[np.ndarray, np.ndarray]:
    h, lo = df["high"].to_numpy(), df["low"].to_numpy()
    w = 2 * n + 1
    if len(df) < w:
        return np.array([]), np.array([])
    hi_max = pd.Series(h).rolling(w, center=True).max().to_numpy()
    lo_min = pd.Series(lo).rolling(w, center=True).min().to_numpy()
    return h[h == hi_max], lo[lo == lo_min]


def round_step(price: float) -> float:
    """BTC 84,000 -> 1,000; ETH 4,000 -> 100; SOL 200 -> 10."""
    return 10 ** (math.floor(math.log10(price)) - 1) if price > 0 else 1.0


def _slope(c: pd.Series, n: int) -> float:
    """Channel direction: least-squares slope of the last n closes, in % of price per bar."""
    y = c.to_numpy()[-n:]
    if len(y) < 3:
        return 0.0
    return float(np.polyfit(np.arange(len(y)), y, 1)[0] / y[-1])


def _reclaim(h1: pd.DataFrame) -> int:
    """+1: price broke above the old high, fell back under it, and is now back above on volume
    (the 10/21 BTC move); -1: broke above and is now back under (false breakout).
    Mirrored for the old low."""
    if len(h1) < RECLAIM_BARS + 48:
        return 0
    old, recent = h1.iloc[:-RECLAIM_BARS].tail(48), h1.tail(RECLAIM_BARS)
    top, bottom = old["high"].max(), old["low"].min()
    c, v = recent["close"].to_numpy(), recent["volume"].to_numpy()
    avg = h1["volume"].tail(RECLAIM_BARS + 48).mean()
    last = len(c) - 1
    if (recent["high"] > top).any():
        if c[-1] < top:
            return -1
        under = np.flatnonzero(c < top)
        if len(under) and (v[under[-1] + 1:] > VOL_MULT * avg).any() and under[-1] < last:
            return 1
    if (recent["low"] < bottom).any():
        if c[-1] > bottom:
            return 1
        over = np.flatnonzero(c > bottom)
        if len(over) and (v[over[-1] + 1:] > VOL_MULT * avg).any() and over[-1] < last:
            return -1
    return 0


def snapshot(h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame,
             price: float | None = None) -> dict:
    """Everything on closed candles; `price` (default the last 1h close) sets the distances."""
    px = float(price if price is not None else h1["close"].iloc[-1])
    highs = np.concatenate([_pivots(h1.tail(120))[0], _pivots(h4.tail(60))[0]])
    lows = np.concatenate([_pivots(h1.tail(120))[1], _pivots(h4.tail(60))[1]])
    above, below = highs[highs > px], lows[lows < px]
    res = float(above.min()) if len(above) else float(h4["high"].tail(60).max())
    sup = float(below.max()) if len(below) else float(h4["low"].tail(60).min())
    step = round_step(px)

    c4 = h4["close"]
    macd4 = (ema(c4, 12) - ema(c4, 26)).iloc[-1] if len(c4) >= 26 else 0.0
    cd = d1["close"]
    mid = cd.rolling(20).mean().iloc[-1] if len(cd) >= 20 else cd.mean()
    box_hi, box_lo = d1["high"].tail(20).max(), d1["low"].tail(20).min()
    s1, s4 = _slope(h1["close"], 24), _slope(c4, 20)
    reclaim = _reclaim(h1)
    above_mid = 1 if cd.iloc[-1] > mid else -1
    score = int(np.sign(s1) + np.sign(s4) + np.sign(macd4) + above_mid + reclaim)
    return {
        "res": res, "sup": sup,
        "dist_res": res / px - 1, "dist_sup": px / sup - 1 if sup > 0 else 1.0,
        "round_dist": abs(px - round(px / step) * step) / px,
        "slope1h": s1, "slope4h": s4,
        "macd4h": float(macd4 / px),
        "boll_d": above_mid,
        "box_d": float((cd.iloc[-1] - box_lo) / (box_hi - box_lo)) if box_hi > box_lo else 0.5,
        "reclaim": reclaim, "score": score,
    }


def history(h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame, since_ms: int) -> pd.DataFrame:
    """snapshot() at every 1h close from since_ms on, each row usable from `avail` (ms)."""
    rows = []
    for i in np.flatnonzero(h1["ts"].to_numpy() >= since_ms - 3_600_000):
        t = int(h1["ts"].iloc[i]) + 3_600_000           # the moment this 1h bar closed
        a, b, c = h1.iloc[:i + 1], h4[h4["ts"] + 4 * 3_600_000 <= t], d1[d1["ts"] + 86_400_000 <= t]
        if len(a) < 30 or len(b) < 30 or len(c) < 5:
            continue
        rows.append({"avail": t, **snapshot(a, b, c)})
    return pd.DataFrame(rows)
