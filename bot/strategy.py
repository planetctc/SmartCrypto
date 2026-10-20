"""Trend-following breakout strategy.

Long when the 1h trend is up (close above a rising EMA) and a closed 5m bar breaks the
high of the previous N bars on above-average volume. Short is the mirror image.
The same prepare()/signal_at() code runs in backtest and live so results line up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

TF_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
         "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000}


@dataclass
class Signal:
    side: str          # "long" or "short"
    entry: float       # reference price (close of the signal bar)
    stop: float
    take_profit: float
    atr: float
    strength: float    # 0..1, maps to margin between min and max
    reason: str


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def prepare(df: pd.DataFrame, df_trend: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add indicator columns to the entry-timeframe frame. Input frames hold CLOSED bars only,
    with columns ts (ms, bar open time), open, high, low, close, volume."""
    s = cfg.strategy
    out = df.copy()
    out["atr"] = atr(out, s.atr_period)
    out["hh"] = out["high"].rolling(s.breakout_lookback).max().shift(1)
    out["ll"] = out["low"].rolling(s.breakout_lookback).min().shift(1)
    out["vol_avg"] = out["volume"].rolling(s.volume_lookback).mean().shift(1)
    out["close_time"] = out["ts"] + TF_MS[cfg.timeframe]

    t = df_trend[["ts", "close"]].copy()
    t["ema"] = ema(t["close"], s.trend_ema)
    rising = t["ema"] > t["ema"].shift(3)
    falling = t["ema"] < t["ema"].shift(3)
    t["trend"] = np.where((t["close"] > t["ema"]) & rising, 1,
                          np.where((t["close"] < t["ema"]) & falling, -1, 0))
    t.loc[t.index[: s.trend_ema], "trend"] = 0  # EMA not warmed up yet
    t["close_time"] = t["ts"] + TF_MS[cfg.trend_timeframe]
    t = t.rename(columns={"close": "trend_close", "ema": "trend_ema"})[
        ["close_time", "trend", "trend_close", "trend_ema"]]

    # only use trend bars that had CLOSED by the time the entry bar closed (no look-ahead)
    out = pd.merge_asof(out.sort_values("close_time"), t.sort_values("close_time"),
                        on="close_time", direction="backward")
    out["trend"] = out["trend"].fillna(0).astype(int)
    return out


def signal_at(row: pd.Series, cfg: Config) -> Signal | None:
    s, r = cfg.strategy, cfg.risk
    if any(pd.isna(row[c]) for c in ("atr", "hh", "ll", "vol_avg", "trend_ema")):
        return None
    if row["trend"] == 0 or row["vol_avg"] <= 0:
        return None
    vol_ratio = row["volume"] / row["vol_avg"]
    if vol_ratio < s.volume_factor:
        return None

    close, a = row["close"], row["atr"]
    if row["trend"] == 1 and close > row["hh"]:
        side, breakout = "long", (close - row["hh"]) / a
    elif row["trend"] == -1 and close < row["ll"]:
        side, breakout = "short", (row["ll"] - close) / a
    else:
        return None

    stop_dist = max(a * s.atr_stop_mult, close * r.min_stop_pct)
    sign = 1 if side == "long" else -1
    stop = close - sign * stop_dist
    tp = close + sign * stop_dist * s.reward_risk

    trend_dist = abs(row["trend_close"] - row["trend_ema"]) / row["trend_ema"]
    strength = float(np.mean([
        np.clip((vol_ratio - s.volume_factor) / (3.0 - s.volume_factor), 0, 1),
        np.clip(trend_dist / 0.01, 0, 1),
        np.clip(breakout, 0, 1),
    ]))
    return Signal(side, close, stop, tp, a, strength,
                  f"{side} breakout vol x{vol_ratio:.1f} trend {trend_dist:.2%}")


@dataclass
class ExitDecision:
    close: bool
    reason: str = ""
    new_stop: float | None = None


def manage_position(side: str, entry: float, stop: float, initial_stop: float, price: float,
                    best_price: float, atr_now: float, bars_held: int, cfg: Config) -> ExitDecision:
    """Bot-side trade management on top of the exchange's hard stop/take-profit.

    Moves the stop to break-even (plus fees) after +breakeven_at_r R, then trails it by
    trail_atr_mult * ATR. Returns close=True when the managed stop is crossed or the trade
    has been open too long.
    """
    s = cfg.strategy
    sign = 1 if side == "long" else -1
    risk = abs(entry - initial_stop)
    fee_pad = entry * 2 * (cfg.fees.taker + cfg.fees.slippage)
    new_stop = stop

    if sign * (best_price - entry) >= s.breakeven_at_r * risk:
        be = entry + sign * fee_pad
        trail = best_price - sign * s.trail_atr_mult * atr_now
        candidate = max(be, trail) if side == "long" else min(be, trail)
        if sign * (candidate - new_stop) > 0:
            new_stop = candidate

    if sign * (price - new_stop) <= 0:
        return ExitDecision(True, "managed stop hit", new_stop)
    if bars_held >= s.max_hold_bars:
        return ExitDecision(True, "max hold time", new_stop)
    return ExitDecision(False, "", new_stop if new_stop != stop else None)
