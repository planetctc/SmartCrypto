"""Trend-following breakout strategy.

Long when the 1h trend is up (close above a rising EMA) and a closed 5m bar breaks the
high of the previous N bars on above-average volume. Short is the mirror image.
The same prepare()/signal_at() code runs in backtest and live so results line up.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

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
    tp = close + sign * stop_dist * s.take_profit_r[-1]

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
    # a trade whose stop is already locked in profit is left to the trailing stop
    locked = sign * (new_stop - entry) > 0
    if bars_held >= s.max_hold_bars and not locked:
        return ExitDecision(True, "max hold time", new_stop)
    return ExitDecision(False, "", new_stop if new_stop != stop else None)



def round_step(price: float) -> float:
    """Size of a 'round number' for this price: 100 for BTC ~80000, 10 for ETH ~2400,
    0.1 for SOL ~150, 0.001 for XRP ~1.3, 0.0001 for DOGE ~0.15 (kept under 0.5% of price
    so it never moves a stop by much)."""
    step = 10.0 ** (math.floor(math.log10(price)) - 2)
    return step / 10 if step / price > 0.005 else step


def off_round(price: float, side: str, kind: str) -> float:
    """Keep stops and take-profits off round numbers, where everyone else's orders sit.

    Stop: just beyond the round level it is on or near (60700 -> 60675 for a long), so a
    sweep of the round number does not take us out. Take-profit: just before the nearest
    round level (61800 -> 61765 for a long), so it fills before price stalls there.
    """
    step = round_step(price)
    sign = 1 if side == "long" else -1
    if kind == "stop":
        level = (math.floor if side == "long" else math.ceil)(price / step) * step
        if abs(price - level) < 0.3 * step:
            return level - sign * 0.25 * step
        return price
    level = round(price / step) * step
    return level - sign * 0.35 * step


@dataclass
class TradePlan:
    stop: float
    add_price: float          # 补仓 here (equal to entry when scale-in is off)
    take_profits: list[float]  # partial levels first, the exchange take-profit last


def trade_plan(side: str, entry: float, stop_dist: float, cfg: Config) -> TradePlan:
    """Stop, 补仓 price and take-profit ladder for a new trade, all from the stop distance."""
    s = cfg.strategy
    sign = 1 if side == "long" else -1
    stop = entry - sign * stop_dist
    tps = [entry + sign * stop_dist * r for r in s.take_profit_r]
    if s.avoid_round_numbers:
        stop = off_round(stop, side, "stop")
        tps = [off_round(tp, side, "tp") for tp in tps]
    add = entry - sign * stop_dist * s.scale_in_at_r if s.scale_in_frac < 1 else entry
    return TradePlan(stop, add, tps)
