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

from .config import VOTES, Config

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
    votes: tuple = ()  # score signal: the indicators that agreed with the side


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def prepare(df: pd.DataFrame, df_trend: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add indicator columns to the entry-timeframe frame. Input frames hold CLOSED bars only,
    with columns ts (ms, bar open time), open, high, low, close, volume."""
    s = cfg.strategy
    out = df.copy()
    out["atr"] = atr(out, s.atr_period)
    out["hh"] = out["high"].rolling(s.breakout_lookback).max().shift(1)
    out["ll"] = out["low"].rolling(s.breakout_lookback).min().shift(1)
    out["vol_avg"] = out["volume"].rolling(s.volume_lookback).mean().shift(1)
    out["ema_fast"] = ema(out["close"], s.fast_ema)
    out["bb_mid"] = out["close"].rolling(s.bb_period).mean()
    out["bb_sd"] = out["close"].rolling(s.bb_period).std()
    out["rsi"] = rsi(out["close"], s.rsi_period)
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
    if s.signal_type == "score":
        out = votes(out, cfg)
    return out


def votes(out: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One +1 (bullish) / -1 (bearish) / 0 column per indicator, plus their sum in "score".

    Every column only uses the bar itself and earlier bars, so it is the same live and in
    backtests. Indicator settings are the common defaults traders use.
    """
    s = cfg.strategy
    c, o, h, lo = out["close"], out["open"], out["high"], out["low"]
    sign = lambda cond_up, cond_down: np.where(cond_up, 1, np.where(cond_down, -1, 0))  # noqa: E731

    out["v_trend"] = out["trend"]
    slow = ema(c, 50)
    out["v_ema"] = sign((c > out["ema_fast"]) & (out["ema_fast"] > slow),
                        (c < out["ema_fast"]) & (out["ema_fast"] < slow))
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    out["v_macd"] = sign((hist > 0) & (hist > hist.shift(1)), (hist < 0) & (hist < hist.shift(1)))
    # KDJ (9,3,3): K above D and not yet overbought is bullish, mirror for bearish
    low9, high9 = lo.rolling(9).min(), h.rolling(9).max()
    rsv = 100 * (c - low9) / (high9 - low9).replace(0, np.nan)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    out["v_kdj"] = sign((k > d) & (k < 80), (k < d) & (k > 20))
    r = out["rsi"]
    out["v_rsi"] = sign((r > 50) & (r < 70), (r < 50) & (r > 30))
    upper = out["bb_mid"] + 2 * out["bb_sd"]
    lower = out["bb_mid"] - 2 * out["bb_sd"]
    out["v_boll"] = sign((c > out["bb_mid"]) & (c < upper), (c < out["bb_mid"]) & (c > lower))
    big = out["volume"] > 1.2 * out["vol_avg"]
    out["v_volume"] = sign(big & (c > o), big & (c < o))
    # candles: engulfing, or a long wick rejecting one side (hammer / shooting star)
    body = (c - o).abs()
    po, pc = o.shift(1), c.shift(1)
    rng = (h - lo).replace(0, np.nan)
    bull = ((c > o) & (pc < po) & (c >= po) & (o <= pc)) | \
           ((np.minimum(o, c) - lo >= 2 * body) & ((c - lo) / rng > 0.6))
    bear = ((c < o) & (pc > po) & (c <= po) & (o >= pc)) | \
           ((h - np.maximum(o, c) >= 2 * body) & ((h - c) / rng > 0.6))
    out["v_candle"] = sign(bull & ~bear, bear & ~bull)
    out["v_breakout"] = sign(c > out["hh"], c < out["ll"])
    cols = [f"v_{name}" for name in VOTES]
    out["score"] = out[cols].sum(axis=1)
    out["prev_score"] = out["score"].shift(1)
    return out


def _side(row: pd.Series, cfg: Config) -> tuple[str, float] | None:
    """Which way the closed bar says to trade, and how clean the setup is (0..1)."""
    s = cfg.strategy
    close, a, trend = row["close"], row["atr"], row["trend"]
    if s.signal_type == "score":
        # enter on the bar where enough indicators line up, not on every bar they stay lined up
        n, prev = row["score"], row["prev_score"]
        if pd.isna(prev):
            return None
        if n >= s.min_score > prev:
            return "long", n / len(VOTES)
        if n <= -s.min_score < prev:
            return "short", -n / len(VOTES)
        return None
    if s.signal_type in ("breakout", "combo"):
        if trend == 1 and close > row["hh"]:
            return "long", (close - row["hh"]) / a
        if trend == -1 and close < row["ll"]:
            return "short", (row["ll"] - close) / a
        if s.signal_type == "breakout":
            return None
    if s.signal_type == "combo":
        # not only new highs/lows: in a trend, a bounce into the fast average that gets
        # rejected (bar pokes through it and closes back on the trend side) is an entry too
        if trend == 1 and row["low"] <= row["ema_fast"] < close:
            return "long", 0.5
        if trend == -1 and row["high"] >= row["ema_fast"] > close:
            return "short", 0.5
        return None
    if s.signal_type == "pullback":
        # trend up, the bar dipped to the fast average and closed back above it, green
        if trend == 1 and row["low"] <= row["ema_fast"] < close and close > row["open"]:
            return "long", (close - row["ema_fast"]) / a
        if trend == -1 and row["high"] >= row["ema_fast"] > close and close < row["open"]:
            return "short", (row["ema_fast"] - close) / a
        return None
    # reversion: stretched far outside the band with an extreme RSI, not against a clear trend
    if pd.isna(row["bb_sd"]) or pd.isna(row["rsi"]):
        return None
    band = s.bb_std * row["bb_sd"]
    if trend != -1 and close < row["bb_mid"] - band and row["rsi"] < s.rsi_low:
        return "long", (row["bb_mid"] - band - close) / a
    if trend != 1 and close > row["bb_mid"] + band and row["rsi"] > 100 - s.rsi_low:
        return "short", (close - row["bb_mid"] - band) / a
    return None


def signal_at(row: pd.Series, cfg: Config) -> Signal | None:
    s, r = cfg.strategy, cfg.risk
    if any(pd.isna(row[c]) for c in ("atr", "hh", "ll", "vol_avg", "trend_ema")):
        return None
    if row["vol_avg"] <= 0 or (row["trend"] == 0 and s.signal_type in ("breakout", "pullback", "combo")):
        return None
    vol_ratio = row["volume"] / row["vol_avg"]
    if vol_ratio < s.volume_factor:
        return None
    trend_dist = abs(row["trend_close"] - row["trend_ema"]) / row["trend_ema"]
    if trend_dist < s.trend_min_gap:
        return None  # trend too weak: price is hugging its average, breakouts there fail more
    picked = _side(row, cfg)
    if picked is None:
        return None
    side, breakout = picked
    close, a = row["close"], row["atr"]

    stop_dist = max(a * s.atr_stop_mult, close * r.min_stop_pct)
    sign = 1 if side == "long" else -1
    stop = close - sign * stop_dist
    tp = close + sign * stop_dist * s.take_profit_r[-1]

    strength = float(np.mean([
        np.clip((vol_ratio - s.volume_factor) / (3.0 - s.volume_factor), 0, 1),
        np.clip(trend_dist / 0.01, 0, 1),
        np.clip(breakout, 0, 1),
    ]))
    reason = f"{side} {s.signal_type} vol x{vol_ratio:.1f} trend {trend_dist:.2%}"
    if s.signal_type == "score":
        want = 1 if side == "long" else -1
        agree = tuple(v for v in VOTES if row[f"v_{v}"] == want)
        reason = f"{side} 综合 {abs(int(row['score']))}/{len(VOTES)}：{'、'.join(VOTES[v] for v in agree)}"
        return Signal(side, close, stop, tp, a, strength, reason, agree)
    return Signal(side, close, stop, tp, a, strength, reason)


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


def scales_in(cfg: Config) -> bool:
    """True when a trade keeps part of its size back for a 补仓."""
    return cfg.strategy.scale_in_frac < 1 and cfg.strategy.scale_in_at_r > 0


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
        moved = []
        for tp in tps:  # keep every level its own price (ETH 1.2R and 2R can share a round level)
            nice = off_round(tp, side, "tp")
            moved.append(nice if not moved or sign * (nice - moved[-1]) > 0 else tp)
        tps = moved
    add = entry - sign * stop_dist * s.scale_in_at_r if scales_in(cfg) else entry
    return TradePlan(stop, add, tps)
