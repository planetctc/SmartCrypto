"""机器人自己挑: every few hours the bot re-tests, on its own order-book recording, every entry
setting together with every way of getting out, the way it really trades (limit entry that only
fills if price comes back through it; limit exits; stops at market), and switches to the best.

Picked on the worse of the two parts (first 2/3, last 1/3), so a setting that only worked in
one part does not win. "verified" = both parts beat the round-trip cost; real money only
trades a verified setting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .flowtest import (MAKER_RT, MAKER_WAIT_S, MIN_HOURS, THRESHOLDS,
                       WINDOWS, features, hours_of, manual_exit)

MAX_HOLD_S = 4 * 3600
TAKER_EXTRA = 0.0004            # a stop / forced exit pays taker instead of maker
TPS = (0.001, 0.002, 0.003, 0.005)
SLS = (0.003, 0.005, 0.008)
MARGINS = (0.0, 0.0005, 0.001)  # 人工平法: what a waiting trade asks for above break-even
TRENDS = (0, 60)                # 0 = any direction; 60 = only with the last hour's move
BURSTS = (2, 4)                 # live 10/23: signals on ordinary volume (0.2-0.9x) lost
MIN_EACH = 10                   # trades needed in each part


@dataclass
class Choice:
    window: int
    threshold: float
    burst: float
    exit: str            # time / manual / bracket
    hold: int = 0        # seconds (time, manual)
    tp: float = 0.0      # bracket
    sl: float = 0.008
    margin: float = 0.0  # manual: a waiting trade asks for break-even + this
    rule: tuple = ()
    skip: tuple = ()     # coins that lost with this setting in BOTH parts: not traded
    trend: int = 0       # only trade with the last `trend` minutes' direction (0 = any)
    sel: float = 0.0     # average net per trade (after cost), first 2/3
    val: float = 0.0     # last 1/3
    trades: int = 0
    per_day: float = 0.0
    verified: bool = False

    def describe(self) -> str:
        out = {"time": f"拿 {self.hold // 60} 分钟挂单平",
               "manual": (f"{self.hold // 60} 分钟后有赚就走、没赚挂"
                          + (f"回本再多 {self.margin:.2%}" if self.margin else "回本价")
                          + f"等、4 小时后找近高点平、止损 {self.sl:.1%}"),
               "bracket": f"止盈 {self.tp:.1%} 挂单、止损 {self.sl:.1%}、最多拿 4 小时"}[self.exit]
        vol = f"成交量 {self.burst:g} 倍以上" if self.burst else "成交量不限"
        if self.trend:
            vol += f"、只顺最近 {self.trend} 分钟的方向"
        return (f"看 {self.window} 秒成交、主动方向 ≥{self.threshold:.0%}、{vol}，"
                f"{out}；每单净赚 挑选段 {self.sel:+.3%}、验证段 {self.val:+.3%}，"
                f"约 {self.per_day:.0f} 单/天")


def _bracket(ts, mid, i, px, k, tp, sl):
    end = np.searchsorted(ts, ts[i] + MAX_HOLD_S * 1000, side="right")
    r = k * (mid[i:end] / px - 1)
    win = np.flatnonzero(r >= tp)
    lose = np.flatnonzero(r <= -sl)
    w = win[0] if len(win) else len(r)
    lo = lose[0] if len(lose) else len(r)
    if w < lo:
        return tp, i + w
    if lo < len(r):
        return -sl - TAKER_EXTRA, i + lo
    return r[-1] - TAKER_EXTRA, i + len(r) - 1


def trend_sign(ts: np.ndarray, mid: np.ndarray, minutes: int) -> np.ndarray:
    """+1 / -1: price up / down over the last `minutes`; 0 without that much history."""
    j = np.searchsorted(ts, ts - minutes * 60_000)
    ok = (ts - ts[j] >= minutes * 60_000 * 0.9)
    return np.where(ok, np.sign(mid / mid[j] - 1), 0).astype(int)


def _entries(d: pd.DataFrame, threshold: float, burst: float,
             trend: int = 0) -> tuple[np.ndarray, ...]:
    """Signal rows with the row/price their limit entry would fill at (-1: not filled).
    trend > 0: only bursts in the direction price went over the last `trend` minutes (live
    10/23: ADA shorts into a rising ADA and XRP longs into a falling XRP made most losses)."""
    flow, tilt = d["flow"].to_numpy(), d["tilt"].to_numpy()
    bid, ask, ts = d["bid"].to_numpy(), d["ask"].to_numpy(), d["ts"].to_numpy()
    strong = d["burst"].to_numpy() >= burst
    side = np.where(strong & (flow >= threshold) & (tilt >= 0), 1,
                    np.where(strong & (flow <= -threshold) & (tilt <= 0), -1, 0))
    if trend:
        side = np.where(side == trend_sign(ts, d["mid"].to_numpy(), trend), side, 0)
    sig = np.flatnonzero(side)
    fill = np.full(len(sig), -1)
    px = np.where(side[sig] > 0, bid[sig], ask[sig])
    ends = np.searchsorted(ts, ts[sig] + MAKER_WAIT_S * 1000, side="right")
    for n, (i, e) in enumerate(zip(sig, ends)):
        seg = bid[i + 1:e] < px[n] if side[i] > 0 else ask[i + 1:e] > px[n]
        hit = np.flatnonzero(seg)
        if len(hit):
            fill[n] = i + 1 + hit[0]
    return sig, side[sig], px, fill, ends


def _run(d, entries, rule) -> list[tuple[int, float]]:
    """(signal ts, net move) one trade at a time for one coin."""
    sig, side, px, fill, ends = entries
    ts, mid = d["ts"].to_numpy(), d["mid"].to_numpy()
    out, free = [], -1
    for n, i in enumerate(sig):
        if ts[i] < free:
            continue
        f = fill[n]
        if f < 0:
            free = ts[min(ends[n], len(ts) - 1)]
            continue
        kind = rule[0]
        if kind == "time":
            j = np.searchsorted(ts, ts[f] + rule[1] * 1000)
            if j >= len(ts):
                break
            move = side[n] * (mid[j] / px[n] - 1)
        elif kind == "manual":
            move, j, how = manual_exit(ts, mid, f, px[n], side[n], rule[1], MAKER_RT,
                                       rule[2], MAKER_RT + rule[3])
            if how == "open":  # still waiting when the data ends: count it at the last price
                out.append((ts[i], move - MAKER_RT))
                break
        else:
            move, j = _bracket(ts, mid, f, px[n], side[n], rule[1], rule[2])
        out.append((ts[i], move - MAKER_RT))
        free = ts[j] + 1
    return out


def choose(frames: dict[str, pd.DataFrame]) -> tuple[Choice | None, str]:
    """Best entry + exit on the recording, and a short report."""
    hours = hours_of(frames)
    if not frames or hours < MIN_HOURS:
        return None, f"盘口数据还不够 {MIN_HOURS:.0f} 小时（现在 {hours:.1f} 小时），先用默认设置。"
    t0 = min(f["ts"].iloc[0] for f in frames.values())
    t1 = max(f["ts"].iloc[-1] for f in frames.values())
    cut = t0 + (t1 - t0) * 2 / 3
    # no plain "time is up, close" exits: they were the biggest loss live (nick: 自己死)
    rules = [("manual", h, sl, m) for h in (600, 1200) for sl in (0.005, 0.008)
                     for m in MARGINS] + \
        [("bracket", tp, sl) for tp in TPS for sl in SLS]
    found: list[Choice] = []
    for w in WINDOWS:
        feats = {c: features(f, w) for c, f in frames.items()}
        for th in THRESHOLDS:
            for k, tr in [(k, tr) for k in BURSTS for tr in TRENDS]:
                ent = {c: _entries(d, th, k, tr) for c, d in feats.items()}
                for rule in rules:
                    rows = [x for c, d in feats.items() for x in _run(d, ent[c], rule)]
                    a = [m for t, m in rows if t < cut]
                    b = [m for t, m in rows if t >= cut]
                    if len(a) < MIN_EACH or len(b) < MIN_EACH:
                        continue
                    ch = Choice(w, th, k, rule[0], sel=float(np.mean(a)), val=float(np.mean(b)),
                                trades=len(rows), per_day=len(rows) / hours * 24, rule=rule,
                                trend=tr)
                    if rule[0] == "bracket":
                        ch.tp, ch.sl = rule[1], rule[2]
                    else:
                        ch.hold = rule[1]
                    if rule[0] == "manual":
                        ch.sl, ch.margin = rule[2], rule[3]
                    ch.verified = ch.sel > 0 and ch.val > 0  # judged again below
                    found.append(ch)
    if not found:
        return None, "数据里成交的单太少，先用默认设置。"
    # picked on the FIRST part only; the last part was never used to choose, so it says
    # honestly whether the pick works (ranking by both parts let lucky combos win: 10/23 the
    # pick showed +0.086% / +0.111% while the plain analysis found nothing above cost)
    found.sort(key=lambda c: -c.sel)
    best = found[0]
    # coins that lost with this setting in both parts (live 2026-10-23: both stops were ADA)
    skip, per_coin = [], []
    for c, f in frames.items():
        d = features(f, best.window)
        rows = _run(d, _entries(d, best.threshold, best.burst, best.trend), best.rule)
        a = [m for t, m in rows if t < cut]
        b = [m for t, m in rows if t >= cut]
        if len(a) >= 3 and len(b) >= 3 and np.mean(a) < 0 and np.mean(b) < 0:
            skip.append(c)
        per_coin.append(f"{c} {len(rows)} 单 {np.mean([m for _, m in rows]) if rows else 0:+.3%}")
    best.skip = tuple(skip)
    lines = [f"机器人自己挑（{hours:.1f} 小时盘口数据，{len(found)} 种进场×平仓组合，"
             "照真实挂单成交算，已扣手续费）：", ""]
    lines.append("（只按前 2/3 挑，后 1/3 是没看过的数据，用来检验）")
    for c in found[:5]:
        lines.append(("✓ " if c.verified else "✗ ") + c.describe())
    lines += ["", "第一个做法每个币（每单净赚）：" + "，".join(per_coin)]
    if skip:
        lines.append("这些币两段都亏，先不做：" + "、".join(skip))
    lines += ["", ("用第一个：两段都赚。" if best.verified
                   else "用第一个，但还没有两段都赚的组合：模拟盘照跑收集数据，真钱不会开单。")]
    return best, "\n".join(lines)


def as_dict(c: Choice) -> dict:
    return asdict(c)
