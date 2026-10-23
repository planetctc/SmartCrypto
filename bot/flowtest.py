"""分析盘口: test order-flow signals on the order book recorded by 记录盘口.

Every candle setup tried (1/5/15 minutes, with the trend, against it, ranges, bigger
targets) lost about the same before fees as its exact opposite: candles carry no edge at
these speeds. What a bot sees that a person can't is the book itself, second by second.
This checks, on the recorded data, whether one-sided taker flow plus a matching book tilt
is followed by a price move bigger than the cost of a round trip.

Signal at second t (no look-ahead: only rows up to t):
  flow  = (buy - sell) / (buy + sell) taker volume over the last `window` seconds
  tilt  = (bid_qty5 - ask_qty5) / (bid_qty5 + ask_qty5) at t
  burst = taker volume in the window / its average over the last hour
  long  when flow >= threshold, tilt >= 0 and burst >= intensity; short mirrored
Result: mid price `hold` seconds later vs mid at t, one trade at a time per coin.
Windows and holds are in seconds by timestamp, so rows recorded slower than 1 s still count right.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = (10, 30, 60)          # seconds of taker flow
THRESHOLDS = (0.5, 0.7)         # how one-sided the flow must be
INTENSITY = (0, 2, 4)           # window volume vs its 1-hour average (0 = any; 4 = a burst)
HOLDS = (60, 180, 600)          # seconds to hold
MAX_GAP_MS = 5000               # rows further apart than this are the old, too-coarse recording
MAKER_RT = 0.0004               # maker in + maker out
TAKER_RT = 0.0012 + 0.0004      # taker in + out plus slippage
MIN_HOURS = 6.0
MIN_TRADES = 30


def load(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Recorded rows per coin, one frame per coin, sorted by time. Only the fine rows are
    kept (at most MAX_GAP_MS after the previous one): the first recorder version polled each
    coin only every ~20 s, and mixing that in blurs every second-level signal."""
    out: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(Path(data_dir, "orderflow").glob("*_*.csv")):
        coin = p.stem.rsplit("_", 1)[0]
        try:
            out.setdefault(coin, []).append(pd.read_csv(p))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    frames = {}
    for c, fs in out.items():
        f = pd.concat(fs, ignore_index=True).sort_values("ts").drop_duplicates("ts")
        f = f[f["ts"].diff() <= MAX_GAP_MS].reset_index(drop=True)
        if len(f) > 1:
            frames[c] = f
    return frames


def hours_of(frames: dict[str, pd.DataFrame]) -> float:
    """Recorded time of the coin with the most fine data (gaps not counted)."""
    return max((f["ts"].diff().clip(upper=MAX_GAP_MS).sum() / 3_600_000
                for f in frames.values()), default=0.0)


def features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Flow over the last `window` SECONDS (by timestamp, whatever the row spacing)."""
    d = df.copy()
    d.index = pd.to_datetime(d["ts"], unit="ms")
    d["mid"] = (d["bid"] + d["ask"]) / 2
    buy = d["buy_qty"].rolling(f"{window}s").sum()
    sell = d["sell_qty"].rolling(f"{window}s").sum()
    vol = buy + sell
    d["flow"] = ((buy - sell) / vol).where(vol > 0, 0.0)
    typical = vol.rolling("3600s", min_periods=60).mean()
    d["burst"] = (vol / typical).where(typical > 0, 0.0).fillna(0.0)
    book = d["bid_qty5"] + d["ask_qty5"]
    d["tilt"] = ((d["bid_qty5"] - d["ask_qty5"]) / book).where(book > 0, 0.0)
    d["spread"] = (d["ask"] - d["bid"]) / d["mid"]
    return d.reset_index(drop=True)


def trades(d: pd.DataFrame, threshold: float, hold: int, intensity: float = 0) -> pd.DataFrame:
    """One trade at a time: enter on a signal, exit at the first row `hold` seconds later
    (skipped when the recording has a gap there)."""
    flow, tilt, mid = d["flow"].to_numpy(), d["tilt"].to_numpy(), d["mid"].to_numpy()
    ts, spread, burst = d["ts"].to_numpy(), d["spread"].to_numpy(), d["burst"].to_numpy()
    strong = burst >= intensity
    side = np.where(strong & (flow >= threshold) & (tilt >= 0), 1,
                    np.where(strong & (flow <= -threshold) & (tilt <= 0), -1, 0))
    exit_at = np.searchsorted(ts, ts + hold * 1000)
    rows, busy_until = [], -1
    for i in np.flatnonzero(side):
        j = exit_at[i]
        if i < busy_until or j >= len(d) or ts[j] - ts[i] > hold * 1500:
            continue
        rows.append((ts[i], side[i] * (mid[j] / mid[i] - 1), spread[i]))
        busy_until = j
    return pd.DataFrame(rows, columns=["ts", "move", "spread"])


def spacing(frames: dict[str, pd.DataFrame]) -> float:
    """Median seconds between rows of one coin."""
    gaps = [np.median(np.diff(f["ts"].to_numpy())) / 1000 for f in frames.values() if len(f) > 1]
    return float(np.median(gaps)) if gaps else 0.0


def analyse(frames: dict[str, pd.DataFrame]) -> str:
    if not frames:
        return "还没有盘口数据。先点「记录盘口」录几个小时（最好一两天），再来分析。"
    t0 = min(f["ts"].iloc[0] for f in frames.values())
    t1 = max(f["ts"].iloc[-1] for f in frames.values())
    hours = hours_of(frames)
    gap = spacing(frames)
    head = [f"盘口数据：{'、'.join(frames)}，1 秒级的数据共 {hours:.1f} 小时（旧版 20 秒一行的数据不算），"
            f"{sum(len(f) for f in frames.values())} 行，每个币约 {gap:.0f} 秒一行"]
    if hours < MIN_HOURS:
        return "\n".join(head + [f"1 秒级的数据还不够 {MIN_HOURS:.0f} 小时，继续记录盘口再来分析。"])
    cut = t0 + (t1 - t0) * 2 / 3
    results = []
    for w in WINDOWS:
        feats = {c: features(f, w) for c, f in frames.items()}
        for th in THRESHOLDS:
            for k in INTENSITY:
                for hold in HOLDS:
                    tr = pd.concat([trades(f, th, hold, k) for f in feats.values()],
                                   ignore_index=True)
                    if tr.empty:
                        continue
                    a, b = tr[tr.ts < cut], tr[tr.ts >= cut]
                    results.append((w, th, k, hold, len(tr),
                                    a.move.mean() if len(a) else np.nan,
                                    b.move.mean() if len(b) else np.nan, len(a), len(b),
                                    tr.spread.mean(), len(tr) / hours * 24))
    if not results:
        return "\n".join(head + ["这些数据里一个信号都没有。"])
    # rank by the worse of the two parts; too few trades in either part ranks last
    results.sort(key=lambda r: -(min(r[5], r[6]) if min(r[7], r[8]) >= 10 else -1.0))
    lines = head + [
        f"前 2/3 时间用来挑，后 1/3 验证。每单成本：挂单进出约 {MAKER_RT:.2%}，"
        f"市价进出约 {TAKER_RT:.2%}。", "平均每单价格走多少（已按方向算，正数 = 走对了）：", ""]
    good = []
    for w, th, k, hold, n, ma, mb, na, nb, spread, per_day in results[:12]:
        ok = n >= MIN_TRADES and ma > MAKER_RT and mb > MAKER_RT
        tag = "  ✓ 两段都盖过挂单成本" if ok else ""
        burst = f"、成交量是平时 {k} 倍以上" if k else ""
        lines.append(f"看 {w} 秒成交、主动方向 ≥{th:.0%}{burst}、拿 {hold} 秒："
                     f"{n} 单（约 {per_day:.0f} 单/天）{tag}")
        lines.append(f"   挑选段 {ma:+.3%}（{na} 单），验证段 {mb:+.3%}（{nb} 单），"
                     f"平均买卖价差 {spread:.3%}")
        if ok:
            good.append((w, th, k, hold))
    lines.append("")
    if good:
        w, th, k, hold = good[0]
        lines.append(f"结论：有用。最好的是看 {w} 秒成交、主动方向 ≥{th:.0%}"
                     f"{f'、成交量是平时 {k} 倍以上' if k else ''}、拿 {hold} 秒，"
                     f"两段都比挂单成本 {MAKER_RT:.2%} 走得多。把这个结果发给 Claude，改成实盘信号。")
    else:
        lines.append(f"结论：这段数据里，盘口信号之后的价格变动还盖不过成本（挂单进出 {MAKER_RT:.2%}）。"
                     "多录几天再看，或者换更活跃的时段。")
    return "\n".join(lines)
