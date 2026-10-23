"""分析盘口: test order-flow signals on the order book recorded by 记录盘口.

Every candle setup tried (1/5/15 minutes, with the trend, against it, ranges, bigger
targets) lost about the same before fees as its exact opposite: candles carry no edge at
these speeds. What a bot sees that a person can't is the book itself, second by second.
This checks, on the recorded data, whether one-sided taker flow plus a matching book tilt
is followed by a price move bigger than the cost of a round trip.

Signal at second t (no look-ahead: only rows up to t):
  flow  = (buy - sell) / (buy + sell) taker volume over the last `window` seconds
  tilt  = (bid_qty5 - ask_qty5) / (bid_qty5 + ask_qty5) at t
  long  when flow >= threshold and tilt >= 0, short mirrored
Result: mid price `hold` seconds later vs mid at t, one trade at a time per coin.
Windows and holds are in seconds by timestamp, so rows recorded slower than 1 s still count right.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = (10, 30, 60)          # seconds of taker flow
THRESHOLDS = (0.5, 0.7, 0.9)    # how one-sided the flow must be
HOLDS = (30, 60, 180, 600)      # seconds to hold
MAKER_RT = 0.0004               # maker in + maker out
TAKER_RT = 0.0012 + 0.0004      # taker in + out plus slippage
MIN_HOURS = 6.0
MIN_TRADES = 30


def load(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Recorded rows per coin, one frame per coin, sorted by time."""
    out: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(Path(data_dir, "orderflow").glob("*_*.csv")):
        coin = p.stem.rsplit("_", 1)[0]
        try:
            out.setdefault(coin, []).append(pd.read_csv(p))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    return {c: pd.concat(fs, ignore_index=True).sort_values("ts").drop_duplicates("ts")
            .reset_index(drop=True) for c, fs in out.items()}


def features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Flow over the last `window` SECONDS (by timestamp, whatever the row spacing)."""
    d = df.copy()
    d.index = pd.to_datetime(d["ts"], unit="ms")
    d["mid"] = (d["bid"] + d["ask"]) / 2
    buy = d["buy_qty"].rolling(f"{window}s").sum()
    sell = d["sell_qty"].rolling(f"{window}s").sum()
    d["flow"] = ((buy - sell) / (buy + sell)).where(buy + sell > 0, 0.0)
    book = d["bid_qty5"] + d["ask_qty5"]
    d["tilt"] = ((d["bid_qty5"] - d["ask_qty5"]) / book).where(book > 0, 0.0)
    d["spread"] = (d["ask"] - d["bid"]) / d["mid"]
    return d.reset_index(drop=True)


def trades(d: pd.DataFrame, threshold: float, hold: int) -> pd.DataFrame:
    """One trade at a time: enter on a signal, exit at the first row `hold` seconds later
    (skipped when the recording has a gap there)."""
    flow, tilt, mid = d["flow"].to_numpy(), d["tilt"].to_numpy(), d["mid"].to_numpy()
    ts, spread = d["ts"].to_numpy(), d["spread"].to_numpy()
    exit_at = np.searchsorted(ts, ts + hold * 1000)
    rows, i, n = [], 0, len(d)
    while i < n:
        j = exit_at[i]
        if j >= n:
            break
        side = 1 if flow[i] >= threshold and tilt[i] >= 0 else \
            -1 if flow[i] <= -threshold and tilt[i] <= 0 else 0
        if side and ts[j] - ts[i] <= hold * 1500:
            rows.append((ts[i], side * (mid[j] / mid[i] - 1), spread[i]))
            i = j
        else:
            i += 1
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
    hours = (t1 - t0) / 3_600_000
    gap = spacing(frames)
    head = [f"盘口数据：{'、'.join(frames)}，共 {hours:.1f} 小时，"
            f"{sum(len(f) for f in frames.values())} 行，每个币约 {gap:.0f} 秒一行"]
    if gap > 5:
        head.append(f"注意：每个币 {gap:.0f} 秒才一行，太粗，看不清几十秒内的变化。"
                    "请用新版「记录盘口」重新录（现在每个币约 1 秒一行）。")
    if hours < MIN_HOURS:
        return "\n".join(head + [f"数据太少（不到 {MIN_HOURS:.0f} 小时），继续记录盘口再来分析。"])
    cut = t0 + (t1 - t0) * 2 / 3
    results = []
    for w in WINDOWS:
        feats = {c: features(f, w) for c, f in frames.items()}
        for th in THRESHOLDS:
            for hold in HOLDS:
                tr = pd.concat([trades(f, th, hold) for f in feats.values()], ignore_index=True)
                if tr.empty:
                    continue
                a, b = tr[tr.ts < cut], tr[tr.ts >= cut]
                results.append((w, th, hold, len(tr), a.move.mean() if len(a) else np.nan,
                                b.move.mean() if len(b) else np.nan, len(a), len(b),
                                tr.spread.mean()))
    if not results:
        return "\n".join(head + ["这些数据里一个信号都没有。"])
    results.sort(key=lambda r: -np.nan_to_num(min(r[4], r[5]), nan=-1))
    lines = head + [
        f"前 2/3 时间用来挑，后 1/3 验证。每单成本：挂单进出约 {MAKER_RT:.2%}，"
        f"市价进出约 {TAKER_RT:.2%}。", "平均每单价格走多少（已按方向算，正数 = 走对了）：", ""]
    good = []
    for w, th, hold, n, ma, mb, na, nb, spread in results[:12]:
        ok = n >= MIN_TRADES and ma > MAKER_RT and mb > MAKER_RT
        tag = "  ✓ 两段都盖过挂单成本" if ok else ""
        lines.append(f"看 {w} 秒成交、主动方向 ≥{th:.0%}、拿 {hold} 秒：{n} 单{tag}")
        lines.append(f"   挑选段 {ma:+.3%}（{na} 单），验证段 {mb:+.3%}（{nb} 单），"
                     f"平均买卖价差 {spread:.3%}")
        if ok:
            good.append((w, th, hold, ma, mb))
    lines.append("")
    if good:
        w, th, hold, ma, mb = good[0]
        lines.append(f"结论：有用。最好的是看 {w} 秒成交、主动方向 ≥{th:.0%}、拿 {hold} 秒，"
                     f"两段都比挂单成本 {MAKER_RT:.2%} 走得多。把这个结果发给 Claude，改成实盘信号。")
    else:
        lines.append(f"结论：这段数据里，盘口信号之后的价格变动还盖不过成本（挂单进出 {MAKER_RT:.2%}）。"
                     "多录几天再看，或者换更活跃的时段。")
    return "\n".join(lines)
