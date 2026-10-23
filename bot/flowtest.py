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
HOLDS = (60, 180, 600, 1200, 1800)  # seconds to hold
MAX_GAP_MS = 5000               # rows further apart than this are the old, too-coarse recording
MAKER_RT = 0.0004               # maker in + maker out
TAKER_RT = 0.0012 + 0.0004      # taker in + out plus slippage
TAKER_IN_RT = 0.0006 + 0.0002   # market in (price = the ask/bid, so the spread is counted) + limit out
STOP = 0.008                    # bot's protective stop
MAX_HOLD_S = 4 * 3600           # 人工平法: after 4 hours look for the nearest good exit
LATE_WINDOW_S = 1800            #   = the best price of the last 30 minutes
LATE_GIVEUP_S = 3600            #   not back there within another hour: out at the mid
MAKER_WAIT_S = 20               # the bot cancels an unfilled limit entry after this long
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


def manual_exit(ts: np.ndarray, mid: np.ndarray, i: int, px: float, k: int, hold: int,
                cost: float, stop_pct: float = STOP, wait_for: float | None = None
                ) -> tuple[float, int, str]:
    """人工平法 from fill row i at price px: (move, exit row, how).
    At `hold`: in profit after cost -> out. Otherwise wait for break-even (move = cost).
    After 4 hours: out when price comes back to the best of the 30 minutes before, or at
    the mid an hour later. A 0.8% stop (market, so taker fee) the whole time.
    Data ending first: marked at the last mid ("open").
    wait_for: what the resting exit of a trade not in profit asks for (default: cost =
    break-even after fees; more = a little profit on top)."""
    target_be = cost if wait_for is None else wait_for
    t0 = ts[i]
    end = np.searchsorted(ts, t0 + (MAX_HOLD_S + LATE_GIVEUP_S) * 1000, side="right")
    r = k * (mid[i:end] / px - 1)
    t = ts[i:end]
    extra = 0.0004  # taker instead of maker on a stop / forced exit
    stop = np.flatnonzero(r <= -stop_pct)
    first_stop = stop[0] if len(stop) else len(r)
    h = np.searchsorted(t, t0 + hold * 1000)
    if h >= len(r):
        return (r[-1], i + len(r) - 1, "open") if first_stop >= len(r) else \
            (-stop_pct - extra, i + first_stop, "stop")
    if first_stop <= h:
        return -stop_pct - extra, i + first_stop, "stop"
    if r[h] >= cost:
        return r[h], i + h, "profit"
    late = np.searchsorted(t, t0 + MAX_HOLD_S * 1000)
    be = np.flatnonzero(r[h:late] >= target_be)
    be_at = h + be[0] if len(be) else len(r)
    if be_at < first_stop and be_at < late:
        return target_be, i + be_at, "breakeven"
    if first_stop < late:
        return -stop_pct - extra, i + first_stop, "stop"
    if late >= len(r):
        return r[-1], i + len(r) - 1, "open"
    lo = np.searchsorted(t, t[late] - LATE_WINDOW_S * 1000)
    target = r[lo:late + 1].max()
    back = np.flatnonzero(r[late:] >= target)
    back_at = late + back[0] if len(back) else len(r)
    if back_at < first_stop and back_at < len(r):
        return target, i + back_at, "late"
    if first_stop < len(r):
        return -stop_pct - extra, i + first_stop, "stop"
    if end >= len(ts):
        return r[-1], i + len(r) - 1, "open"
    return r[-1] - extra, i + len(r) - 1, "late"


def real_trades(d: pd.DataFrame, threshold: float, hold: int, intensity: float,
                entry: str, manual: bool = False) -> tuple[pd.DataFrame, int]:
    """What the bot can really get: (trades, signals tried).

    entry "maker": buy at the bid when the signal fires (sell at the ask for a short); it only
    fills if price comes back THROUGH that level within MAKER_WAIT_S (a later bid below it),
    and then the hold starts from the fill. Unfilled: the next signal may try again.
    entry "taker": buy at the ask right away. Exit at the mid `hold` seconds later."""
    flow, tilt, mid = d["flow"].to_numpy(), d["tilt"].to_numpy(), d["mid"].to_numpy()
    bid, ask = d["bid"].to_numpy(), d["ask"].to_numpy()
    ts, burst = d["ts"].to_numpy(), d["burst"].to_numpy()
    strong = burst >= intensity
    side = np.where(strong & (flow >= threshold) & (tilt >= 0), 1,
                    np.where(strong & (flow <= -threshold) & (tilt <= 0), -1, 0))
    rows, tried, free_at = [], 0, -1
    for i in np.flatnonzero(side):
        if ts[i] < free_at:
            continue
        tried += 1
        k = side[i]
        if entry == "taker":
            px, t_in = (ask[i] if k > 0 else bid[i]), ts[i]
        else:
            px = bid[i] if k > 0 else ask[i]
            end = np.searchsorted(ts, ts[i] + MAKER_WAIT_S * 1000, side="right")
            seg = slice(i + 1, end)
            through = np.flatnonzero(bid[seg] < px) if k > 0 else np.flatnonzero(ask[seg] > px)
            if not len(through):
                free_at = ts[min(end, len(ts) - 1)]
                continue
            t_in = ts[i + 1 + through[0]]
        if manual:
            fi = int(np.searchsorted(ts, t_in))
            move, j, how = manual_exit(ts, mid, fi, px, k, hold, MAKER_RT)
            rows.append((ts[i], move, how, (ts[j] - t_in) / 60_000))
            free_at = ts[j] + 1
            continue
        j = np.searchsorted(ts, t_in + hold * 1000)
        if j >= len(d) or ts[j] - t_in > hold * 1500:
            free_at = t_in
            continue
        rows.append((ts[i], k * (mid[j] / px - 1), "time", hold / 60))
        free_at = ts[j]
    return pd.DataFrame(rows, columns=["ts", "move", "how", "minutes"]), tried


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
    lines += ["", "照机器人真实下单再算一次（前 3 组）：", ""]
    real_ok = []
    for w, th, k, hold, *_ in results[:3]:
        feats = {c: features(f, w) for c, f in frames.items()}
        burst = f"、成交量 {k} 倍" if k else ""
        lines.append(f"看 {w} 秒、≥{th:.0%}{burst}、拿 {hold} 秒：")
        for entry, cost, name in (("maker", MAKER_RT, "挂单进（价格回来才成交）"),
                                  ("taker", TAKER_IN_RT, "市价进、挂单出")):
            parts = [real_trades(f, th, hold, k, entry) for f in feats.values()]
            tr = pd.concat([p for p, _ in parts], ignore_index=True)
            tried = sum(n for _, n in parts)
            if tr.empty:
                lines.append(f"   {name}：{tried} 次信号，一单都没成交")
                continue
            a, b = tr[tr.ts < cut], tr[tr.ts >= cut]
            ok = len(a) >= 10 and len(b) >= 10 and a.move.mean() > cost and b.move.mean() > cost
            lines.append(f"   {name}：{tried} 次信号成交 {len(tr)} 单，挑选段 "
                         f"{a.move.mean() if len(a) else float('nan'):+.3%}，验证段 "
                         f"{b.move.mean() if len(b) else float('nan'):+.3%}，成本 {cost:.2%}"
                         + ("  ✓ 真实下单也赚" if ok else ""))
            if ok:
                real_ok.append(f"看 {w} 秒、≥{th:.0%}{burst}、拿 {hold} 秒、{name}")
        parts = [real_trades(f, th, hold, k, "maker", manual=True) for f in feats.values()]
        tr = pd.concat([p for p, _ in parts], ignore_index=True)
        if len(tr):
            a, b = tr[tr.ts < cut], tr[tr.ts >= cut]
            ok = len(a) >= 10 and len(b) >= 10 and a.move.mean() > MAKER_RT \
                and b.move.mean() > MAKER_RT
            n = tr["how"].value_counts()
            lines.append(f"   挂单进、人工平法（有赚就走、没赚等回本、4 小时后找最近高点平、0.8% 止损）："
                         f"{len(tr)} 单，挑选段 {a.move.mean() if len(a) else float('nan'):+.3%}，"
                         f"验证段 {b.move.mean() if len(b) else float('nan'):+.3%}，成本 {MAKER_RT:.2%}"
                         + ("  ✓ 真实下单也赚" if ok else ""))
            lines.append(f"      有赚就走 {n.get('profit', 0)}、等到回本 {n.get('breakeven', 0)}、"
                         f"4 小时后平 {n.get('late', 0)}、止损 {n.get('stop', 0)}、"
                         f"数据结束还没平 {n.get('open', 0)}；平均拿 {tr.minutes.mean():.0f} 分钟，"
                         f"不亏的单占 {(tr.move >= MAKER_RT - 1e-12).mean():.0%}")
            if ok:
                real_ok.append(f"看 {w} 秒、≥{th:.0%}{burst}、拿 {hold} 秒、人工平法")
    lines.append("")
    if good and not real_ok:
        lines.append("结论：价格方向看对了，但照真实下单（挂单要等价格回来、市价要多付手续费）算，"
                     "还盖不过成本。")
        return "\n".join(lines)
    if real_ok:
        lines.append("结论：真实下单也赚的是：" + "；".join(real_ok) + "。把这个结果发给 Claude。")
        return "\n".join(lines)
    if good:
        w, th, k, hold = good[0]
        lines.append(f"结论：有用。最好的是看 {w} 秒成交、主动方向 ≥{th:.0%}"
                     f"{f'、成交量是平时 {k} 倍以上' if k else ''}、拿 {hold} 秒，"
                     f"两段都比挂单成本 {MAKER_RT:.2%} 走得多。把这个结果发给 Claude，改成实盘信号。")
    else:
        lines.append(f"结论：这段数据里，盘口信号之后的价格变动还盖不过成本（挂单进出 {MAKER_RT:.2%}）。"
                     "多录几天再看，或者换更活跃的时段。")
    return "\n".join(lines)
