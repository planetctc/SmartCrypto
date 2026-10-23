"""全面分析: every kind of data together, and let the computer weigh them.

Each one alone (candle indicators, order book) was not enough. This puts all of them side by
side every 10 seconds and fits one model on the first 2/3 of the recording:
  order book : taker flow over 10/30/60 s, volume burst, book tilt (5 and 15 levels), spread
  candles    : 1/5/15-minute returns, RSI, MACD, Bollinger position, ATR, distance from EMA50,
               volume vs average (all on CLOSED 1-minute bars built from the recording)
  positioning: funding rate, open interest change over 15 minutes, long/short account ratio
  大结构     : 1h/4h/daily like the analyst notes (bot/structure.py): distance to resistance and
               support, round numbers, 1h and 4h channel direction, 4h MACD vs zero, daily close
               vs Bollinger middle and box, false breakout / volume reclaim, total score
The model (ridge regression) predicts the move over the next 1 / 3 / 10 minutes. It is then
traded on the last 1/3 it never saw: only when the prediction is in its strongest 10% / 5% /
1%, one trade at a time per coin, and the result is compared with the round-trip cost.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .flowtest import MAKER_RT, MIN_HOURS, features, hours_of, load
from .strategy import atr, ema, rsi

STEP_S = 10
HOLDS = (60, 180, 600)
QUANTILES = (0.90, 0.95, 0.99)
RIDGE = 10.0
NAMES = {
    "flow10": "10 秒主动买卖", "flow30": "30 秒主动买卖", "flow60": "60 秒主动买卖",
    "burst": "放量程度", "tilt5": "前 5 档挂单厚度", "tilt15": "前 15 档挂单厚度",
    "spread": "买卖价差", "ret1": "1 分钟涨跌", "ret5": "5 分钟涨跌", "ret15": "15 分钟涨跌",
    "rsi": "RSI", "macd": "MACD", "boll": "布林带位置", "atr": "波动大小(ATR)",
    "ema_dist": "离 EMA50 多远", "vol_ratio": "成交量比平时", "funding": "资金费率",
    "oi_chg": "持仓量变化", "long_short": "多空比",
    "dist_res": "离阻力多远", "dist_sup": "离支撑多远", "round_dist": "离整数关口多远",
    "slope1h": "1 小时通道方向", "slope4h": "4 小时通道方向", "macd4h": "4 小时 MACD 零轴上下",
    "boll_d": "日线在布林中轨上下", "box_d": "日线在箱体的位置", "reclaim": "假突破/放量收回",
    "score": "大结构总分",
}
STRUCT = ("res", "sup", "round_dist", "slope1h", "slope4h", "macd4h", "boll_d", "box_d",
          "reclaim", "score")


def fetch_structure(coins: list[str], since_ms: int, data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """1h/4h/daily candles from Bitget for each coin, turned into structure.history()."""
    from .data import download
    from .structure import history
    days = int((pd.Timestamp.now("UTC").value // 1_000_000 - since_ms) / 86_400_000) + 1
    out = {}
    for c in coins:
        sym = f"{c}/USDT:USDT"
        try:
            h1, h4, d1 = (pd.read_csv(download(sym, tf, days + extra, data_dir=str(data_dir)))
                          for tf, extra in (("1h", 12), ("4h", 40), ("1d", 60)))
            out[c] = history(h1, h4, d1, since_ms)
        except Exception as e:  # noqa: BLE001 - a coin without candles just lacks these columns
            print(f"{c} 大结构没读到：{e}")
    return out


def load_market(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    out: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(Path(data_dir, "market").glob("*_*.csv")):
        try:
            out.setdefault(p.stem.rsplit("_", 1)[0], []).append(pd.read_csv(p))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    return {c: pd.concat(fs, ignore_index=True).sort_values("ts").reset_index(drop=True)
            for c, fs in out.items()}


def coin_table(raw: pd.DataFrame, market: pd.DataFrame | None,
               struct: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row every STEP_S seconds with every feature (past data only) and the future moves."""
    f10, f30, f60 = (features(raw, w) for w in (10, 30, 60))
    d = f30.copy()
    d["flow10"], d["flow30"], d["flow60"] = f10["flow"], f30["flow"], f60["flow"]
    book15 = d["bid_qty15"] + d["ask_qty15"]
    d["tilt5"] = d["tilt"]
    d["tilt15"] = ((d["bid_qty15"] - d["ask_qty15"]) / book15).where(book15 > 0, 0.0)

    # closed 1-minute bars from the recorded mid price and taker volume
    t = pd.to_datetime(d["ts"], unit="ms")
    m = pd.Series(d["mid"].to_numpy(), index=t)
    v = pd.Series((d["buy_qty"] + d["sell_qty"]).to_numpy(), index=t)
    bars = m.resample("1min").ohlc().dropna()
    bars["volume"] = v.resample("1min").sum().reindex(bars.index).fillna(0)
    c = bars["close"]
    ind = pd.DataFrame(index=bars.index)
    ind["ret1"], ind["ret5"], ind["ret15"] = c.pct_change(1), c.pct_change(5), c.pct_change(15)
    ind["rsi"] = rsi(c, 14)
    macd = ema(c, 12) - ema(c, 26)
    ind["macd"] = (macd - ema(macd, 9)) / c
    sd = c.rolling(20).std()
    ind["boll"] = (c - c.rolling(20).mean()) / (2 * sd).replace(0, np.nan)
    ind["atr"] = atr(bars.reset_index(drop=True), 14).to_numpy() / c
    ind["ema_dist"] = c / ema(c, 50) - 1
    ind["vol_ratio"] = bars["volume"] / bars["volume"].rolling(20).mean().replace(0, np.nan)
    ind["avail"] = (ind.index + pd.Timedelta(minutes=1)).astype("int64") // 1_000_000

    s = d.iloc[::STEP_S].copy()
    s = pd.merge_asof(s.sort_values("ts"), ind.sort_values("avail"), left_on="ts",
                      right_on="avail", direction="backward")
    if market is not None and len(market):
        mk = market.copy()
        for col in ("funding", "open_interest", "long_short"):
            mk[col] = pd.to_numeric(mk[col], errors="coerce")
        mk["oi_chg"] = mk["open_interest"].pct_change(15)
        s = pd.merge_asof(s, mk[["ts", "funding", "oi_chg", "long_short"]], on="ts",
                          direction="backward")
    for col in ("funding", "oi_chg", "long_short"):
        if col not in s:
            s[col] = np.nan
    if struct is not None and len(struct):
        st = struct.sort_values("avail").rename(columns={"avail": "st_avail"})
        s = pd.merge_asof(s, st[["st_avail", *STRUCT]], left_on="ts", right_on="st_avail",
                          direction="backward")
        s["dist_res"] = s["res"] / s["mid"] - 1     # from the live price, not the hourly close
        s["dist_sup"] = s["mid"] / s["sup"] - 1
    ts = d["ts"].to_numpy()
    mid = d["mid"].to_numpy()
    for h in HOLDS:
        j = np.searchsorted(ts, s["ts"].to_numpy() + h * 1000)
        ok = j < len(d)
        fwd = np.full(len(s), np.nan)
        jj = j[ok]
        gap_ok = ts[jj] - s["ts"].to_numpy()[ok] <= h * 1500
        vals = mid[jj] / s["mid"].to_numpy()[ok] - 1
        fwd[np.flatnonzero(ok)[gap_ok]] = vals[gap_ok]
        s[f"fwd{h}"] = fwd
    return s


def _trade(sub: pd.DataFrame, pred: np.ndarray, thr: float, hold: int) -> pd.DataFrame:
    """Strongest predictions only, one trade at a time per coin."""
    rows = []
    for coin, g in sub.groupby("coin"):
        p = pred[g.index.to_numpy()]
        busy = -1
        for ts, pi, fwd in zip(g["ts"].to_numpy(), p, g[f"fwd{hold}"].to_numpy()):
            if ts < busy or abs(pi) < thr or np.isnan(fwd):
                continue
            rows.append((ts, np.sign(pi) * fwd))
            busy = ts + hold * 1000
    return pd.DataFrame(rows, columns=["ts", "move"])


def study(data_dir: str | Path, structure: dict[str, pd.DataFrame] | None = None) -> str:
    frames = load(data_dir)
    hours = hours_of(frames)
    if not frames or hours < MIN_HOURS:
        return f"全面分析：1 秒级的盘口数据还不够 {MIN_HOURS:.0f} 小时（现在 {hours:.1f} 小时），继续记录。"
    market = load_market(data_dir)
    structure = structure or {}
    table = pd.concat([coin_table(f, market.get(c), structure.get(c)).assign(coin=c)
                       for c, f in frames.items()], ignore_index=True)
    feats = [k for k in NAMES if k in table and table[k].notna().mean() > 0.5]
    t0, t1 = table["ts"].min(), table["ts"].max()
    cut = t0 + (t1 - t0) * 2 / 3
    days = hours / 24
    lines = [f"全面分析：{len(frames)} 个币，{hours:.1f} 小时，每 {STEP_S} 秒看一次，"
             f"一起看 {len(feats)} 种数据（盘口、K 线指标、资金费率/持仓量/多空比、"
             f"1h/4h/日线大结构）。",
             "电脑在前 2/3 时间里自己算每种数据该占多少分量，再在后 1/3（没看过的）上照着做。",
             f"每单成本：挂单进出约 {MAKER_RT:.2%}。", ""]
    best = None
    for hold in HOLDS:
        y = f"fwd{hold}"
        data = table.dropna(subset=[y]).reset_index(drop=True)
        train = data[data.ts < cut]
        if len(train) < 500:
            continue
        mu, sd = train[feats].mean(), train[feats].std().replace(0, 1)
        X = ((data[feats] - mu) / sd).fillna(0).clip(-5, 5).to_numpy()
        tr = (data.ts < cut).to_numpy()
        A = X[tr]
        w = np.linalg.solve(A.T @ A + RIDGE * np.eye(len(feats)), A.T @ data.loc[tr, y].to_numpy())
        pred = X @ w
        for q in QUANTILES:
            thr = float(np.quantile(np.abs(pred[tr]), q))
            a = _trade(data[tr], pred, thr, hold)
            b = _trade(data[~tr], pred, thr, hold)
            if len(a) < 10 or len(b) < 10:
                continue
            ok = a.move.mean() > MAKER_RT and b.move.mean() > MAKER_RT
            per_day = (len(a) + len(b)) / days
            lines.append(f"拿 {hold // 60} 分钟、只做最强的 {1 - q:.0%} 信号：约 {per_day:.0f} 单/天"
                         + ("  ✓ 两段都盖过成本" if ok else ""))
            lines.append(f"   挑选段 {a.move.mean():+.3%}（{len(a)} 单），"
                         f"验证段 {b.move.mean():+.3%}（{len(b)} 单）")
            if best is None or b.move.mean() > best[0]:
                best = (b.move.mean(), hold, q, w, ok)
    if best is None:
        return "\n".join(lines + ["数据里的单数太少，继续记录再分析。"])
    move, hold, q, w, ok = best
    order = np.argsort(-np.abs(w))[:6]
    lines += ["", f"电脑认为最有用的数据（拿 {hold // 60} 分钟那组，正 = 越大越看涨）："]
    lines += [f"   {NAMES[feats[i]]}：{w[i] / np.abs(w).max():+.2f}" for i in order]
    lines.append("")
    lines.append(f"结论：{'有用！把这个结果发给 Claude，改成实盘信号。' if ok else ''}"
                 f"验证段最好每单 {move:+.3%}（拿 {hold // 60} 分钟、最强 {1 - q:.0%}），"
                 f"成本 {MAKER_RT:.2%}。" + ("" if ok else "还没盖过成本，继续记录数据。"))
    return "\n".join(lines)
