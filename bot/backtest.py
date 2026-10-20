"""Bar-by-bar backtest using the same strategy and risk code as live trading.

Conservative assumptions: entries fill at the next bar's open plus slippage, taker fee on both
sides, if a bar touches both stop and take-profit the stop is assumed to hit first, and a bar
that opens beyond the stop fills at that worse open price (gap). Funding fees are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .risk import RiskManager, liquidation_price
from .strategy import TF_MS, manage_position, prepare, signal_at


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    step = TF_MS[tf]
    g = df.assign(bucket=(df["ts"] // step) * step).groupby("bucket")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    }).reset_index().rename(columns={"bucket": "ts"})
    return out


@dataclass
class BtPos:
    side: str
    qty: float
    entry: float
    initial_stop: float
    stop: float
    tp: float
    best: float
    liq: float
    margin: float
    entry_fee: float
    opened_i: int
    opened_ts: int


def run_backtest(cfg: Config, frames: dict[str, pd.DataFrame], start_balance: float = 100.0):
    fee, slip = cfg.fees.taker, cfg.fees.slippage
    step = TF_MS[cfg.timeframe]
    risk = RiskManager(cfg)

    maker = cfg.entry_order == "maker"
    # ts -> row dict per symbol: much faster than DataFrame.loc inside the loop
    prepared: dict[str, dict[int, dict]] = {}
    for sym, df in frames.items():
        p = prepare(df, resample(df, cfg.trend_timeframe), cfg)
        prepared[sym] = {int(r["ts"]): r for r in p.to_dict("records")}
    timeline = sorted(set().union(*[p.keys() for p in prepared.values()]))

    cash = start_balance
    pos: dict[str, BtPos] = {}
    pending: dict[str, object] = {}   # symbol -> signal waiting for next bar open
    trades, curve = [], []
    halted_days = set()

    def close_pos(sym, px, ts, why, i):
        nonlocal cash
        p = pos.pop(sym)
        sign = 1 if p.side == "long" else -1
        if why == "liquidated":
            pnl, fill = -p.margin, p.liq
        else:
            fill = px * (1 - sign * slip)
            pnl = sign * (fill - p.entry) * p.qty - fee * fill * p.qty
        cash += pnl
        risk.record_close(ts, sym, pnl, pnl < 0 and why in ("stop", "managed stop hit"), step)
        trades.append({"symbol": sym, "side": p.side, "open_time": p.opened_ts, "close_time": ts,
                       "entry": p.entry, "exit": fill, "qty": p.qty, "margin": p.margin,
                       "pnl": pnl - p.entry_fee,  # include the opening fee
                       "fees": p.entry_fee + (0.0 if why == "liquidated" else fee * fill * p.qty),
                       "reason": why, "bars": i - p.opened_i})

    for i, ts in enumerate(timeline):
        # 1) fill pending entries at this bar's open
        for sym, sig in list(pending.items()):
            row = prepared[sym].get(ts)
            pending.pop(sym)
            if row is None or sym in pos:
                continue
            equity = cash
            ok, _ = risk.can_open(ts, sym, len(pos), equity)
            if not ok:
                continue
            sign = 1 if sig.side == "long" else -1
            dist = abs(sig.entry - sig.stop)
            if maker:
                # post-only limit at the signal close; only counts as filled if price trades
                # THROUGH it (a touch alone is not enough), otherwise the order times out
                # (a post-only order would be rejected if the market already opened past it)
                fill = sig.entry
                if sign * (row["open"] - fill) < 0:
                    continue
                through = row["low"] < fill if sig.side == "long" else row["high"] > fill
                if not through:
                    continue
                entry_fee = cfg.fees.maker
            else:
                fill = row["open"] * (1 + sign * slip)
                if sign * (fill - sig.entry) > 0.5 * dist:
                    continue
                entry_fee = fee
            stop = fill - sign * dist
            tp = fill + sign * dist * cfg.strategy.reward_risk
            used = sum(p.margin for p in pos.values())
            sizing, _ = risk.size(fill, stop, sig.side, sig.strength, cash - used, equity)
            if sizing is None:
                continue
            cash -= entry_fee * sizing.notional
            risk.record_open()
            pos[sym] = BtPos(sig.side, sizing.qty, fill, stop, stop, tp, fill,
                             liquidation_price(fill, sig.side, cfg.leverage,
                                               cfg.risk.maintenance_margin_rate),
                             sizing.margin, entry_fee * sizing.notional, i, ts)

        # 2) manage open positions against this bar's range
        for sym in list(pos):
            row, p = prepared[sym].get(ts), pos[sym]
            if row is None:
                continue
            sign = 1 if p.side == "long" else -1
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            adverse, favorable = (l, h) if p.side == "long" else (h, l)
            if sign * (o - p.liq) <= 0:
                close_pos(sym, p.liq, ts, "liquidated", i)
            elif sign * (o - p.stop) <= 0:
                close_pos(sym, o, ts, "stop", i)            # gapped through stop
            elif sign * (adverse - p.stop) <= 0:
                why = "stop" if p.stop == p.initial_stop else "managed stop hit"
                close_pos(sym, p.stop, ts, why, i)
            elif maker and p.opened_i == i:
                # maker fill bar: we can't know if the high/low that would count for us came
                # before or after the fill, so only the stop is checked on this bar
                continue
            elif sign * (favorable - p.tp) >= 0:
                close_pos(sym, p.tp, ts, "take-profit", i)
            else:
                p.best = max(p.best, h) if p.side == "long" else min(p.best, l)
                d = manage_position(p.side, p.entry, p.stop, p.initial_stop, c, p.best,
                                    row["atr"], i - p.opened_i, cfg)
                if d.new_stop is not None:
                    p.stop = d.new_stop
                if d.close:
                    close_pos(sym, c, ts, d.reason, i)

        # 3) equity, daily loss limit
        upnl = 0.0
        for sym, p in pos.items():
            if ts in prepared[sym]:
                sign = 1 if p.side == "long" else -1
                upnl += sign * (prepared[sym][ts]["close"] - p.entry) * p.qty
        equity = cash + upnl
        if risk.daily_loss_hit(ts, equity):
            halted_days.add(risk.state.day_key)
            pending.clear()
            for sym in list(pos):
                if ts in prepared[sym]:
                    close_pos(sym, prepared[sym][ts]["close"], ts, "daily loss limit", i)
            equity = cash
        curve.append((ts, equity))
        if risk.state.killed:
            break

        # 4) new signals on this closed bar, filled next bar
        for sym, p in prepared.items():
            if sym in pos or ts not in p:
                continue
            sig = signal_at(p[ts], cfg)
            if sig:
                pending[sym] = sig

    return summarize(trades, curve, start_balance, halted_days, risk.state.killed)


def summarize(trades, curve, start_balance, halted_days, killed):
    tdf = pd.DataFrame(trades)
    cdf = pd.DataFrame(curve, columns=["ts", "equity"])
    stats = {"start_balance": start_balance,
             "end_equity": float(cdf["equity"].iloc[-1]) if len(cdf) else start_balance,
             "trades": len(tdf), "days_daily_limit_hit": len(halted_days),
             "stopped_by_min_equity": killed}
    if len(cdf):
        peak = cdf["equity"].cummax()
        stats["max_drawdown_pct"] = float(((cdf["equity"] - peak) / peak).min() * 100)
        stats["days"] = round((cdf["ts"].iloc[-1] - cdf["ts"].iloc[0]) / 86_400_000, 1)
    if len(tdf):
        wins, losses = tdf[tdf.pnl > 0], tdf[tdf.pnl <= 0]
        stats.update({
            "win_rate_pct": round(len(wins) / len(tdf) * 100, 1),
            "avg_win": round(float(wins.pnl.mean()), 2) if len(wins) else 0.0,
            "avg_loss": round(float(losses.pnl.mean()), 2) if len(losses) else 0.0,
            "profit_factor": round(float(wins.pnl.sum() / -losses.pnl.sum()), 2)
            if len(losses) and losses.pnl.sum() < 0 else float("inf"),
            "liquidations": int((tdf.reason == "liquidated").sum()),
            "fees_paid": round(float(tdf.fees.sum()), 2),
            "exit_reasons": tdf.reason.value_counts().to_dict(),
        })
    stats["return_pct"] = round((stats["end_equity"] / start_balance - 1) * 100, 1)
    return stats, tdf, cdf


def synthetic(n_bars: int = 20_000, seed: int = 0, start_price: float = 60_000.0,
              step_ms: int = 300_000) -> pd.DataFrame:
    """Random-walk candles with trending/choppy regimes, for tests only (not real data)."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.0004, n_bars // 500 + 1), 500)[:n_bars]
    rets = drift + rng.standard_t(4, n_bars) * 0.0012
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, 0.0008, n_bars)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    vol = rng.lognormal(3, 0.6, n_bars) * (1 + 20 * np.abs(rets))
    ts = 1_700_000_000_000 // step_ms * step_ms + np.arange(n_bars) * step_ms
    return pd.DataFrame({"ts": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": vol})
