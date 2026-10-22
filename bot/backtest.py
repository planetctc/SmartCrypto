"""Bar-by-bar backtest using the same strategy and risk code as live trading.

Conservative assumptions: entries fill at the next bar's open plus slippage, taker fee on both
sides, if a bar touches both stop and take-profit the stop is assumed to hit first, and a bar
that opens beyond the stop fills at that worse open price (gap). Funding fees are ignored.

When 1-minute bars are passed as `fine`, stops and take-profits are checked minute by minute
inside each bar instead, so "both touched in one bar" only stays ambiguous within one minute.
That matters for fast trades whose stop and target often sit inside a single 5m/15m bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .risk import RiskManager, effective_leverage, liquidation_price
from .strategy import TF_MS, manage_position, prepare, scales_in, signal_at, trade_plan


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
    entry: float          # average entry (changes when adding)
    initial_stop: float
    stop: float
    tp: float
    best: float
    liq: float
    margin: float
    entry_fee: float
    opened_i: int
    opened_ts: int
    first_entry: float = 0.0
    add_price: float = 0.0     # 补仓 level (0 = none planned)
    add_plan: float = 0.0
    add_qty: float = 0.0
    add_fill: float = 0.0
    add_state: str = ""        # "" waiting, "open" after 补仓, "done" after 平加仓
    adds: int = 0
    tps: list = field(default_factory=list)
    tp_qtys: list = field(default_factory=list)
    realized: float = 0.0      # partial-close profit, fees included (already in cash)
    planned_margin: float = 0.0  # margin of both batches together, as sized
    votes: str = ""              # score signal: indicators that agreed, comma separated


def minute_bars(df: pd.DataFrame, tf: str) -> dict[int, np.ndarray]:
    """1m bars grouped by the start of the tf bar they belong to: {bar ts: [[o, h, l, c], ...]}."""
    step = TF_MS[tf]
    arr = df[["open", "high", "low", "close"]].to_numpy()
    keys = (df["ts"].to_numpy() // step) * step
    cuts = np.flatnonzero(np.diff(keys)) + 1
    return {int(keys[g[0]]): arr[g] for g in np.split(np.arange(len(df)), cuts) if len(g)}


def run_backtest(cfg: Config, frames: dict[str, pd.DataFrame], start_balance: float = 100.0,
                 fine: dict[str, pd.DataFrame] | None = None):
    fee, slip = cfg.fees.taker, cfg.fees.slippage
    step = TF_MS[cfg.timeframe]
    # minute-by-minute exits (not with 补仓, whose rules are written per bar)
    minutes = {sym: minute_bars(df, cfg.timeframe) for sym, df in fine.items()} \
        if fine and not scales_in(cfg) else {}
    risk = RiskManager(cfg)

    maker = cfg.entry_order == "maker"
    tp_maker = cfg.tp_order == "maker"
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
        elif why == "take-profit" and tp_maker:
            fill = px  # resting limit: exact price, maker fee
            pnl = sign * (fill - p.entry) * p.qty - cfg.fees.maker * fill * p.qty
        else:
            fill = px * (1 - sign * slip)
            pnl = sign * (fill - p.entry) * p.qty - fee * fill * p.qty
        cash += pnl
        pnl += p.realized
        risk.record_close(ts, sym, pnl, pnl < 0 and why in ("stop", "managed stop hit"), step)
        trades.append({"symbol": sym, "side": p.side, "open_time": p.opened_ts, "close_time": ts,
                       "entry": p.entry, "exit": fill, "qty": p.qty, "margin": p.planned_margin,
                       "pnl": pnl - p.entry_fee,  # include the opening fee
                       "fees": p.entry_fee + (0.0 if why == "liquidated" else
                                              (cfg.fees.maker if why == "take-profit" and tp_maker
                                               else fee) * fill * p.qty),
                       "reason": why, "bars": i - p.opened_i, "adds": p.adds, "votes": p.votes})

    def scale(p, sym, row, i):
        """补仓 / 平加仓 / 分批止盈 inside one bar, same rules as live. Adverse moves are
        assumed to come first, and nothing favourable counts on the bar of a 补仓."""
        nonlocal cash
        sign = 1 if p.side == "long" else -1
        adverse, favorable = (row["low"], row["high"]) if p.side == "long" else \
            (row["high"], row["low"])
        cost = fee + slip
        if (p.add_plan > 0 and not p.add_state and p.stop == p.initial_stop
                and sign * (adverse - p.add_price) <= 0):
            fill = p.add_price * (1 + sign * slip)
            cash -= fee * fill * p.add_plan
            p.entry_fee += fee * fill * p.add_plan
            p.entry = (p.entry * p.qty + fill * p.add_plan) / (p.qty + p.add_plan)
            p.qty += p.add_plan
            p.margin += fill * p.add_plan / cfg.leverage
            p.add_qty, p.add_fill, p.add_state, p.adds = p.add_plan, fill, "open", 1
            lev = effective_leverage(cfg, p.entry * p.qty, cash, cfg.leverage)
            p.liq = liquidation_price(p.entry, p.side, lev, cfg.risk.maintenance_margin_rate)
            return True
        if p.add_state == "open":
            level = p.entry * (1 + sign * 2 * cost)
            if sign * (favorable - level) >= 0:
                gain = sign * (level - p.add_fill) * p.add_qty - cost * (level + p.add_fill) * p.add_qty
                cash += gain + fee * p.add_fill * p.add_qty  # entry fee was paid already
                p.realized += gain
                p.qty -= p.add_qty
                p.entry_fee -= fee * p.add_fill * p.add_qty
                p.margin -= p.add_fill * p.add_qty / cfg.leverage
                p.entry = p.first_entry
                p.add_qty, p.add_state = 0.0, "done"
            return False
        while p.tps and sign * (favorable - p.tps[0]) >= 0:
            level, q = p.tps.pop(0), min(p.tp_qtys.pop(0), p.qty)
            if q <= 0 or q >= p.qty:
                continue
            fill = level * (1 - sign * slip)
            gain = sign * (fill - p.entry) * q - fee * fill * q
            cash += gain
            p.realized += gain - p.entry_fee * q / p.qty
            p.entry_fee -= p.entry_fee * q / p.qty
            p.margin -= p.margin * q / p.qty
            p.qty -= q
            be = p.first_entry * (1 + sign * 2 * cost)
            if sign * (be - p.stop) > 0:
                p.stop = be
        return False

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
            plan = trade_plan(sig.side, fill, dist, cfg)
            stop, tp = plan.stop, plan.take_profits[-1]
            used = sum(p.margin for p in pos.values())
            sizing, _ = risk.size(fill, stop, sig.side, sig.strength, cash - used, equity)
            if sizing is None:
                continue
            st = cfg.strategy
            qty = sizing.qty * st.scale_in_frac
            cash -= entry_fee * fill * qty
            risk.record_open()
            pos[sym] = BtPos(sig.side, qty, fill, stop, stop, tp, fill,
                             liquidation_price(fill, sig.side,
                                               effective_leverage(cfg, sizing.notional, equity,
                                                                  cfg.leverage),
                                               cfg.risk.maintenance_margin_rate),
                             sizing.margin * st.scale_in_frac, entry_fee * fill * qty, i, ts,
                             fill, plan.add_price if scales_in(cfg) else 0.0,
                             sizing.qty - qty if scales_in(cfg) else 0.0,
                             tps=plan.take_profits[:-1],
                             tp_qtys=[qty * f for f in st.take_profit_split],
                             planned_margin=sizing.margin, votes=",".join(sig.votes))

        # 2) manage open positions against this bar's range (or its minutes)
        for sym in list(pos):
            row, p = prepared[sym].get(ts), pos[sym]
            if row is None:
                continue
            sign = 1 if p.side == "long" else -1
            fill_bar = maker and p.opened_i == i
            sub = minutes.get(sym, {}).get(ts) if minutes else None
            if sub is None:
                sub = np.array([[row["open"], row["high"], row["low"], row["close"]]])
                start = 0
            elif fill_bar:
                # the limit filled in the first minute that traded through it
                lim = p.first_entry
                through = sub[:, 2] < lim if p.side == "long" else sub[:, 1] > lim
                start = int(np.argmax(through)) if through.any() else len(sub) - 1
            else:
                start = 0
            o = sub[start][0]
            added = sign * (o - p.liq) > 0 and sign * (o - p.stop) > 0 and \
                not fill_bar and scale(p, sym, row, i)
            done = False
            for k in range(start, len(sub)):
                o, h, l, c = sub[k]
                adverse, favorable = (l, h) if p.side == "long" else (h, l)
                if sign * (o - p.liq) <= 0 and not (fill_bar and k == start):
                    close_pos(sym, p.liq, ts, "liquidated", i)
                elif sign * (o - p.stop) <= 0 and not (fill_bar and k == start):
                    close_pos(sym, o, ts, "stop", i)            # gapped through stop
                elif sign * (adverse - p.stop) <= 0:
                    why = "stop" if p.stop == p.initial_stop else "managed stop hit"
                    close_pos(sym, p.stop, ts, why, i)
                elif added or (fill_bar and k == start):
                    # fill (or 补仓) minute: we can't know if the high/low that would count
                    # for us came before or after it, so only the stop is checked there
                    continue
                elif (sign * (favorable - p.tp) > 0 if tp_maker else sign * (favorable - p.tp) >= 0) \
                        and p.add_state != "open":
                    # a resting limit only counts as filled when price trades through it
                    close_pos(sym, p.tp, ts, "take-profit", i)
                else:
                    p.best = max(p.best, h) if p.side == "long" else min(p.best, l)
                    continue
                done = True
                break
            if done or (len(sub) == 1 and (added or fill_bar)):
                continue
            d = manage_position(p.side, p.first_entry, p.stop, p.initial_stop, row["close"],
                                p.best, row["atr"], i - p.opened_i, cfg)
            if d.new_stop is not None:
                p.stop = d.new_stop
            if d.close:
                close_pos(sym, row["close"], ts, d.reason, i)

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
