"""Real-time loop: read prices, manage open trades, open new ones, enforce risk limits."""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .config import Config
from .exchange import Position
from .risk import RiskManager, RiskState
from .strategy import TF_MS, manage_position, prepare, scales_in, signal_at, trade_plan

log = logging.getLogger("bot.engine")

HISTORY_BARS = 300


@dataclass
class Trade:
    side: str
    qty: float
    entry: float
    initial_stop: float
    stop: float
    take_profit: float
    best: float
    opened_ms: int
    avg_entry: float = 0.0    # average entry while the 补仓 batch is open (0 = entry)
    add_price: float = 0.0    # 分批进场: 补仓 when price reaches this (0 = no second batch)
    add_plan: float = 0.0     # qty of the second batch
    add_qty: float = 0.0      # qty of the second batch that is open now
    add_fill: float = 0.0
    add_state: str = ""       # "" = waiting, "open" = 补仓 done, "done" = 平加仓 done
    tps: list = field(default_factory=list)      # 分批止盈 levels still to hit (exchange has the last)
    tp_qtys: list = field(default_factory=list)  # qty to close at each of those levels
    realized: float = 0.0     # profit already taken from partial closes
    tp_order_id: str = ""     # tp_order maker: id of the resting limit take-profit
    tp_tries: int = 0         # failed attempts to place it (bot watches the price after 3)


@dataclass
class PendingEntry:
    order_id: str
    side: str
    qty: float
    price: float
    stop: float
    take_profit: float
    expires_ms: int
    plan: dict = field(default_factory=dict)  # 补仓 / 分批止盈 settings for the Trade


class Engine:
    def __init__(self, cfg: Config, broker):
        self.cfg = cfg
        self.broker = broker
        self.tf_ms = TF_MS[cfg.timeframe]
        self.limit_tp = cfg.tp_order == "maker"
        self.state_path = Path(cfg.state_file)
        self.trades: dict[str, Trade] = {}
        self.pending: dict[str, PendingEntry] = {}
        self.last_bar: dict[str, int] = {}
        self.atr: dict[str, float] = {}
        self.risk = RiskManager(cfg, self._load_state())

    # ---------- persistence ----------
    def _load_state(self) -> RiskState:
        if not self.state_path.exists():
            return RiskState()
        d = json.loads(self.state_path.read_text())
        known = lambda cls, v: cls(**{k: x for k, x in v.items()  # noqa: E731
                                      if k in {f.name for f in fields(cls)}})
        self.trades = {k: known(Trade, v) for k, v in d.get("trades", {}).items()}
        self.pending = {k: known(PendingEntry, v) for k, v in d.get("pending", {}).items()}
        return RiskState.from_dict(d.get("risk", {}))

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"risk": self.risk.state.to_dict(),
                                   "trades": {k: asdict(v) for k, v in self.trades.items()},
                                   "pending": {k: asdict(v) for k, v in self.pending.items()}},
                                  indent=2))
        tmp.replace(self.state_path)

    # ---------- main loop ----------
    def run(self) -> None:
        for s in self.cfg.symbols:
            self.broker.setup_symbol(s)
        log.info("started: mode=%s symbols=%s leverage=%sx margin %s-%s USDT",
                 self.cfg.mode, self.cfg.symbols, self.cfg.leverage,
                 self.cfg.min_margin_usd, self.cfg.max_margin_usd)
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("loop error, will retry")
            time.sleep(self.cfg.loop_seconds)

    def step(self) -> None:
        b, now = self.broker, self.broker.now_ms()
        if hasattr(b, "tick"):
            b.tick()
            for sym, pnl, why in b.closed_by_exchange:
                log.info("%s closed by %s pnl=%.2f", sym, why, pnl)
            b.closed_by_exchange.clear()

        equity, free = self._balance()
        if equity <= 0:
            return  # balance unreadable/empty: do nothing rather than trip the kill switch
        positions = b.positions()
        self._check_pending(now, positions)
        self._forget_closed(now, positions)

        if self.risk.daily_loss_hit(now, equity) or self.risk.state.killed:
            for sym in list(self.pending):
                self._cancel_pending(sym, now, positions)
            if positions:
                log.warning("RISK STOP (equity %.2f, day start %.2f): closing everything",
                            equity, self.risk.state.day_start_equity)
                self._close_all(now, positions)
            self._save_state()
            return

        for sym in self.cfg.symbols:
            self._refresh_bars(sym, now, positions)
            if sym in positions:
                self._manage(sym, positions[sym], now)
        self._save_state()

    def _balance(self) -> tuple[float, float]:
        """Equity and free balance as the risk rules see them. With capital_usd set, the bot
        acts as if the account held only that amount plus its own profit/loss since it started
        (so a demo account with 50,000 test USDT behaves like a 100 USDT account)."""
        equity, free = self.broker.balance()
        cap = self.cfg.capital_usd
        if cap <= 0:
            return equity, free
        st = self.risk.state
        if equity <= 0:
            if not getattr(self, "_warned_zero", False):
                self._warned_zero = True
                log.warning("账户余额读到 0，机器人不会开单。请确认模拟盘合约账户里有 USDT。")
            return 0.0, 0.0
        if st.equity_baseline <= 0:
            st.equity_baseline = equity
            log.info("capital_usd=%.2f: real equity %.2f taken as baseline", cap, equity)
        virtual = cap + (equity - st.equity_baseline)
        return virtual, min(free, virtual - (equity - free))

    # ---------- maker entries ----------
    def _check_pending(self, now: int, positions: dict[str, Position]) -> None:
        for sym in list(self.pending):
            pe = self.pending[sym]
            status, filled, _ = self.broker.order_status(sym, pe.order_id)
            if status == "open" and now < pe.expires_ms:
                continue
            if status == "open":
                self._cancel_pending(sym, now, positions)
                continue
            self.pending.pop(sym)
            if filled > 0:
                self._on_filled(sym, pe, now)
            else:
                log.info("%s maker entry %s not filled (%s)", sym, pe.order_id, status)
        if self.trades.keys() - positions.keys():
            positions.clear()
            positions.update(self.broker.positions())

    def _cancel_pending(self, sym: str, now: int, positions: dict[str, Position]) -> None:
        pe = self.pending.pop(sym)
        self.broker.cancel_order(sym, pe.order_id)
        status, filled, _ = self.broker.order_status(sym, pe.order_id)
        if filled > 0:
            log.info("%s maker entry partly filled %s/%s before cancel", sym, filled, pe.qty)
            self._on_filled(sym, pe, now)
        else:
            log.info("%s maker entry %s cancelled (timeout)", sym, pe.order_id)

    def _on_filled(self, sym: str, pe: PendingEntry, now: int) -> None:
        pos = self.broker.positions().get(sym)
        if pos is None:
            return
        self.risk.record_open()
        self.trades[sym] = Trade(pe.side, pos.qty, pos.entry, pe.stop, pe.stop, pe.take_profit,
                                 pos.entry, now, **pe.plan)
        log.info("FILLED %s %s qty=%s @ %.10g stop=%.10g tp=%.10g",
                 sym, pe.side, pos.qty, pos.entry, pe.stop, pe.take_profit)
        self._liq_safety(sym, pos, pe.stop, now)
        if sym in self.trades:
            self._place_tp(sym, self.trades[sym])

    def _place_tp(self, sym: str, t: Trade) -> None:
        """tp_order maker: rest a reduce-only limit at the take-profit (maker fee). If the
        exchange refuses it, the bot itself closes at market when price reaches the level."""
        if not self.limit_tp or t.tp_order_id or t.tp_tries >= 3:
            return
        try:
            t.tp_order_id = self.broker.place_tp(sym, t.side, t.qty, t.take_profit)
            log.info("TP-LIMIT %s %s qty=%s @ %.10g", sym, t.side, t.qty, t.take_profit)
        except Exception as e:  # noqa: BLE001 - any refusal falls back to watching the price
            t.tp_tries += 1
            log.warning("%s limit take-profit not placed (%s); bot will close at %.10g itself",
                        sym, e, t.take_profit)

    def _cancel_tp(self, sym: str, t: Trade | None) -> None:
        if t is not None and t.tp_order_id:
            self.broker.cancel_order(sym, t.tp_order_id)
            t.tp_order_id = ""

    def _liq_safety(self, sym: str, pos: Position, stop: float, now: int) -> None:
        """If the exchange's real liquidation price sits inside our stop, get out now."""
        sign = 1 if pos.side == "long" else -1
        if pos.liq_price > 0 and sign * (stop - pos.liq_price) <= 0:
            log.error("%s liquidation %.10g is before stop %.10g, closing immediately",
                      sym, pos.liq_price, stop)
            self._close(sym, pos, now, "liq safety")

    # ---------- helpers ----------
    def _round(self, sym: str, px: float, side: str) -> float:
        """BTC/ETH prices in whole points like a trader would set them. Long rounds down and
        short rounds up, so entries never chase and stops only get a little wider."""
        if sym.split("/")[0] not in self.cfg.whole_price_coins:
            return px
        return float(math.floor(px) if side == "long" else math.ceil(px))

    def _forget_closed(self, now: int, positions: dict[str, Position]) -> None:
        """A tracked trade with no exchange position was closed by the exchange stop/TP."""
        for sym in [s for s in self.trades if s not in positions]:
            t = self.trades.pop(sym)
            self._cancel_tp(sym, t)  # stop hit: the limit take-profit must not stay behind
            px = self.broker.price(sym)
            sign = 1 if t.side == "long" else -1
            avg = t.avg_entry or t.entry
            fees = self.cfg.fees.taker * (avg + px) * t.qty
            pnl = sign * (px - avg) * t.qty - fees + t.realized
            log.info("%s position gone (exchange stop/TP), approx pnl %.2f", sym, pnl)
            self.risk.record_close(now, sym, pnl, pnl < 0, self.tf_ms)

    def _close_all(self, now: int, positions: dict[str, Position]) -> None:
        for sym, p in positions.items():
            self._close(sym, p, now, "risk stop")

    def _close(self, sym: str, p: Position, now: int, why: str) -> None:
        self._cancel_tp(sym, self.trades.get(sym))
        fill = self.broker.close(sym, p)
        sign = 1 if p.side == "long" else -1
        fees = 2 * self.cfg.fees.taker * p.entry * p.qty
        t = self.trades.pop(sym, None)
        pnl = sign * (fill - p.entry) * p.qty - fees + (t.realized if t else 0.0)
        log.info("CLOSE %s %s qty=%s @ %.10g (%s) pnl≈%.2f", sym, p.side, p.qty, fill, why, pnl)
        stopped = pnl < 0 and why not in ("max hold time", "replaced by new signal")
        self.risk.record_close(now, sym, pnl, stopped, self.tf_ms)

    def _refresh_bars(self, sym: str, now: int, positions: dict[str, Position]) -> None:
        cur_bar = (now // self.tf_ms) * self.tf_ms - self.tf_ms  # last fully closed bar
        if self.last_bar.get(sym) == cur_bar:
            return
        df = self.broker.ohlcv(sym, self.cfg.timeframe, HISTORY_BARS)
        trend_bars = self.cfg.strategy.trend_ema * 3 + 10
        dft = self.broker.ohlcv(sym, self.cfg.trend_timeframe, trend_bars)
        if df.empty or dft.empty:
            return
        prepared = prepare(df, dft, self.cfg)
        row = prepared.iloc[-1]
        self.last_bar[sym] = int(row["ts"])
        self.atr[sym] = float(row["atr"])
        if sym in positions or sym in self.pending:
            return
        sig = signal_at(row, self.cfg)
        if sig:
            self._try_open(sym, sig, now, positions)

    def _make_room(self, sym: str, now: int, positions: dict[str, Position], equity: float) -> int:
        """replace_on_new_signal: free slots for a new signal on `sym` by cancelling other
        coins' waiting limit entries and closing their trades that are not yet protected
        (stop still below break-even). Returns how many slots were freed. Does nothing if
        the new trade would be refused anyway (daily limit, pause, cooldown...)."""
        others = [s for s in self.pending if s != sym]
        for s in positions.keys() | self.trades.keys():
            t = self.trades.get(s)
            if s == sym or t is None:
                continue
            sign = 1 if t.side == "long" else -1
            if sign * (t.stop - t.entry) < 0:
                others.append(s)
        busy = len(positions.keys() | self.trades.keys()) + len(self.pending)
        if not others or not self.risk.can_open(now, sym, busy - len(others), equity)[0]:
            return 0
        for s in others:
            if s in self.pending:
                self._cancel_pending(s, now, positions)
            elif s in positions:
                log.info("%s new signal on %s: closing %s to take it", s, sym, s)
                self._close(s, positions.pop(s), now, "replaced by new signal")
            else:
                self.trades.pop(s, None)
        return len(others)

    def _try_open(self, sym, sig, now, positions) -> None:
        equity, free = self._balance()
        # positions was read at the start of this loop: count trades opened since then too
        busy = len(positions.keys() | self.trades.keys()) + len(self.pending)
        if self.cfg.replace_on_new_signal and busy >= self.cfg.max_open_positions:
            busy -= self._make_room(sym, now, positions, equity)
        ok, why = self.risk.can_open(now, sym, busy, equity)
        if not ok:
            log.info("%s signal skipped: %s", sym, why)
            return
        price = self.broker.price(sym)
        dist = abs(sig.entry - sig.stop)
        sign = 1 if sig.side == "long" else -1
        if sign * (price - sig.entry) > 0.5 * dist:
            log.info("%s signal skipped: price already ran %.10g past signal", sym, price - sig.entry)
            return
        maker = self.cfg.entry_order == "maker"
        if maker:  # rest at the signal close, never on the wrong side of the market
            price = self._round(sym, min(price, sig.entry) if sig.side == "long"
                                else max(price, sig.entry), sig.side)
        plan = trade_plan(sig.side, price, dist, self.cfg)
        stop = self._round(sym, plan.stop, sig.side)
        tps = [self._round(sym, x, sig.side) for x in plan.take_profits]
        tp = tps[-1]
        lev = self.broker.leverage_for(sym) if hasattr(self.broker, "leverage_for") else None
        sizing, why = self.risk.size(price, stop, sig.side, sig.strength, free, equity, lev)
        if sizing is None:
            log.info("%s signal skipped: %s", sym, why)
            return
        # 分批进场: open the first batch now, keep the rest for the 补仓 price
        s = self.cfg.strategy
        qty = self.broker.round_qty(sym, sizing.qty * s.scale_in_frac)
        if qty <= 0:
            log.info("%s signal skipped: size below exchange minimum", sym)
            return
        add = self.broker.round_qty(sym, sizing.qty - qty) if scales_in(self.cfg) else 0.0
        extra = {"add_price": self._round(sym, plan.add_price, sig.side) if add > 0 else 0.0,
                 "add_plan": add, "tps": tps[:-1],
                 "tp_qtys": [self.broker.round_qty(sym, qty * f) for f in s.take_profit_split]}
        margin = sizing.margin * qty / sizing.qty

        exch_tp = 0.0 if self.limit_tp else tp  # 0 = no trigger TP, a limit order goes in after
        if maker:
            oid = self.broker.open_maker(sym, sig.side, qty, price, stop, exch_tp)
            self.pending[sym] = PendingEntry(oid, sig.side, qty, price, stop, tp,
                                             now + self.cfg.maker_timeout_seconds * 1000, extra)
            log.info("MAKER %s %s qty=%s @ %.10g stop=%.10g tp=%.10g margin≈%.2f strength=%.2f (%s)",
                     sym, sig.side, qty, price, stop, tp, margin, sig.strength, sig.reason)
            self._log_plan(sym, extra, tps)
            return

        fill = self.broker.open(sym, sig.side, qty, stop, exch_tp)
        self.risk.record_open()
        log.info("OPEN %s %s qty=%s @ %.10g stop=%.10g tp=%.10g margin≈%.2f strength=%.2f (%s)",
                 sym, sig.side, qty, fill, stop, tp, margin, sig.strength, sig.reason)
        self._log_plan(sym, extra, tps)
        self.trades[sym] = Trade(sig.side, qty, fill, stop, stop, tp, fill, now, **extra)
        pos = self.broker.positions().get(sym)
        if pos:
            self._liq_safety(sym, pos, stop, now)
        if sym in self.trades:
            self._place_tp(sym, self.trades[sym])

    def _log_plan(self, sym: str, extra: dict, tps: list[float]) -> None:
        log.info("PLAN %s add=%s @ %.10g take-profits=%s", sym, extra["add_plan"],
                 extra["add_price"], "/".join(f"{x:.10g}" for x in tps))

    def _manage(self, sym: str, p: Position, now: int) -> None:
        t = self.trades.get(sym)
        if t is None and sym in self.pending:
            # our maker entry filled before the order status caught up: track it as ours
            self._on_filled(sym, self.pending.pop(sym), now)
            t = self.trades.get(sym)
            if t is None:
                return
        if t is None:
            # position we did not open (manual or from before a crash): give it a tight managed stop
            sign = 1 if p.side == "long" else -1
            stop = p.entry * (1 - sign * self.cfg.risk.max_stop_pct)
            t = Trade(p.side, p.qty, p.entry, stop, stop, p.entry, p.entry, now)
            self.trades[sym] = t
            log.warning("%s adopted unknown position, managed stop %.10g", sym, stop)
        price = self.broker.price(sym)
        if self.limit_tp and t.take_profit != t.entry:
            sign = 1 if t.side == "long" else -1
            if not t.tp_order_id and sign * (price - t.take_profit) >= 0:
                self._close(sym, p, now, "take-profit")  # limit refused: take it at market
                return
            self._place_tp(sym, t)
        t.best = max(t.best, price) if t.side == "long" else min(t.best, price)
        bars = (now - t.opened_ms) // self.tf_ms
        d = manage_position(t.side, t.entry, t.stop, t.initial_stop, price, t.best,
                            self.atr.get(sym, 0.0), bars, self.cfg)
        if d.new_stop is not None:
            t.stop = d.new_stop
        if d.close:
            self._close(sym, p, now, d.reason)
        else:
            self._scale(sym, t, price)

    def _scale(self, sym: str, t: Trade, price: float) -> None:
        """补仓, 平加仓 and 分批止盈 for an open trade (the video's way of trading)."""
        sign = 1 if t.side == "long" else -1
        cost = self.cfg.fees.taker + self.cfg.fees.slippage
        if (t.add_plan > 0 and not t.add_state and t.stop == t.initial_stop
                and sign * (t.add_price - price) >= 0):
            # 补仓: price came back into the entry zone, add the second batch, same stop
            fill = self.broker.open(sym, t.side, t.add_plan, t.stop, t.take_profit)
            t.avg_entry = (t.entry * t.qty + fill * t.add_plan) / (t.qty + t.add_plan)
            t.qty += t.add_plan
            t.add_qty, t.add_fill, t.add_state = t.add_plan, fill, "open"
            log.info("ADD %s %s qty=%s @ %.10g -> total qty=%s avg=%.10g stop=%.10g",
                     sym, t.side, t.add_plan, fill, t.qty, t.avg_entry, t.stop)
            return
        if t.add_state == "open":
            # 平加仓: back at the new average cost (plus fees), take the 补仓 batch off again
            if sign * (price - t.avg_entry * (1 + sign * 2 * cost)) < 0:
                return
            fill = self.broker.reduce(sym, t.side, t.add_qty)
            gain = sign * (fill - t.add_fill) * t.add_qty - cost * (fill + t.add_fill) * t.add_qty
            t.realized += gain
            t.qty -= t.add_qty
            t.add_qty, t.avg_entry, t.add_state = 0.0, 0.0, "done"
            log.info("REDUCE %s %s qty=%s @ %.10g back at cost, gain %.2f",
                     sym, t.side, t.add_plan, fill, gain)
            return
        while t.tps and sign * (price - t.tps[0]) >= 0:
            level, q = t.tps.pop(0), min(t.tp_qtys.pop(0), t.qty)
            if q <= 0 or q >= t.qty:
                continue
            fill = self.broker.reduce(sym, t.side, q)
            gain = sign * (fill - t.entry) * q - cost * (fill + t.entry) * q
            t.realized += gain
            t.qty -= q
            # 移动止损: once a take-profit is in, the rest can no longer turn into a loss
            be = self._round(sym, t.entry * (1 + sign * 2 * cost), "short" if sign > 0 else "long")
            if sign * (be - t.stop) > 0:
                t.stop = be
            log.info("PARTIAL %s %s qty=%s @ %.10g (take-profit %.10g) gain %.2f, stop now %.10g",
                     sym, t.side, q, fill, level, gain, t.stop)
