"""Real-time loop: read prices, manage open trades, open new ones, enforce risk limits."""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .exchange import Position
from .risk import RiskManager, RiskState, effective_leverage
from .strategy import TF_MS, manage_position, prepare, pyramid_add_qty, signal_at

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
    avg_entry: float = 0.0   # average entry after adds (0 = same as entry)
    base_qty: float = 0.0    # size of the first entry (0 = same as qty)
    adds: int = 0
    last_add: float = 0.0    # price of the latest add (0 = entry)


@dataclass
class PendingEntry:
    order_id: str
    side: str
    qty: float
    price: float
    stop: float
    take_profit: float
    expires_ms: int


class Engine:
    def __init__(self, cfg: Config, broker):
        self.cfg = cfg
        self.broker = broker
        self.tf_ms = TF_MS[cfg.timeframe]
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
        self.trades = {k: Trade(**v) for k, v in d.get("trades", {}).items()}
        self.pending = {k: PendingEntry(**v) for k, v in d.get("pending", {}).items()}
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
                                 pos.entry, now)
        log.info("FILLED %s %s qty=%s @ %.10g stop=%.10g tp=%.10g",
                 sym, pe.side, pos.qty, pos.entry, pe.stop, pe.take_profit)
        self._liq_safety(sym, pos, pe.stop, now)

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
            px = self.broker.price(sym)
            sign = 1 if t.side == "long" else -1
            pnl = sign * (px - (t.avg_entry or t.entry)) * t.qty
            log.info("%s position gone (exchange stop/TP), approx pnl %.2f", sym, pnl)
            self.risk.record_close(now, sym, pnl, pnl < 0, self.tf_ms)

    def _close_all(self, now: int, positions: dict[str, Position]) -> None:
        for sym, p in positions.items():
            self._close(sym, p, now, "risk stop")

    def _close(self, sym: str, p: Position, now: int, why: str) -> None:
        fill = self.broker.close(sym, p)
        sign = 1 if p.side == "long" else -1
        fees = 2 * self.cfg.fees.taker * p.entry * p.qty
        pnl = sign * (fill - p.entry) * p.qty - fees
        log.info("CLOSE %s %s qty=%s @ %.10g (%s) pnl≈%.2f", sym, p.side, p.qty, fill, why, pnl)
        self.trades.pop(sym, None)
        stopped = pnl < 0 and why != "max hold time"
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

    def _try_open(self, sym, sig, now, positions) -> None:
        equity, free = self._balance()
        ok, why = self.risk.can_open(now, sym, len(positions) + len(self.pending), equity)
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
            price = min(price, sig.entry) if sig.side == "long" else max(price, sig.entry)
        if maker:
            price = self._round(sym, price, sig.side)
        stop = self._round(sym, price - sign * dist, sig.side)
        tp = self._round(sym, price + sign * dist * self.cfg.strategy.reward_risk, sig.side)
        lev = self.broker.leverage_for(sym) if hasattr(self.broker, "leverage_for") else None
        sizing, why = self.risk.size(price, stop, sig.side, sig.strength, free, equity, lev)
        if sizing is None:
            log.info("%s signal skipped: %s", sym, why)
            return
        qty = self.broker.round_qty(sym, sizing.qty)
        if qty <= 0:
            log.info("%s signal skipped: size below exchange minimum", sym)
            return

        if maker:
            oid = self.broker.open_maker(sym, sig.side, qty, price, stop, tp)
            self.pending[sym] = PendingEntry(oid, sig.side, qty, price, stop, tp,
                                             now + self.cfg.maker_timeout_seconds * 1000)
            log.info("MAKER %s %s qty=%s @ %.10g stop=%.10g tp=%.10g margin≈%.2f strength=%.2f (%s)",
                     sym, sig.side, qty, price, stop, tp, sizing.margin, sig.strength, sig.reason)
            return

        fill = self.broker.open(sym, sig.side, qty, stop, tp)
        self.risk.record_open()
        log.info("OPEN %s %s qty=%s @ %.10g stop=%.10g tp=%.10g margin≈%.2f strength=%.2f (%s)",
                 sym, sig.side, qty, fill, stop, tp, sizing.margin, sig.strength, sig.reason)
        self.trades[sym] = Trade(sig.side, qty, fill, stop, stop, tp, fill, now)
        pos = self.broker.positions().get(sym)
        if pos:
            self._liq_safety(sym, pos, stop, now)

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
        t.best = max(t.best, price) if t.side == "long" else min(t.best, price)
        bars = (now - t.opened_ms) // self.tf_ms
        d = manage_position(t.side, t.entry, t.stop, t.initial_stop, price, t.best,
                            self.atr.get(sym, 0.0), bars, self.cfg)
        if d.new_stop is not None:
            t.stop = d.new_stop
        if d.close:
            self._close(sym, p, now, d.reason)
        elif self.cfg.strategy.pyramid_max_adds:
            self._try_add(sym, t, price, now)

    def _try_add(self, sym: str, t: Trade, price: float, now: int) -> None:
        """平加仓: add to a winner whose stop is locked in profit (rules in pyramid_add_qty)."""
        lev = self.broker.leverage_for(sym) if hasattr(self.broker, "leverage_for") \
            else self.cfg.leverage
        avg = t.avg_entry or t.entry
        equity, free = self._balance()
        most = (t.qty + (t.base_qty or t.qty) * self.cfg.strategy.pyramid_add_frac) * price
        q = pyramid_add_qty(t.side, t.entry, t.initial_stop, t.stop, avg, t.qty,
                            t.base_qty or t.qty, t.adds, t.last_add or t.entry, price,
                            effective_leverage(self.cfg, most, equity, lev), self.cfg)
        q = self.broker.round_qty(sym, q) if q > 0 else 0.0
        if q <= 0:
            return
        if q * price / lev > free * 0.95:
            log.info("%s add skipped: not enough free margin", sym)
            return
        t.base_qty = t.base_qty or t.qty
        fill = self.broker.open(sym, t.side, q, self._round(sym, t.stop, t.side), t.take_profit)
        t.avg_entry = (avg * t.qty + fill * q) / (t.qty + q)
        t.qty += q
        t.adds += 1
        t.last_add = fill
        log.info("ADD %s %s qty=%s @ %.10g -> total qty=%s avg=%.10g stop=%.10g (add %d)",
                 sym, t.side, q, fill, t.qty, t.avg_entry, t.stop, t.adds)
        pos = self.broker.positions().get(sym)
        if pos:
            self._liq_safety(sym, pos, t.stop, now)
