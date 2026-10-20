"""Real-time loop: read prices, manage open trades, open new ones, enforce risk limits."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .exchange import Position
from .risk import RiskManager, RiskState
from .strategy import TF_MS, manage_position, prepare, signal_at

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


class Engine:
    def __init__(self, cfg: Config, broker):
        self.cfg = cfg
        self.broker = broker
        self.tf_ms = TF_MS[cfg.timeframe]
        self.state_path = Path(cfg.state_file)
        self.trades: dict[str, Trade] = {}
        self.last_bar: dict[str, int] = {}
        self.atr: dict[str, float] = {}
        self.risk = RiskManager(cfg, self._load_state())

    # ---------- persistence ----------
    def _load_state(self) -> RiskState:
        if not self.state_path.exists():
            return RiskState()
        d = json.loads(self.state_path.read_text())
        self.trades = {k: Trade(**v) for k, v in d.get("trades", {}).items()}
        return RiskState.from_dict(d.get("risk", {}))

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"risk": self.risk.state.to_dict(),
                                   "trades": {k: asdict(v) for k, v in self.trades.items()}},
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

        equity, free = b.balance()
        positions = b.positions()
        self._forget_closed(now, positions)

        if self.risk.daily_loss_hit(now, equity) or self.risk.state.killed:
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

    # ---------- helpers ----------
    def _forget_closed(self, now: int, positions: dict[str, Position]) -> None:
        """A tracked trade with no exchange position was closed by the exchange stop/TP."""
        for sym in [s for s in self.trades if s not in positions]:
            t = self.trades.pop(sym)
            px = self.broker.price(sym)
            sign = 1 if t.side == "long" else -1
            pnl = sign * (px - t.entry) * t.qty
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
        log.info("CLOSE %s %s qty=%s @ %.4f (%s) pnl≈%.2f", sym, p.side, p.qty, fill, why, pnl)
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
        if sym in positions:
            return
        sig = signal_at(row, self.cfg)
        if sig:
            self._try_open(sym, sig, now, positions)

    def _try_open(self, sym, sig, now, positions) -> None:
        equity, free = self.broker.balance()
        ok, why = self.risk.can_open(now, sym, len(positions), equity)
        if not ok:
            log.info("%s signal skipped: %s", sym, why)
            return
        price = self.broker.price(sym)
        dist = abs(sig.entry - sig.stop)
        sign = 1 if sig.side == "long" else -1
        if sign * (price - sig.entry) > 0.5 * dist:
            log.info("%s signal skipped: price already ran %.4f past signal", sym, price - sig.entry)
            return
        stop = price - sign * dist
        tp = price + sign * dist * self.cfg.strategy.reward_risk
        sizing, why = self.risk.size(price, stop, sig.side, sig.strength, free, equity)
        if sizing is None:
            log.info("%s signal skipped: %s", sym, why)
            return
        qty = self.broker.round_qty(sym, sizing.qty)
        if qty <= 0:
            log.info("%s signal skipped: size below exchange minimum", sym)
            return

        fill = self.broker.open(sym, sig.side, qty, stop, tp)
        log.info("OPEN %s %s qty=%s @ %.4f stop=%.4f tp=%.4f margin≈%.2f strength=%.2f (%s)",
                 sym, sig.side, qty, fill, stop, tp, sizing.margin, sig.strength, sig.reason)
        self.trades[sym] = Trade(sig.side, qty, fill, stop, stop, tp, fill, now)

        # safety: if the exchange's real liquidation price sits inside our stop, get out now
        pos = self.broker.positions().get(sym)
        if pos and pos.liq_price > 0 and sign * (stop - pos.liq_price) <= 0:
            log.error("%s liquidation %.4f is before stop %.4f, closing immediately",
                      sym, pos.liq_price, stop)
            self._close(sym, pos, now, "liq safety")

    def _manage(self, sym: str, p: Position, now: int) -> None:
        t = self.trades.get(sym)
        if t is None:
            # position we did not open (manual or from before a crash): give it a tight managed stop
            sign = 1 if p.side == "long" else -1
            stop = p.entry * (1 - sign * self.cfg.risk.max_stop_pct)
            t = Trade(p.side, p.qty, p.entry, stop, stop, p.entry, p.entry, now)
            self.trades[sym] = t
            log.warning("%s adopted unknown position, managed stop %.4f", sym, stop)
        price = self.broker.price(sym)
        t.best = max(t.best, price) if t.side == "long" else min(t.best, price)
        bars = (now - t.opened_ms) // self.tf_ms
        d = manage_position(t.side, t.entry, t.stop, t.initial_stop, price, t.best,
                            self.atr.get(sym, 0.0), bars, self.cfg)
        if d.new_stop is not None:
            t.stop = d.new_stop
        if d.close:
            self._close(sym, p, now, d.reason)
