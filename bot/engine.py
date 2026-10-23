"""Real-time loop: read prices, manage open trades, open new ones, enforce risk limits."""
from __future__ import annotations

import csv
import json
import logging
import copy
import faulthandler
import math
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .exchange import Position
from .orderflow import flow_agrees
from .risk import RiskManager, RiskState
from .flowsignal import FRESH_MS, FlowFeed, flow_side

LATE_WINDOW_MS = 1_800_000   # 人工平法 after the max hold: best price of the last 30 minutes
LATE_REFRESH_MS = 600_000    #   re-aimed every 10 minutes
LATE_GIVEUP_MS = 3_600_000   #   market an hour after the max hold
ENTRY_RETRY_MS = 60_000      # 盘口模式: after an unfilled limit entry wait before the next one
PAUSE_MS = 4 * 3_600_000     # 盘口模式: a coin whose last 3+ live trades lost is paused this long
from .structure import round_step, snapshot
from .strategy import (TF_MS, Signal, manage_position, prepare, rsi, scales_in, signal_at,
                       trade_plan)

log = logging.getLogger("bot.engine")

HISTORY_BARS = 300


def sign_of(side: str) -> int:
    return 1 if side == "long" else -1


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
    reason: str = ""          # why it was opened and what the bot saw, for the trade journal
    kind: str = ""            # setup that opened it (breakout / bounce ...), learned per kind
    tps: list = field(default_factory=list)      # 分批止盈 levels still to hit (exchange has the last)
    tp_qtys: list = field(default_factory=list)  # qty to close at each of those levels
    realized: float = 0.0     # profit already taken from partial closes
    tp_order_id: str = ""     # tp_order maker: id of the resting limit take-profit
    tp_tries: int = 0         # failed attempts to place it (bot watches the price after 3)
    exit_px: float = 0.0      # 盘口模式: price of the resting limit exit after the hold time
    exit_kind: str = ""       #   profit / breakeven / late: which 人工平法 step that exit is
    exit_since: int = 0       #   when it was placed
    tf: str = ""              # 快慢一起: entry timeframe that opened it (its hold time and stops)


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
        self.flow_mode = cfg.strategy.signal_type == "flow"
        self.flow_verified = False
        self.entry_wait: dict[str, int] = {}   # 盘口模式: no new limit entry before this time
        self.coin_pause: dict[str, int] = {}   # 盘口模式: coin losing live, paused until
        self.feed: FlowFeed | None = None
        self.risk = RiskManager(cfg, self._load_state())
        self.lanes = {tf: lane_config(cfg, tf) for tf in cfg.scan_timeframes
                      if tf in TREND_OF and tf != cfg.timeframe}

    @contextmanager
    def _as_lane(self, tf: str):
        """Run with the settings of entry timeframe tf (its candles, hold time, entry wait)."""
        lane = self.lanes.get(tf)
        if lane is None:
            yield
            return
        saved = self.cfg, self.tf_ms
        self.cfg, self.tf_ms = lane, TF_MS[tf]
        try:
            yield
        finally:
            self.cfg, self.tf_ms = saved

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
        has = getattr(self.broker, "has_symbol", None)
        missing = [s for s in self.cfg.symbols if has and not has(s)]
        for s in missing:
            log.warning("no futures market for %s on Bitget, skipping it", s)
        self.cfg.symbols = [s for s in self.cfg.symbols if s not in missing]
        for lane in self.lanes.values():
            lane.symbols = self.cfg.symbols
        if not self.cfg.symbols:
            raise SystemExit("选的币 Bitget 都没有合约")
        for s in self.cfg.symbols:
            self.broker.setup_symbol(s)
        log.info("started: mode=%s symbols=%s leverage=%sx margin %s-%s USDT",
                 self.cfg.mode, self.cfg.symbols, self.cfg.leverage,
                 self.cfg.min_margin_usd, self.cfg.max_margin_usd)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        stuck = open(self.state_path.parent / "stuck.txt", "w", encoding="utf-8")
        while True:
            # a step that takes over 2 minutes writes where it is stuck to data/stuck.txt
            faulthandler.dump_traceback_later(120, repeat=True, file=stuck)
            t0 = time.time()
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("loop error, will retry")
            finally:
                faulthandler.cancel_dump_traceback_later()
            if time.time() - t0 > 60:
                log.warning("SLOW step took %.0f s", time.time() - t0)
            time.sleep(self.cfg.loop_seconds)

    def step(self) -> None:
        b, now = self.broker, self.broker.now_ms()
        if hasattr(b, "tick"):
            b.tick()
            for sym, pnl, why in b.closed_by_exchange:
                log.info("%s closed by %s pnl=%.2f", sym, why, pnl)
            b.closed_by_exchange.clear()

        first = not self.last_bar
        if first:
            log.info("CHECK account")
        equity, free = self._balance()
        if equity <= 0:
            return  # balance unreadable/empty: do nothing rather than trip the kill switch
        positions = b.positions()
        if first:
            log.info("CHECK equity %.2f positions %d pending %d", equity, len(positions),
                     len(self.pending))
        self._check_pending(now, positions)
        self._forget_closed(now, positions)

        if self.risk.daily_loss_hit(now, equity) or (self.risk.state.killed and self.risk.limits_on):
            for sym in list(self.pending):
                self._cancel_pending(sym, now, positions)
            if positions:
                log.warning("RISK STOP (equity %.2f, day start %.2f): closing everything",
                            equity, self.risk.state.day_start_equity)
                self._close_all(now, positions)
            self._save_state()
            return

        if self.flow_mode:
            self._flow_step(now, positions)
            self._save_state()
            return
        # slowest timeframe first: when two agree on the same coin the bigger picture opens it
        tfs = sorted({self.cfg.timeframe, *self.lanes}, key=lambda x: -TF_MS[x])
        scanned = {}
        if not self.last_bar:
            log.info("SCANNING %d coins on %s", len(self.cfg.symbols), " ".join(tfs))
        for sym in self.cfg.symbols:
            for tf in tfs:
                with self._as_lane(tf):
                    x = self._refresh_bars(sym, now, positions)
                if x:
                    scanned.setdefault(tf, []).append(x)
            if sym in positions:
                with self._as_lane(self.trades[sym].tf if sym in self.trades else ""):
                    self._manage(sym, positions[sym], now)
        if self.cfg.live_watch:
            for sym in self.cfg.symbols:
                if sym not in positions and sym not in self.trades and sym not in self.pending:
                    try:
                        self._watch(sym, now, positions)
                    except Exception as e:  # noqa: BLE001 - one coin's live read never stops the rest
                        log.warning("%s live watch not read (%s)", sym, e)
        for tf, scans in scanned.items():
            # one line per closed bar so the log shows the bot is looking, even with no trade
            trends = [t for t, _ in scans]
            log.info("SCAN %d coins up=%d down=%d flat=%d signals=%d%s", len(scans),
                     trends.count(1), trends.count(-1), trends.count(0),
                     sum(1 for _, sig in scans if sig), f" tf={tf}" if self.lanes else "")
        if scanned:
            fast = TF_MS[tfs[-1]]
            nxt = (now // fast + 1) * fast
            log.info("NEXT SCAN %s", datetime.fromtimestamp(nxt / 1000, timezone.utc).isoformat())
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
        # demo: every start counts from a fresh 100 (live 10/24: other runs on the same demo
        # account had moved it more than 100 below the old baseline, the virtual account read
        # below 0 and the bot silently stopped trading)
        rebase = self.cfg.mode != "live" and not getattr(self, "_rebased", False)
        if st.equity_baseline <= 0 or rebase:
            self._rebased = True
            st.equity_baseline = equity
            log.info("capital_usd=%.2f: real equity %.2f taken as baseline", cap, equity)
        virtual = cap + (equity - st.equity_baseline)
        if virtual <= 0 and not getattr(self, "_warned_zero", False):
            self._warned_zero = True
            log.warning("账户余额读到 0，机器人不会开单。请确认模拟盘合约账户里有 USDT。")
        return virtual, min(free, virtual - (equity - free))

    # ---------- maker entries ----------
    def _check_pending(self, now: int, positions: dict[str, Position]) -> None:
        for sym in list(self.pending):
            pe = self.pending[sym]
            status, filled, _ = self.broker.order_status(sym, pe.order_id)
            if status == "open" and now < pe.expires_ms:
                continue
            if status == "open":
                self._cancel_pending(sym, now, positions, fallback=True)
                continue
            self.pending.pop(sym)
            if filled > 0:
                self._on_filled(sym, pe, now)
            else:
                log.info("%s maker entry %s not filled (%s)", sym, pe.order_id, status)
                self.entry_wait[sym] = now + ENTRY_RETRY_MS
                self._market_fallback(sym, pe, now, positions)
        if self.trades.keys() - positions.keys():
            positions.clear()
            positions.update(self.broker.positions())

    def _cancel_pending(self, sym: str, now: int, positions: dict[str, Position],
                        fallback: bool = False) -> None:
        pe = self.pending.pop(sym)
        self.broker.cancel_order(sym, pe.order_id)
        status, filled, _ = self.broker.order_status(sym, pe.order_id)
        if filled > 0:
            log.info("%s maker entry partly filled %s/%s before cancel", sym, filled, pe.qty)
            self._on_filled(sym, pe, now)
        else:
            log.info("%s maker entry %s cancelled (timeout)", sym, pe.order_id)
            self.entry_wait[sym] = now + ENTRY_RETRY_MS
            if fallback:
                self._market_fallback(sym, pe, now, positions)

    def _market_fallback(self, sym: str, pe: PendingEntry, now: int,
                         positions: dict[str, Position]) -> None:
        """maker_fallback_market: the limit entry did not fill, take the trade at market
        anyway (never miss a signal) unless price already ran too far or through the stop."""
        if not self.cfg.maker_fallback_market:
            return
        price = self.broker.price(sym)
        sign = sign_of(pe.side)
        dist = abs(pe.price - pe.stop)
        if sign * (price - pe.price) > 0.5 * dist or sign * (price - pe.stop) <= 0:
            log.info("%s signal skipped: price already ran %.10g past signal", sym,
                     price - pe.price)
            return
        fill = self.broker.open(sym, pe.side, pe.qty, pe.stop, 0.0 if self.limit_tp else pe.take_profit)
        self.risk.record_open()
        tp = pe.take_profit
        if self.limit_tp and fill > 0:
            r = self.cfg.strategy.take_profit_r[-1]
            tp = self._round(sym, fill + sign * abs(fill - pe.stop) * r, pe.side)
        lev = self.broker.leverage_for(sym) if hasattr(self.broker, "leverage_for") else self.cfg.leverage
        log.info("OPEN %s %s qty=%s @ %.10g stop=%.10g tp=%.10g margin≈%.2f strength=%.2f (%s)",
                 sym, pe.side, pe.qty, fill, pe.stop, tp, fill * pe.qty / lev, 0.0,
                 "market after limit not filled")
        self.trades[sym] = Trade(pe.side, pe.qty, fill, pe.stop, pe.stop, tp, fill, now, **pe.plan)
        pos = self.broker.positions().get(sym)
        if pos:
            positions[sym] = pos
            self._liq_safety(sym, pos, pe.stop, now)
        if sym in self.trades:
            self._place_tp(sym, self.trades[sym])

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
            with self._as_lane(self.trades[sym].tf):
                self._forget_one(sym, now)

    def _forget_one(self, sym: str, now: int) -> None:
        if True:
            t = self.trades.pop(sym)
            self._cancel_tp(sym, t)  # stop hit: the limit take-profit must not stay behind
            now_px = self.broker.price(sym)
            sign = 1 if t.side == "long" else -1
            avg = t.avg_entry or t.entry
            # the exchange closed it at its stop or its take-profit: count whichever of the two
            # the price is nearest now, not the price a few seconds later (that made stops look
            # far worse than they were, live ADA -4.43 for a 0.3% stop)
            px = min((t.stop, t.take_profit), key=lambda x: abs(x - now_px)) \
                if t.take_profit > 0 else t.stop
            fee = self.cfg.fees.maker if px == t.take_profit and self.limit_tp else self.cfg.fees.taker
            if t.exit_px and abs(now_px - t.exit_px) < abs(now_px - t.stop):
                px, fee = t.exit_px, self.cfg.fees.maker  # our limit exit filled
            f = self.cfg.fees
            fees = (f.maker if self.cfg.entry_order == "maker" else f.taker) * avg * t.qty \
                + fee * px * t.qty
            pnl = sign * (px - avg) * t.qty - fees + t.realized
            limit_exit = bool(t.exit_px) and px == t.exit_px
            if limit_exit:
                log.info("%s exit limit filled @ %.10g, approx pnl %.2f", sym, px, pnl)
            else:
                log.info("%s position gone (exchange stop/TP), approx pnl %.2f", sym, pnl)
            self.risk.record_close(now, sym, pnl, pnl < 0, self.tf_ms,
                                   (f"kind:{t.kind}",) if t.kind else ())
            self._pause_if_losing(sym, now)
            self._journal(sym, t.side, avg, px, "limit exit after hold" if limit_exit
                          else "exchange stop/TP", pnl, t.opened_ms, now, t.reason)

    def _pause_if_losing(self, sym: str, now: int) -> None:
        """盘口模式: live 10/23 ADA lost 7 flow trades in a row (-15.64 of -19.80) while the
        bot kept trading it at half size; a coin whose last 3+ live results add up to a loss
        is left alone for PAUSE_MS."""
        if not self.flow_mode:
            return
        recent = self.risk.state.recent.get(sym, [])[-3:]
        if len(recent) >= 3 and sum(recent) < 0:
            self.coin_pause[sym] = now + PAUSE_MS
            log.info("PAUSE %s last %d trades %.2f, not traded for %d hours", sym, len(recent),
                     sum(recent), PAUSE_MS // 3_600_000)

    def _journal(self, sym: str, side: str, entry: float, exit_: float, why: str, pnl: float,
                 opened_ms: int, now: int, reason: str = "") -> None:
        """Every closed live trade, one row in data/trades.csv, for the daily review."""
        path = self.state_path.parent / "trades.csv"
        new = not path.exists()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["closed_utc", "coin", "side", "entry", "exit", "why", "pnl",
                                "minutes", "signal"])
                w.writerow([datetime.fromtimestamp(now / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            sym.split("/")[0], side, f"{entry:.10g}", f"{exit_:.10g}", why,
                            f"{pnl:.2f}", round((now - opened_ms) / 60_000), reason])
        except OSError as e:
            log.warning("trade journal not written: %s", e)
        today = self.risk.state
        log.info("TODAY %d trades, pnl %.2f", today.trades_today, today.realized_today)

    def _close_all(self, now: int, positions: dict[str, Position]) -> None:
        for sym, p in positions.items():
            self._close(sym, p, now, "risk stop")

    def _close(self, sym: str, p: Position, now: int, why: str) -> None:
        self._cancel_tp(sym, self.trades.get(sym))
        fill = self.broker.close(sym, p)
        sign = 1 if p.side == "long" else -1
        f = self.cfg.fees
        fees = ((f.maker if self.cfg.entry_order == "maker" else f.taker) * p.entry
                + f.taker * fill) * p.qty
        t = self.trades.pop(sym, None)
        pnl = sign * (fill - p.entry) * p.qty - fees + (t.realized if t else 0.0)
        log.info("CLOSE %s %s qty=%s @ %.10g (%s) pnl≈%.2f", sym, p.side, p.qty, fill, why, pnl)
        stopped = pnl < 0 and why not in ("max hold time", "replaced by new signal")
        self.risk.record_close(now, sym, pnl, stopped, self.tf_ms,
                               (f"kind:{t.kind}",) if t and t.kind else ())
        self._pause_if_losing(sym, now)
        self._journal(sym, p.side, p.entry, fill, why, pnl, t.opened_ms if t else now, now,
                      t.reason if t else "")

    def _refresh_bars(self, sym: str, now: int,
                      positions: dict[str, Position]) -> tuple[int, bool] | None:
        """On a newly closed bar: (trend -1/0/1, signal found); None if nothing new."""
        cur_bar = (now // self.tf_ms) * self.tf_ms - self.tf_ms  # last fully closed bar
        key = sym if self.cfg.timeframe not in self.lanes else f"{sym}@{self.cfg.timeframe}"
        if self.last_bar.get(key) == cur_bar:
            return None
        df = self.broker.ohlcv(sym, self.cfg.timeframe, HISTORY_BARS)
        trend_bars = self.cfg.strategy.trend_ema * 3 + 10
        dft = self.broker.ohlcv(sym, self.cfg.trend_timeframe, trend_bars)
        if df.empty or dft.empty:
            return None
        prepared = prepare(df, dft, self.cfg)
        row = prepared.iloc[-1]
        self.last_bar[key] = int(row["ts"])
        self.atr[key] = float(row["atr"])
        trend = int(row["trend"])
        if sym in positions or sym in self.pending:
            return trend, False
        sig = signal_at(row, self.cfg)
        if self.tf_ms >= TF_MS["15m"] and not sig:
            self._look(sym, row, trend, now)
        if sig:
            self._try_open(sym, sig, now, positions)
        return trend, sig is not None

    def _watch(self, sym: str, now: int, positions: dict[str, Position]) -> None:
        """实时看盘: price at the big structure's support/resistance (or a round number) right
        now, and live takers + the book pushing one way: bounce off the level or break it."""
        if now < self.__dict__.setdefault("_watch_until", {}).get(sym, 0):
            return
        st = self._structure(sym, now)
        if not st or not hasattr(self.broker, "flow"):
            return
        px = self.broker.price(sym)
        near = self.cfg.watch_near
        step = round_step(px)
        levels = [("sup", st["sup"]), ("res", st["res"]),
                  ("round", math.floor(px / step) * step), ("round", math.ceil(px / step) * step)]
        at = [(k, lv) for k, lv in levels if lv > 0 and abs(px / lv - 1) < near]
        if not at:
            return
        f = self.broker.flow(sym)
        taker = f["buy_qty"] + f["sell_qty"]
        if taker <= 0:
            return
        buy = f["buy_qty"] / taker
        book = f["bid_qty5"] + f["ask_qty5"]
        tilt = (f["bid_qty5"] - f["ask_qty5"]) / book if book else 0.0
        side = "long" if buy >= self.cfg.watch_min_share and tilt > -0.3 else \
            "short" if 1 - buy >= self.cfg.watch_min_share and tilt < 0.3 else None
        if side is None:
            return
        sign = 1 if side == "long" else -1
        name, lv = at[0]
        # above the level and buyers in charge: it held (bounce) or was just broken upward
        kind = "range" if sign * (px - lv) >= 0 and name in ("sup", "round") and side == "long" \
            or sign * (px - lv) >= 0 and name in ("res", "round") and side == "short" else "breakout"
        if sign * (px - lv) < 0:
            return  # still on the wrong side of the level: wait for it to hold or break
        r = self.cfg.risk
        dist = min(max(abs(px - lv) + px * 0.001, px * r.min_stop_pct), px * r.max_stop_pct)
        stop = px - sign * dist
        tp = px + sign * dist * self.cfg.strategy.take_profit_r[-1]
        what = {"sup": "支撑", "res": "阻力", "round": "整数关口"}[name]
        reason = (f"{side} 看盘 {what} {lv:.10g} 主动{'买' if sign > 0 else '卖'} "
                  f"{(buy if sign > 0 else 1 - buy):.0%} 挂单 {sign * tilt:+.2f}")
        log.info("WATCH %s %s at %s %.10g taker %.0f%% book %+.2f", sym, side, name, lv,
                 (buy if sign > 0 else 1 - buy) * 100, sign * tilt)
        self._watch_until[sym] = now + self.cfg.watch_cooldown_s * 1000
        sig = Signal(side, px, stop, tp, dist, 0.5, reason, kind=kind,
                     book={"bid": f["bid"], "ask": f["ask"]})
        self._try_open(sym, sig, now, positions)

    def _votes(self, sym: str, now: int) -> int | None:
        """Net votes of the 9 indicators on the last closed 15m bar, cached per bar."""
        if not self.cfg.votes_need:
            return None
        cache = self.__dict__.setdefault("_votes_cache", {})
        bar = now // TF_MS["15m"]
        if sym not in cache or cache[sym][0] != bar:
            try:
                c15 = self.lanes.get("15m") or lane_config(self.cfg, "15m")
                df = prepare(self.broker.ohlcv(sym, "15m", HISTORY_BARS),
                             self.broker.ohlcv(sym, "1h", c15.strategy.trend_ema * 3 + 10), c15)
                val = int(df["score"].iloc[-1])
            except Exception as e:  # noqa: BLE001 - never block trading on this
                log.warning("%s indicators not read (%s)", sym, e)
                val = None
            cache[sym] = (bar, val)
        return cache[sym][1]

    def _positioning(self, sym: str, now: int) -> tuple[float | None, float | None]:
        """(funding rate, open interest change over ~15 min), refreshed every 5 minutes."""
        if not (self.cfg.funding_max or self.cfg.oi_confirm):
            return None, None
        ex = getattr(self.broker, "ex", None)
        if ex is None:
            return None, None
        cache = self.__dict__.setdefault("_pos_cache", {})
        hist = self.__dict__.setdefault("_oi_hist", {}).setdefault(sym, [])
        slot = now // 300_000
        if sym not in cache or cache[sym][0] != slot:
            fund = oi = None
            try:
                fund = float(ex.fetch_funding_rate(sym).get("fundingRate") or 0) or None
            except Exception as e:  # noqa: BLE001
                log.warning("%s funding not read (%s)", sym, e)
            try:
                d = ex.fetch_open_interest(sym)
                oi = float(d.get("openInterestAmount") or d.get("openInterestValue") or 0) or None
            except Exception as e:  # noqa: BLE001
                log.warning("%s open interest not read (%s)", sym, e)
            chg = None
            if oi:
                hist.append((now, oi))
                del hist[:-6]
                old = next((v for t, v in hist if now - t >= 840_000), None)
                chg = oi / old - 1 if old else None
            cache[sym] = (slot, (fund, chg))
        return cache[sym][1]

    def _rsi(self, sym: str, now: int) -> float | None:
        """RSI(14) on 15m candles, cached per 15m bar; None if off or not read."""
        if not self.cfg.overbought:
            return None
        cache = self.__dict__.setdefault("_rsi_cache", {})
        bar = now // TF_MS["15m"]
        if sym not in cache or cache[sym][0] != bar:
            try:
                c = self.broker.ohlcv(sym, "15m", 100)["close"]
                val = float(rsi(c, 14).iloc[-1])
            except Exception as e:  # noqa: BLE001 - never block trading on this
                log.warning("%s RSI not read (%s)", sym, e)
                val = None
            cache[sym] = (bar, val)
        return cache[sym][1]

    def _long_short(self, sym: str, now: int) -> float | None:
        """Bitget long/short account ratio, cached 5 minutes; None if off or not available."""
        if not (self.cfg.crowd_long_max or self.cfg.crowd_short_min):
            return None
        ex = getattr(self.broker, "ex", None)
        if ex is None or not hasattr(ex, "fetch_long_short_ratio_history"):
            return None
        cache = self.__dict__.setdefault("_ls_cache", {})
        slot = now // 300_000
        if sym not in cache or cache[sym][0] != slot:
            try:
                hist = ex.fetch_long_short_ratio_history(sym, "5m", None, 1)
                v = hist[-1].get("longShortRatio") if hist else None
                val = float(v) if v not in (None, "") else None
            except Exception as e:  # noqa: BLE001
                log.warning("%s long/short ratio not read (%s)", sym, e)
                val = None
            cache[sym] = (slot, val)
        return cache[sym][1]

    def _look(self, sym: str, row, trend: int, now: int) -> None:
        """What the bot sees on a coin with no signal yet (15m and 1h scans): the trend, how
        far price still is from the breakout, volume, and the big-structure score."""
        px = float(row["close"])
        up, down = float(row["hh"]) / px - 1, px / float(row["ll"]) - 1
        vol = float(row["volume"]) / float(row["vol_avg"]) if row["vol_avg"] > 0 else 0.0
        st = self._structure(sym, now)
        r, ls = self._rsi(sym, now), self._long_short(sym, now)
        log.info("LOOK %s %s trend %+d up %.3f%% down %.3f%% vol %.1fx struct %s rsi %s ls %s", sym,
                 self.cfg.timeframe, trend, up * 100, down * 100, vol,
                 "-" if not st else f"{st['score']:+d}", "-" if r is None else f"{r:.0f}",
                 "-" if ls is None else f"{ls:.2f}")

    # ---------- 盘口模式 ----------
    def _flow_step(self, now: int, positions: dict[str, Position]) -> None:
        if self.feed is None:
            self.feed = FlowFeed(self.state_path.parent, self.cfg.symbols)
        feed, c = self.feed, self.cfg
        feed.update(now)
        feed.ensure_recorder(now, c.symbols)
        self._autotune(now)
        # real money only on a setting that made money in both parts of the test
        closed_for_new = c.mode == "live" and not self.flow_verified
        live = signals = 0
        for sym in c.symbols:
            if sym in positions:
                self._manage(sym, positions[sym], now)
                continue
            f = feed.latest(sym, c.flow_window)
            if not f or now - f["ts"] > FRESH_MS:
                continue
            live += 1
            if sym not in self.coin_pause:  # after a restart: judge the remembered results once
                self.coin_pause[sym] = 0
                self._pause_if_losing(sym, now)
            if (sym in self.pending or sym in self.trades or closed_for_new
                    or sym.split("/")[0] in c.flow_skip_coins
                    or now < self.entry_wait.get(sym, 0) or now < self.coin_pause.get(sym, 0)):
                continue
            side = flow_side(f, c.flow_threshold, c.flow_burst)
            if side is None:
                continue
            if c.flow_trend_minutes and feed.trend(sym, c.flow_trend_minutes, now) != sign_of(side):
                continue
            signals += 1
            sign = sign_of(side)
            mid = f["mid"]
            stop = mid * (1 - sign * c.flow_stop_pct)
            sig = Signal(side, mid, stop, 0.0, 0.0, 1.0,
                         f"flow {f['flow']:+.2f} burst {f['burst']:.1f}x book {f['tilt']:+.2f}",
                         kind="flow", book={"bid": f["bid"], "ask": f["ask"]})
            log.info("FLOWSIG %s %s taker %+.2f volume %.1fx book %+.2f", sym, side, f["flow"],
                     f["burst"], f["tilt"])
            self._try_open(sym, sig, now, positions)
        minute = now // 60_000
        if getattr(self, "_flow_minute", None) != minute:
            self._flow_minute = minute
            log.info("FLOWSCAN %d coins live %d signals %d", len(c.symbols), live, signals)

    def _autotune(self, now: int) -> None:
        """Every flow_autotune_hours re-pick entry + exit from the recording (in a background
        thread, the loop keeps trading meanwhile) and switch to the best."""
        c = self.cfg
        if c.flow_autotune_hours <= 0:
            return
        done = getattr(self, "_tune_done", None)
        if done is not None:
            self._tune_done = None
            choice, report = done
            path = self.state_path.parent / "autotune_report.txt"
            try:
                path.write_text(report, encoding="utf-8")
            except OSError:
                pass
            if choice is not None:
                c.flow_window, c.flow_threshold, c.flow_burst = (choice.window, choice.threshold,
                                                                 choice.burst)
                c.flow_exit, c.flow_stop_pct = choice.exit, choice.sl
                c.flow_be_margin = choice.margin
                c.flow_skip_coins = list(choice.skip)
                c.flow_trend_minutes = choice.trend
                if choice.exit == "bracket":
                    c.flow_tp_pct = choice.tp
                else:
                    c.flow_hold_seconds = choice.hold
                self.flow_verified = choice.verified
                log.info("AUTOTUNE %s | %s%s", "verified" if choice.verified else "unverified",
                         choice.describe(),
                         f"；先不做 {'、'.join(choice.skip)}" if choice.skip else "")
            else:
                log.info("AUTOTUNE none | %s", report.splitlines()[0])
        if now < getattr(self, "_tune_next", 0) or getattr(self, "_tuning", False):
            return
        self._tune_next = now + int(c.flow_autotune_hours * 3_600_000)
        self._tuning = True

        def work():
            from .autotune import choose
            from .flowtest import load
            try:
                self._tune_done = choose(load(self.state_path.parent))
            except Exception as e:  # noqa: BLE001 - keep trading on the current setting
                log.warning("auto-tune failed: %s", e)
            finally:
                self._tuning = False

        threading.Thread(target=work, daemon=True).start()

    def _flow_exit(self, sym: str, p: Position, t: Trade, now: int) -> None:
        """How a 盘口模式 trade gets out (flow_exit, picked by the auto-tune):
          time    : at flow_hold_seconds a limit at the best price on our side; market after
                    flow_exit_seconds more
          manual  : at flow_hold_seconds in profit after fees -> limit at the touch; otherwise
                    a limit at break-even, waiting for price to come back
          bracket : limit take-profit flow_tp_pct away from the start
        manual and bracket: after flow_max_hold_seconds a limit at the best price of the last
        30 minutes (re-aimed every 10 minutes), market only an hour after that.
        Exit limits are left resting, never chased tick by tick (that kept them unfilled).
        The exchange stop (flow_stop_pct) stays on the whole time."""
        c = self.cfg
        age = now - t.opened_ms
        mode = c.flow_exit
        if mode != "bracket" and age < c.flow_hold_seconds * 1000:
            return
        f = self.feed.latest(sym, c.flow_window) if self.feed else None
        if not f or now - f["ts"] > FRESH_MS:
            return
        sign = sign_of(t.side)
        cost = 2 * c.fees.maker
        touch = f["ask"] if sign > 0 else f["bid"]
        away = "short" if sign > 0 else "long"   # round exits away from the market
        if mode == "time":
            if age >= (c.flow_hold_seconds + c.flow_exit_seconds) * 1000:
                self._close(sym, p, now, "max hold time")
            elif not t.exit_kind:
                self._place_exit(sym, t, touch, "time", now)
            return
        maxh = c.flow_max_hold_seconds * 1000
        if age >= maxh + LATE_GIVEUP_MS:
            self._close(sym, p, now, "max hold time")
            return
        if age >= maxh:
            if t.exit_kind == "late" and now - t.exit_since < LATE_REFRESH_MS:
                return
            rows = [r for r in self.feed.rows.get(sym, []) if r["ts"] >= now - LATE_WINDOW_MS]
            mids = [(r["bid"] + r["ask"]) / 2 for r in rows] or [touch]
            best = max(mids) if sign > 0 else min(mids)
            self._place_exit(sym, t, max(best, touch) if sign > 0 else min(best, touch),
                             "late", now)
            return
        if mode == "bracket":
            if not t.exit_kind:
                self._place_exit(sym, t, self._round(sym, t.entry * (1 + sign * c.flow_tp_pct),
                                                     away), "take-profit", now)
            return
        if t.exit_kind == "profit" and now - t.exit_since >= c.flow_exit_seconds * 1000:
            self._cancel_tp(sym, t)  # price moved away from the profit exit: decide again
            t.exit_kind = ""
        if t.exit_kind:
            return
        if sign * (f["mid"] / t.entry - 1) >= cost:
            self._place_exit(sym, t, touch, "profit", now)
        else:
            want = cost + c.flow_be_margin
            self._place_exit(sym, t, self._round(sym, t.entry * (1 + sign * want), away),
                             "breakeven", now)

    def _place_exit(self, sym: str, t: Trade, px: float, kind: str, now: int) -> None:
        self._cancel_tp(sym, t)
        try:
            t.tp_order_id = self.broker.place_tp(sym, t.side, t.qty, px)
        except Exception as e:  # noqa: BLE001 - post-only refused as price moved: next loop
            log.info("%s exit limit not placed (%s), trying again", sym, e)
            t.exit_kind = ""
            return
        t.exit_px, t.exit_kind, t.exit_since = px, kind, now
        log.info("EXIT-LIMIT %s %s qty=%s @ %.10g (%s)", sym, t.side, t.qty, px, kind)

    def _leader_move(self, now: int) -> float | None:
        """BTC's change over the last leader_bars closed bars (cached per bar), None if off.

        Live 2026-10-23: shorts on every coin won while the whole market fell and were stopped
        together when it bounced; BTC leads those turns, so its last few minutes say which way
        the small coins are about to go."""
        n = self.cfg.leader_bars
        if n <= 0:
            return None
        bar = (self.cfg.timeframe, now // self.tf_ms)
        if getattr(self, "_leader", (None,))[0] != bar:
            try:
                df = self.broker.ohlcv(self.cfg.leader_symbol, self.cfg.timeframe, n + 5)
                c = df["close"].to_numpy()
                self._leader = (bar, float(c[-1] / c[-1 - n] - 1) if len(c) > n else None)
            except Exception as e:  # noqa: BLE001 - never block trading on this
                log.warning("leader check not read (%s)", e)
                self._leader = (bar, None)
        return self._leader[1]

    def _structure(self, sym: str, now: int) -> dict | None:
        """1h/4h/daily structure for sym, refreshed once per hour; None if off or not read."""
        if self.cfg.structure_min_score <= 0:
            return None
        cache = self.__dict__.setdefault("_struct", {})
        hour = now // 3_600_000
        if sym not in cache or cache[sym][0] != hour:
            try:
                snap = snapshot(self.broker.ohlcv(sym, "1h", 200), self.broker.ohlcv(sym, "4h", 200),
                                self.broker.ohlcv(sym, "1d", 60))
                log.info("STRUCT %s score %+d res %.10g sup %.10g", sym, snap["score"],
                         snap["res"], snap["sup"])
            except Exception as e:  # noqa: BLE001 - never block trading on this
                log.warning("%s structure not read (%s)", sym, e)
                snap = None
            cache[sym] = (hour, snap)
        return cache[sym][1]

    def _open_risk(self) -> float:
        """USDT lost if every open trade hit its stop now (trades already at break-even add 0)."""
        f = self.cfg.fees
        risk = 0.0
        for t in self.trades.values():
            sign = 1 if t.side == "long" else -1
            qty = t.qty + t.add_qty
            risk += max(0.0, sign * ((t.avg_entry or t.entry) - t.stop)) * qty \
                + qty * t.stop * (f.taker + f.slippage)
        return risk

    def _replaceable(self, sym: str, positions: dict[str, Position]) -> list[str]:
        """replace_on_new_signal: other coins' waiting limit entries and trades whose stop is
        not yet at break-even but which already show a profit after fees. A new signal on `sym`
        may take their place. A trade that is losing is never cut for a swap (live 2026-10-22:
        every such swap locked in a loss); it runs to its own stop or take-profit."""
        f = self.cfg.fees
        now = self.broker.now_ms()
        others = [s for s in self.pending if s != sym]
        for s in positions.keys() | self.trades.keys():
            t = self.trades.get(s)
            if s == sym or t is None or now - t.opened_ms < self.tf_ms:
                continue  # a trade opened on this bar gets at least one bar before a swap
            sign = 1 if t.side == "long" else -1
            p = positions.get(s)
            cost = p.qty * p.entry * (2 * f.taker + f.slippage) if p else 0.0
            if sign * (t.stop - t.entry) < 0 and p is not None and p.unrealized > cost:
                others.append(s)
        return others

    def _swap_out(self, others: list[str], sym: str, now: int,
                  positions: dict[str, Position]) -> None:
        fresh = None
        for s in others:
            if s in self.pending:
                self._cancel_pending(s, now, positions)
                continue
            if s not in positions:
                # opened earlier in this same loop, after `positions` was read
                fresh = fresh if fresh is not None else self.broker.positions()
                if s in fresh:
                    positions[s] = fresh[s]
            if s in positions:
                log.info("%s new signal on %s: closing %s to take it", s, sym, s)
                self._close(s, positions.pop(s), now, "replaced by new signal")
            else:
                self._cancel_tp(s, self.trades.pop(s, None))

    def _try_open(self, sym, sig, now, positions) -> None:
        equity, free = self._balance()
        # positions was read at the start of this loop: count trades opened since then too
        busy = len(positions.keys() | self.trades.keys()) + len(self.pending)
        swap = self._replaceable(sym, positions) \
            if self.cfg.replace_on_new_signal and busy >= self.cfg.max_open_positions else []
        ok, why = self.risk.can_open(now, sym, busy - len(swap), equity)
        if not ok:
            log.info("%s signal skipped: %s", sym, why)
            return
        move = self._leader_move(now)
        if move is not None and sign_of(sig.side) * move < -self.cfg.leader_min_move:
            log.info("%s signal skipped: BTC moving against it (%+.2f%% in %d bars)", sym,
                     move * 100, self.cfg.leader_bars)
            return
        sign = sign_of(sig.side)
        rsi_now = self._rsi(sym, now)
        ob = self.cfg.overbought
        if ob and rsi_now is not None and (rsi_now >= ob if sign > 0 else rsi_now <= 100 - ob):
            log.info("%s signal skipped: %s (RSI %.0f)", sym,
                     "overbought" if sign > 0 else "oversold", rsi_now)
            return
        ls = self._long_short(sym, now)
        if ls is not None and ((sign > 0 and self.cfg.crowd_long_max and ls >= self.cfg.crowd_long_max)
                               or (sign < 0 and self.cfg.crowd_short_min and ls <= self.cfg.crowd_short_min)):
            log.info("%s signal skipped: crowd already %s (long/short %.2f)", sym,
                     "long" if sign > 0 else "short", ls)
            return
        fund, oi_chg = self._positioning(sym, now)
        fm = self.cfg.funding_max
        if fm and fund is not None and sign * fund >= fm:
            log.info("%s signal skipped: funding %+.4f%% crowded %s", sym, fund * 100,
                     "long" if sign > 0 else "short")
            return
        if (self.cfg.oi_confirm and sig.kind != "range" and oi_chg is not None and oi_chg < 0):
            log.info("%s signal skipped: open interest falling %+.2f%% (no new money)", sym,
                     oi_chg * 100)
            return
        votes = self._votes(sym, now)
        if self.cfg.votes_need and votes is not None:
            need_v = -2 if sig.kind == "range" else self.cfg.votes_need
            if sign * votes < need_v:
                log.info("%s signal skipped: indicators %+d of 9 (need %+d)", sym, votes,
                         sign * need_v)
                return
        st = self._structure(sym, now)
        if st:
            if sign * st["score"] <= -self.cfg.structure_min_score:
                log.info("%s signal skipped: structure against it (score %+d)", sym, st["score"])
                return
            need = 0 if sig.kind == "range" else self.cfg.structure_need  # range edge: not against is enough
            if sign * st["score"] < need:
                log.info("%s signal skipped: structure not with it (score %+d, need %+d)", sym,
                         st["score"], sign * need)
                return
            near = self.cfg.structure_near_level
            if sign > 0 and st["res"] > sig.entry and st["res"] / sig.entry - 1 < near:
                log.info("%s signal skipped: long right under resistance %.10g", sym, st["res"])
                return
            if sign < 0 and st["sup"] < sig.entry and sig.entry / st["sup"] - 1 < near:
                log.info("%s signal skipped: short right on support %.10g", sym, st["sup"])
                return
            step = round_step(sig.entry)
            level = (sig.entry // step + (1 if sign > 0 else 0)) * step  # next round number ahead
            if level > 0 and sign * (level - sig.entry) / sig.entry < near and level != sig.entry:
                log.info("%s signal skipped: round number %.10g right ahead", sym, level)
                return
        flow = sig.book
        seen = {"kind": sig.kind, "btc": "" if move is None else f"{move * 100:+.3f}%",
                "atr": f"{sig.atr / sig.entry * 100:.3f}%",
                "hour": datetime.fromtimestamp(now / 1000, timezone.utc).hour}
        if st:
            seen["struct"] = f"{st['score']:+d}"
        if self.cfg.flow_confirm and hasattr(self.broker, "flow"):
            try:
                flow = self.broker.flow(sym)
                ok, share, imb = flow_agrees(sig.side, flow, self.cfg.flow_min_share)
                seen["flow"], seen["book"] = f"{share:.2f}", f"{imb:+.2f}"
            except Exception as e:  # noqa: BLE001 - no flow data must not block trading
                log.warning("%s order flow not read (%s), trading the signal anyway", sym, e)
            else:
                log.info("FLOW %s %s taker %.0f%% book %+.2f %s", sym, sig.side, share * 100,
                         imb, "ok" if ok else "against")
                if not ok:
                    return
        price = self.broker.price(sym)
        dist = abs(sig.entry - sig.stop)
        sign = 1 if sig.side == "long" else -1
        if sign * (price - sig.entry) > 0.5 * dist:
            log.info("%s signal skipped: price already ran %.10g past signal", sym, price - sig.entry)
            return
        maker = self.cfg.entry_order == "maker"
        if maker and flow and flow.get("bid") and flow.get("ask"):
            # rest at the front of our side of the book (sell at best ask, buy at best bid):
            # fills on the next taker instead of waiting for price to come back to the close
            price = flow["bid"] if sig.side == "long" else flow["ask"]
        elif maker:  # rest at the signal close, never on the wrong side of the market
            price = self._round(sym, min(price, sig.entry) if sig.side == "long"
                                else max(price, sig.entry), sig.side)
        # entry may sit a tick or two from the signal close (best bid/ask): keep the stop at
        # least the minimum distance from where we actually enter
        dist = max(dist, price * self.cfg.risk.min_stop_pct)
        plan = trade_plan(sig.side, price, dist, self.cfg)
        stop = self._round(sym, plan.stop, sig.side)
        tps = [self._round(sym, x, sig.side) for x in plan.take_profits]
        tp = tps[-1]
        lev = self.broker.leverage_for(sym) if hasattr(self.broker, "leverage_for") else None
        sizing, why = self.risk.size(price, stop, sig.side, sig.strength, free, equity, lev,
                                     self._open_risk(), sym, (f"kind:{sig.kind}",))
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
                 "tp_qtys": [self.broker.round_qty(sym, qty * f) for f in s.take_profit_split],
                 "reason": sig.reason + " | " + " ".join(f"{k}={v}" for k, v in seen.items()),
                 "kind": sig.kind, "tf": self.cfg.timeframe if self.lanes else ""}
        margin = sizing.margin * qty / sizing.qty
        if swap:  # only now that the new trade is certain to be placed
            self._swap_out(swap, sym, now, positions)

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
        if self.limit_tp and fill > 0:
            # the limit take-profit goes in after the fill: measure it from the real fill, so a
            # slipped market order keeps its full reward instead of a take-profit that is
            # now much closer than the stop (live XRP 2026-10-22: stop 0.46% away, TP 0.14%)
            tp = self._round(sym, fill + sign * abs(fill - stop) * s.take_profit_r[-1], sig.side)
        log.info("OPEN %s %s qty=%s @ %.10g stop=%.10g tp=%.10g margin≈%.2f strength=%.2f (%s)",
                 sym, sig.side, qty, fill, stop, tp, margin, sig.strength, sig.reason)
        self._log_plan(sym, extra, tps)
        self.trades[sym] = Trade(sig.side, qty, fill, stop, stop, tp, fill, now, **extra)
        pos = self.broker.positions().get(sym)
        if pos:
            positions[sym] = pos  # keep this loop's snapshot current for the next coins
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
        if self.flow_mode:
            self._flow_exit(sym, p, t, now)
            return
        price = self.broker.price(sym)
        if self.limit_tp and t.take_profit != t.entry and t.exit_kind != "late":
            sign = 1 if t.side == "long" else -1
            if not t.tp_order_id and sign * (price - t.take_profit) >= 0:
                self._close(sym, p, now, "take-profit")  # limit refused: take it at market
                return
            self._place_tp(sym, t)
        t.best = max(t.best, price) if t.side == "long" else min(t.best, price)
        bars = (now - t.opened_ms) // self.tf_ms
        atr = self.atr.get(f"{sym}@{t.tf}", self.atr.get(sym, 0.0)) if t.tf else self.atr.get(sym, 0.0)
        d = manage_position(t.side, t.entry, t.stop, t.initial_stop, price, t.best,
                            atr, bars, self.cfg)
        if d.new_stop is not None:
            t.stop = d.new_stop
        if d.close and d.reason == "max hold time" and self.cfg.strategy.late_exit:
            # 人工平法 (nick 2026-10-23): when the time is up do not dump the trade at market;
            # rest the exit at the best price of the last 30 minutes, market only an hour later
            hold_ms = self.cfg.strategy.max_hold_bars * self.tf_ms
            if now - t.opened_ms >= hold_ms + LATE_GIVEUP_MS:
                self._close(sym, p, now, "max hold time")
            else:
                self._late_exit(sym, t, price, now)
        elif d.close:
            self._close(sym, p, now, d.reason)
        else:
            self._scale(sym, t, price)

    def _late_exit(self, sym: str, t: Trade, price: float, now: int) -> None:
        if t.exit_kind == "late" and now - t.exit_since < LATE_REFRESH_MS:
            return
        try:
            bars = self.broker.ohlcv(sym, "1m", LATE_WINDOW_MS // 60_000)
            best = float(bars["high"].max() if t.side == "long" else bars["low"].min())
        except Exception as e:  # noqa: BLE001 - no 1m candles: aim at the current price
            log.info("%s 1m candles not read for the exit (%s)", sym, e)
            best = price
        px = max(best, price) if t.side == "long" else min(best, price)
        self._place_exit(sym, t, self._round(sym, px, "short" if t.side == "long" else "long"),
                         "late", now)

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


# entry timeframe -> its trend timeframe (快慢一起)
TREND_OF = {"1m": "15m", "3m": "15m", "5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h"}


def lane_config(cfg: Config, tf: str) -> Config:
    """cfg for entry timeframe tf: its trend candles, an entry that waits one bar, and the
    same hold time in hours as the main timeframe."""
    lane = copy.copy(cfg)
    lane.strategy = copy.copy(cfg.strategy)
    lane.timeframe, lane.trend_timeframe = tf, TREND_OF[tf]
    lane.maker_timeout_seconds = TF_MS[tf] // 1000
    hold_ms = cfg.strategy.max_hold_bars * TF_MS[cfg.timeframe]
    lane.strategy.max_hold_bars = max(1, hold_ms // TF_MS[tf])
    lane.risk = cfg.risk
    return lane
