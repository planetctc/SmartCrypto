"""Risk controls: position sizing, liquidation safety, daily loss limit, loss streak pause.

Every entry has to pass through RiskManager.approve(); nothing here talks to the exchange,
so the same rules run in backtest, paper, demo and live modes.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from .config import Config

log = logging.getLogger("bot.risk")
RECENT = 10       # live results remembered per coin
MIN_RECENT = 3    # results needed before a coin's record changes its size
MIN_MARGIN_USD = 1.0  # smallest trade margin the risk caps may shrink a trade to

MS_PER_DAY = 86_400_000
MS_PER_MIN = 60_000


def liquidation_price(entry: float, side: str, leverage: float, mmr: float) -> float:
    """Approximate isolated-margin liquidation price (ignores fees, so slightly optimistic;
    mmr in config is set conservatively to compensate)."""
    if side == "long":
        return entry * (1 - 1 / leverage + mmr)
    return entry * (1 + 1 / leverage - mmr)


def effective_leverage(cfg: Config, notional: float, equity: float, leverage: float) -> float:
    """Leverage that decides the liquidation price. Isolated: the set leverage. Cross (全仓):
    the whole account backs the position, so it is position size / account equity."""
    if cfg.margin_mode == "cross" and equity > 0:
        return max(1.0, notional / equity)
    return leverage


@dataclass
class Sizing:
    margin: float      # USDT margin for this position
    notional: float    # position value in USDT
    qty: float         # contracts in base coin (e.g. BTC)
    stop_pct: float
    expected_loss: float  # loss if stop is hit, fees and slippage included
    liq_price: float


@dataclass
class RiskState:
    day_key: int = -1
    day_start_equity: float = 0.0
    halted_day: int = -1           # day_key on which the daily loss limit was hit
    consecutive_losses: int = 0
    paused_until_ms: int = 0
    killed: bool = False           # equity fell below min_equity_stop_usd
    cooldown_until_ms: dict[str, int] = field(default_factory=dict)
    trades_today: int = 0
    opens_today: int = 0
    realized_today: float = 0.0
    equity_baseline: float = 0.0   # real account equity when the bot first started (capital_usd mode)
    recent: dict[str, list[float]] = field(default_factory=dict)  # coin -> last RECENT live pnls

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskState":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


class RiskManager:
    def __init__(self, cfg: Config, state: RiskState | None = None):
        self.cfg = cfg
        self.r = cfg.risk
        self.state = state or RiskState()
        # daily loss stop / min equity stop in force? (live_only_limits turns them off off-live)
        self.limits_on = not (self.r.live_only_limits and cfg.mode != "live")

    # ---------- day bookkeeping ----------
    def day_key(self, now_ms: int) -> int:
        return (now_ms + self.r.day_reset_utc_offset * 3_600_000) // MS_PER_DAY

    def update_equity(self, now_ms: int, equity: float) -> None:
        """Call every loop with total equity (balance + unrealized PnL)."""
        dk = self.day_key(now_ms)
        if dk != self.state.day_key:
            self.state.day_key = dk
            self.state.day_start_equity = equity
            self.state.trades_today = 0
            self.state.opens_today = 0
            self.state.realized_today = 0.0
        if self.limits_on and self.r.min_equity_stop_usd > 0 and equity < self.r.min_equity_stop_usd:
            self.state.killed = True

    def daily_loss_hit(self, now_ms: int, equity: float) -> bool:
        """True when today's drawdown reached the limit. Caller must flatten all positions."""
        self.update_equity(now_ms, equity)
        if not self.limits_on:
            return False
        if self.state.halted_day == self.state.day_key:
            return True
        floor = self.state.day_start_equity * (1 - self.r.daily_max_loss_pct)
        if self.state.day_start_equity > 0 and equity <= floor:
            self.state.halted_day = self.state.day_key
            return True
        return False

    def daily_loss_budget_left(self, equity: float) -> float:
        if not self.limits_on:
            return float("inf")
        floor = self.state.day_start_equity * (1 - self.r.daily_max_loss_pct)
        return max(0.0, equity - floor)

    # ---------- trade results ----------
    def record_open(self) -> None:
        self.state.opens_today += 1

    def record_close(self, now_ms: int, symbol: str, pnl: float, stopped_out: bool,
                     bar_ms: int) -> None:
        self.state.realized_today += pnl
        self.state.trades_today += 1
        self.state.recent[symbol] = (self.state.recent.get(symbol, []) + [round(pnl, 4)])[-RECENT:]
        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.r.max_consecutive_losses and \
                    self.state.consecutive_losses >= self.r.max_consecutive_losses:  # 0 = never pause
                self.state.paused_until_ms = now_ms + self.r.pause_after_losses_minutes * MS_PER_MIN
                self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses = 0
        if stopped_out:
            self.state.cooldown_until_ms[symbol] = now_ms + self.r.cooldown_bars_after_stop * bar_ms

    # ---------- entry approval ----------
    def can_open(self, now_ms: int, symbol: str, open_positions: int, equity: float) -> tuple[bool, str]:
        if self.state.killed and self.limits_on:
            return False, f"equity below {self.r.min_equity_stop_usd} USDT, bot stopped"
        if self.daily_loss_hit(now_ms, equity):
            return False, "daily loss limit reached, waiting for next day"
        if now_ms < self.state.paused_until_ms:
            return False, "paused after consecutive losses"
        if now_ms < self.state.cooldown_until_ms.get(symbol, 0):
            return False, "symbol cooling down after stop-loss"
        if self.r.max_trades_per_day and self.state.opens_today >= self.r.max_trades_per_day:
            return False, f"max {self.r.max_trades_per_day} trades per day reached"
        if open_positions >= self.cfg.max_open_positions:
            return False, "max open positions reached"
        return True, "ok"

    def size(self, entry: float, stop: float, side: str, strength: float,
             free_balance: float, equity: float,
             leverage: int | None = None, open_risk: float = 0.0,
             symbol: str = "") -> tuple[Sizing | None, str]:
        """Turn a signal into a position size, or reject it with a reason. open_risk is what the
        trades already open would lose together if all their stops hit; it comes off today's
        loss budget so many coins open at once can never add up to more than the daily limit."""
        cfg, r = self.cfg, self.r
        lev = leverage or cfg.leverage
        stop_pct = abs(entry - stop) / entry
        if stop_pct < r.min_stop_pct * 0.999:
            return None, f"stop {stop_pct:.3%} closer than min {r.min_stop_pct:.3%}"
        if stop_pct > r.max_stop_pct * 1.001:
            return None, f"stop {stop_pct:.3%} wider than max {r.max_stop_pct:.3%}"

        strength = min(1.0, max(0.0, strength))
        margin = cfg.fixed_margin_usd or \
            cfg.min_margin_usd + (cfg.max_margin_usd - cfg.min_margin_usd) * strength
        # learn from live results: a coin that has been losing lately trades at half size
        # (never below the minimum, never skipped), so money flows to the coins that work
        recent = self.state.recent.get(symbol, [])
        if len(recent) >= MIN_RECENT and sum(recent) < 0:
            margin = max(cfg.min_margin_usd, margin / 2)
            log.info("%s last %d trades %.2f: half size", symbol, len(recent), sum(recent))
        f = cfg.fees
        entry_cost = f.maker if cfg.entry_order == "maker" else f.taker + f.slippage
        cost_pct = stop_pct + entry_cost + f.taker + f.slippage  # exits are market orders

        # cap loss per trade and keep inside what is left of today's loss budget
        loss_cap = min(r.max_loss_per_trade_usd, self.daily_loss_budget_left(equity) - open_risk)
        if loss_cap <= 0:
            return None, "open trades already use today's loss budget"
        if margin * lev * cost_pct > loss_cap:
            margin = loss_cap / (lev * cost_pct)
        by_free = free_balance * 0.95 < margin
        margin = min(margin, free_balance * 0.95)
        # risk caps may shrink a trade below the panel's margin: still trade it, smaller,
        # down to MIN_MARGIN_USD (more small trades beat a bot that sits idle)
        if margin < MIN_MARGIN_USD:
            if by_free:
                return None, f"free balance {free_balance:.2f} too small"
            return None, (f"loss budget left {self.daily_loss_budget_left(equity):.2f}, "
                          f"open trades risk {open_risk:.2f}")

        notional = margin * lev
        liq = liquidation_price(entry, side, effective_leverage(cfg, notional, equity, lev),
                                r.maintenance_margin_rate)
        liq_dist = abs(entry - liq) / entry
        if stop_pct + r.liq_buffer_pct > liq_dist:
            return None, (f"stop {stop_pct:.3%} too close to liquidation {liq_dist:.3%} "
                          f"at {lev}x")
        return Sizing(
            margin=margin,
            notional=notional,
            qty=notional / entry,
            stop_pct=stop_pct,
            expected_loss=notional * cost_pct,
            liq_price=liq,
        ), "ok"
