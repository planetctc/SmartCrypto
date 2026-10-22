"""Load config.yaml into typed settings."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class RiskConfig:
    daily_max_loss_pct: float = 0.30
    day_reset_utc_offset: int = 8
    max_loss_per_trade_usd: float = 10.0
    min_equity_stop_usd: float = 50.0
    max_stop_pct: float = 0.0040
    min_stop_pct: float = 0.0015
    liq_buffer_pct: float = 0.0008
    maintenance_margin_rate: float = 0.005
    max_consecutive_losses: int = 4
    pause_after_losses_minutes: int = 120
    cooldown_bars_after_stop: int = 6
    max_trades_per_day: int = 0      # 0 = no limit


# the indicators the score signal adds up (see strategy.votes)
VOTES = {"trend": "大趋势", "ema": "均线排列", "macd": "MACD", "kdj": "KDJ", "rsi": "RSI",
         "boll": "布林带", "volume": "放量", "candle": "K线形态", "breakout": "突破"}


@dataclass
class StrategyConfig:
    # breakout = 突破前高/前低; pullback = 顺势回踩均线; reversion = 超买超卖反转（布林带 + RSI）
    # score = 综合打分：趋势、均线、MACD、KDJ、RSI、布林带、成交量、K 线形态、突破 9 项一起投票
    signal_type: str = "breakout"
    min_score: int = 5               # score: enter when this many of the 9 votes agree (net)
    trend_ema: int = 50
    fast_ema: int = 20               # pullback: the average price pulls back to
    bb_period: int = 20              # reversion: Bollinger band length and width
    bb_std: float = 2.5
    rsi_period: int = 14
    rsi_low: float = 25              # reversion: long only below this RSI, short only above 100-x
    breakout_lookback: int = 20
    volume_lookback: int = 20
    volume_factor: float = 1.2
    trend_min_gap: float = 0.0       # only trade when the trend close is this far from its EMA
    atr_period: int = 14
    atr_stop_mult: float = 1.0
    breakeven_at_r: float = 1.0
    trail_atr_mult: float = 1.5
    max_hold_bars: int = 48
    # 分批进场: first batch now, 补仓 (second batch) if price first goes scale_in_at_r against us
    scale_in_frac: float = 0.5                 # first batch share of the planned size (1 = all at once)
    scale_in_at_r: float = 0.5                 # 0 = no 补仓: trade the first batch only
    # 分批止盈: take-profit levels in multiples of the stop distance; the last one sits on the
    # exchange, the earlier ones each close take_profit_split of the first batch
    take_profit_r: list[float] = field(default_factory=lambda: [1.2, 2.0, 3.0])
    take_profit_split: list[float] = field(default_factory=lambda: [0.34, 0.33])
    avoid_round_numbers: bool = True           # stop beyond / take-profit before round levels


@dataclass
class FeeConfig:
    taker: float = 0.0006
    maker: float = 0.0002
    slippage: float = 0.0002


@dataclass
class Config:
    mode: str = "demo"
    account_type: str = "auto"       # Bitget account: auto, classic (合约账户) or unified (统一账户)
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT:USDT", "ETH/USDT:USDT"])
    timeframe: str = "5m"
    trend_timeframe: str = "1h"
    capital_usd: float = 0.0          # >0: run risk rules as if the account only held this much
    leverage: int = 100
    margin_mode: str = "isolated"    # "isolated" (逐仓) or "cross" (全仓)
    min_margin_usd: float = 5.0
    max_margin_usd: float = 15.0
    fixed_margin_usd: float = 0.0    # >0: every trade uses this margin (risk caps may shrink it)
    max_open_positions: int = 2
    entry_order: str = "market"      # "market" or "maker" (post-only limit, lower fee)
    maker_timeout_seconds: int = 60  # cancel an unfilled maker entry after this long
    # take-profit as a resting reduce-only limit order (maker fee, no slippage) instead of the
    # exchange's trigger-market take-profit
    tp_order: str = "market"
    whole_price_coins: list[str] = field(default_factory=lambda: ["BTC", "ETH"])  # no decimals
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    loop_seconds: int = 10
    state_file: str = "data/state.json"
    log_file: str = "data/bot.log"


def _build(cls, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"config: unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw["risk"] = _build(RiskConfig, raw.get("risk", {}))
    raw["strategy"] = _build(StrategyConfig, raw.get("strategy", {}))
    raw["fees"] = _build(FeeConfig, raw.get("fees", {}))
    cfg = _build(Config, raw)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.mode not in ("paper", "demo", "live"):
        raise ValueError("mode must be paper, demo or live")
    if cfg.account_type not in ("auto", "classic", "unified"):
        raise ValueError("account_type must be auto, classic or unified")
    if cfg.entry_order not in ("market", "maker"):
        raise ValueError("entry_order must be market or maker")
    if cfg.tp_order not in ("market", "maker"):
        raise ValueError("tp_order must be market or maker")
    st = cfg.strategy
    if cfg.tp_order == "maker" and ((st.scale_in_frac < 1 and st.scale_in_at_r > 0) or st.take_profit_split):
        raise ValueError("tp_order maker works with one entry and one take-profit only")
    if cfg.margin_mode not in ("isolated", "cross"):
        raise ValueError("margin_mode must be isolated or cross")
    if not 0 < cfg.min_margin_usd <= cfg.max_margin_usd:
        raise ValueError("need 0 < min_margin_usd <= max_margin_usd")
    if not 1 <= cfg.leverage <= 125:
        raise ValueError("leverage must be between 1 and 125")
    r = cfg.risk
    if not 0 < r.daily_max_loss_pct < 1:
        raise ValueError("risk.daily_max_loss_pct must be between 0 and 1")
    if not 0 < r.min_stop_pct < r.max_stop_pct:
        raise ValueError("need 0 < risk.min_stop_pct < risk.max_stop_pct")
    s = cfg.strategy
    if s.signal_type not in ("breakout", "pullback", "reversion", "score"):
        raise ValueError("strategy.signal_type must be breakout, pullback, reversion or score")
    if not 1 <= s.min_score <= len(VOTES):
        raise ValueError(f"strategy.min_score must be 1..{len(VOTES)}")
    if not 0 < s.scale_in_frac <= 1 or s.scale_in_at_r < 0:
        raise ValueError("need 0 < strategy.scale_in_frac <= 1 and scale_in_at_r >= 0")
    if not s.take_profit_r or s.take_profit_r != sorted(s.take_profit_r) or s.take_profit_r[0] <= 0:
        raise ValueError("strategy.take_profit_r must be rising positive numbers")
    if len(s.take_profit_split) != len(s.take_profit_r) - 1 or sum(s.take_profit_split) >= 1:
        raise ValueError("strategy.take_profit_split needs one share per earlier level, sum < 1")
