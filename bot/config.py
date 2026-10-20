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


@dataclass
class StrategyConfig:
    trend_ema: int = 50
    breakout_lookback: int = 20
    volume_lookback: int = 20
    volume_factor: float = 1.2
    atr_period: int = 14
    atr_stop_mult: float = 1.0
    reward_risk: float = 2.0
    breakeven_at_r: float = 1.0
    trail_atr_mult: float = 1.5
    max_hold_bars: int = 48


@dataclass
class FeeConfig:
    taker: float = 0.0006
    maker: float = 0.0002
    slippage: float = 0.0002


@dataclass
class Config:
    mode: str = "demo"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT:USDT", "ETH/USDT:USDT"])
    timeframe: str = "5m"
    trend_timeframe: str = "1h"
    leverage: int = 100
    margin_mode: str = "isolated"
    min_margin_usd: float = 5.0
    max_margin_usd: float = 15.0
    max_open_positions: int = 2
    entry_order: str = "market"      # "market" or "maker" (post-only limit, lower fee)
    maker_timeout_seconds: int = 60  # cancel an unfilled maker entry after this long
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
    if cfg.entry_order not in ("market", "maker"):
        raise ValueError("entry_order must be market or maker")
    if cfg.margin_mode != "isolated":
        raise ValueError("only isolated margin is supported (limits each trade's loss to its margin)")
    if not 0 < cfg.min_margin_usd <= cfg.max_margin_usd:
        raise ValueError("need 0 < min_margin_usd <= max_margin_usd")
    if not 1 <= cfg.leverage <= 125:
        raise ValueError("leverage must be between 1 and 125")
    r = cfg.risk
    if not 0 < r.daily_max_loss_pct < 1:
        raise ValueError("risk.daily_max_loss_pct must be between 0 and 1")
    if not 0 < r.min_stop_pct < r.max_stop_pct:
        raise ValueError("need 0 < risk.min_stop_pct < risk.max_stop_pct")
