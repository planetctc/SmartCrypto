"""赛马: many ways of trading at once, on live prices, to learn in a day what one bot learns in
a week (nick 2026-10-23: testing must be fast, real trading can be slow).

Every variant is a full bot (same engine as the real one) on its own simulated 100 USDT
account (PaperBroker: maker entries fill only when price trades through them, stops pay taker
fee + slippage). They share one market-data connection, so 8 bots cost about what one does.
Results: data/race/<variant>/trades.csv, summary every 5 minutes in data/race_report.txt.
"""
from __future__ import annotations

import copy
import csv
import logging
import threading
import time
from pathlib import Path

import ccxt

from .config import Config, load_config
from .engine import Engine
from .exchange import PaperBroker, candles_to_df, retry
from .strategy import TF_MS

log = logging.getLogger("bot.race")

# (name, base config, overrides)  - overrides: "a.b" for nested fields
H1 = {"timeframe": "1h", "trend_timeframe": "4h", "maker_timeout_seconds": 3600,
      "strategy.max_hold_bars": 4}
M15 = {"timeframe": "15m", "trend_timeframe": "1h", "maker_timeout_seconds": 900,
       "strategy.max_hold_bars": 16}
M5 = {"timeframe": "5m", "trend_timeframe": "1h", "maker_timeout_seconds": 300,
      "strategy.max_hold_bars": 48}
# 只看 K 线, kept only as the yardstick that shows what 全面看 adds
KLINE_ONLY = {"structure_min_score": 0, "structure_need": 0, "leader_bars": 0,
              "flow_confirm": False}

# every racer looks at everything (nick 10/23: 看盘要全面, 没有的等于非): 1h/4h/daily structure
# leaning with the trade, no support/resistance or round number right ahead, BTC not against
# it, live order book with it. They differ only in entry timeframe and how fast they take profit.
VARIANTS = [
    ("1h全面看", "config.yaml", dict(H1)),
    ("1h全面看·止盈1R", "config.yaml", {**H1, "strategy.take_profit_r": [1.0]}),
    ("15分全面看", "config.yaml", dict(M15)),
    ("15分全面看·止盈1R", "config.yaml", {**M15, "strategy.take_profit_r": [1.0]}),
    ("5分全面看", "config.yaml", dict(M5)),
    ("5分全面看·止盈1R", "config.yaml", {**M5, "strategy.take_profit_r": [1.0]}),
    ("1分全面看", "config_hf.yaml", {"strategy.late_exit": True}),
    ("对照：15分只看K线", "config.yaml", {**M15, **KLINE_ONLY}),
]


class SharedMarket:
    """One connection for every racer: prices cached 2 s, candles until their next bar."""

    def __init__(self):
        self.ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self.ex.load_markets()
        self.lock = threading.Lock()
        self.prices: dict[str, tuple[float, float]] = {}
        self.bars: dict[tuple[str, str], tuple[int, int, object]] = {}

    def price(self, sym: str) -> float:
        with self.lock:
            hit = self.prices.get(sym)
            if hit and time.time() - hit[0] < 2:
                return hit[1]
            px = float(retry(self.ex.fetch_ticker, sym)["last"])
            self.prices[sym] = (time.time(), px)
            return px

    def ohlcv(self, sym: str, tf: str, limit: int):
        now = self.ex.milliseconds()
        bar = now // TF_MS[tf]
        with self.lock:
            hit = self.bars.get((sym, tf))
            if hit and hit[0] == bar and hit[1] >= limit:
                return hit[2].tail(limit).reset_index(drop=True)
            n = max(limit, hit[1] if hit else 0)
            df = candles_to_df(retry(self.ex.fetch_ohlcv, sym, tf, None, n), tf, now)
            self.bars[(sym, tf)] = (bar, n, df)
            return df.tail(limit).reset_index(drop=True)


class RaceBroker(PaperBroker):
    def __init__(self, cfg: Config, market: SharedMarket, start_balance: float = 100.0):
        self.market = market
        self.cfg = cfg
        self.ex = market.ex
        self.cash = start_balance
        self.pos, self.orders, self.closed_by_exchange = {}, {}, []
        self.limits, self.limit_tps, self._next_id = {}, {}, 1

    def price(self, symbol):
        return self.market.price(symbol)

    def ohlcv(self, symbol, tf, limit):
        return self.market.ohlcv(symbol, tf, limit)

    def flow(self, symbol):
        from .orderflow import live_flow
        with self.market.lock:
            return retry(live_flow, self.ex, symbol)


def variant_config(base: str, overrides: dict, name: str, root: Path, symbols, margin) -> Config:
    cfg = copy.deepcopy(load_config(root / base))
    for key, val in overrides.items():
        obj = cfg
        *path, last = key.split(".")
        for p in path:
            obj = getattr(obj, p)
        setattr(obj, last, val)
    cfg.mode = "paper"
    cfg.state_file = str(root / "data" / "race" / name / "state.json")
    if symbols:
        cfg.symbols = list(symbols)
    if margin:
        cfg.fixed_margin_usd = cfg.max_margin_usd = margin
        cfg.min_margin_usd = min(cfg.min_margin_usd, margin)
    return cfg


def summary(root: Path) -> str:
    """Ranking of every variant from its trades.csv."""
    rows = []
    for name, *_ in VARIANTS:
        path = root / "data" / "race" / name / "trades.csv"
        pnls, why = [], {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    p = float(r["pnl"])
                    pnls.append(p)
                    w = why.setdefault(r["why"], [0, 0.0])
                    w[0] += 1
                    w[1] += p
        wins = sum(1 for p in pnls if p > 0)
        rows.append((sum(pnls), name, len(pnls), wins, why))
    rows.sort(key=lambda r: -r[0])
    started = root / "data" / "race" / "started.txt"
    hours = (time.time() - float(started.read_text())) / 3600 if started.exists() else 0
    out = [f"赛马 {hours:.1f} 小时，{len(VARIANTS)} 套做法同时跑（各自 100 USDT 模拟账户，真实行情）", ""]
    for total, name, n, wins, why in rows:
        rate = f"{wins / n:.0%}" if n else "-"
        per = f"{total / n:+.2f}" if n else "-"
        out.append(f"{name}：{n} 单，胜率 {rate}，合计 {total:+.2f}，每单 {per}")
        if why:
            out.append("    " + "，".join(f"{REASONS.get(k, k)} {c} 单 {s:+.2f}"
                                          for k, (c, s) in sorted(why.items(), key=lambda x: x[1][1])))
    out += ["", "单数够多（每套 30 单以上）、合计是正的那套，就是要用在模拟盘上的做法。"]
    return "\n".join(out)


REASONS = {"exchange stop/TP": "止损/止盈", "max hold time": "到时间平", "managed stop hit": "跟踪止损",
           "limit exit after hold": "挂单平仓", "take-profit": "止盈", "risk stop": "风控平仓"}


def run(root: Path, symbols=None, margin=None) -> None:
    market = SharedMarket()
    race_dir = root / "data" / "race"
    race_dir.mkdir(parents=True, exist_ok=True)
    started, names = race_dir / "started.txt", race_dir / "variants.txt"
    now_names = "\n".join(n for n, *_ in VARIANTS)
    if not started.exists() or not names.exists() or names.read_text(encoding="utf-8") != now_names:
        started.write_text(str(time.time()))  # new set of racers: count hours from now
        names.write_text(now_names, encoding="utf-8")
    for name, base, overrides in VARIANTS:
        cfg = variant_config(base, overrides, name, root, symbols, margin)
        eng = Engine(cfg, RaceBroker(cfg, market))
        threading.Thread(target=eng.run, name=name, daemon=True).start()
        time.sleep(2)  # spread the first candle downloads
    report = root / "data" / "race_report.txt"
    while True:
        try:
            report.write_text(summary(root), encoding="utf-8")
        except OSError as e:
            log.warning("race report not written: %s", e)
        time.sleep(300)
