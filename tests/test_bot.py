import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import resample, run_backtest, synthetic  # noqa: E402
from bot.config import Config, load_config  # noqa: E402
from bot.engine import Engine  # noqa: E402
from bot.exchange import Position  # noqa: E402
from bot.risk import RiskManager, liquidation_price  # noqa: E402
from bot.strategy import manage_position, prepare, signal_at  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAY = 86_400_000


def cfg() -> Config:
    return load_config(ROOT / "config.yaml")


def test_config_loads_user_settings():
    c = cfg()
    assert c.leverage == 100 and c.min_margin_usd == 5 and c.max_margin_usd == 15
    assert c.risk.daily_max_loss_pct == 0.30


def test_liquidation_about_half_percent_at_100x():
    liq = liquidation_price(100.0, "long", 100, 0.005)
    assert liq == pytest.approx(99.5)
    assert liquidation_price(100.0, "short", 100, 0.005) == pytest.approx(100.5)


def test_sizing_margin_range_and_stop_inside_liquidation():
    rm = RiskManager(cfg())
    rm.update_equity(0, 100)
    weak, _ = rm.size(100.0, 99.7, "long", 0.0, free_balance=100, equity=100)
    strong, _ = rm.size(100.0, 99.7, "long", 1.0, free_balance=100, equity=100)
    assert weak.margin == pytest.approx(5)
    assert 5 < strong.margin <= 15
    assert strong.expected_loss <= 10 + 1e-9          # max_loss_per_trade_usd
    assert 99.7 > strong.liq_price                     # stop fires before liquidation


def test_sizing_rejects_wide_and_tight_stops():
    rm = RiskManager(cfg())
    rm.update_equity(0, 100)
    assert rm.size(100.0, 99.0, "long", 1, 100, 100)[0] is None    # 1% stop: past liquidation
    assert rm.size(100.0, 99.95, "long", 1, 100, 100)[0] is None   # 0.05%: noise
    assert rm.size(100.0, 100.3, "short", 1, 100, 100)[0] is not None


def test_sizing_respects_remaining_daily_budget():
    rm = RiskManager(cfg())
    rm.update_equity(0, 100)
    s, why = rm.size(100.0, 99.6, "long", 1, 100, equity=72)  # only 2 USDT left before -30%
    assert s is None and "below min" in why


def test_daily_loss_limit_and_reset_next_day():
    rm = RiskManager(cfg())
    t0 = 1_700_000_000_000
    rm.update_equity(t0, 100)
    assert not rm.daily_loss_hit(t0 + 1000, 71)
    assert rm.daily_loss_hit(t0 + 2000, 70)
    assert rm.daily_loss_hit(t0 + 3000, 90)            # stays halted for the rest of the day
    assert not rm.can_open(t0 + 4000, "BTC/USDT:USDT", 0, 90)[0]
    assert not rm.daily_loss_hit(t0 + DAY, 70)         # new day: 70 is the new start
    assert rm.can_open(t0 + DAY, "BTC/USDT:USDT", 0, 70)[0]


def test_day_resets_at_utc_plus_8_midnight():
    rm = RiskManager(cfg())
    midnight_utc8 = 1_700_006_400_000 - 8 * 3_600_000  # 2023-11-15 00:00 UTC+8
    assert rm.day_key(midnight_utc8 - 1) + 1 == rm.day_key(midnight_utc8)


def test_pause_after_consecutive_losses_and_cooldown():
    c = cfg()
    rm = RiskManager(c)
    rm.update_equity(0, 100)
    for i in range(c.risk.max_consecutive_losses):
        rm.record_close(1000, "ETH/USDT:USDT", -1, True, 300_000)
    ok, why = rm.can_open(2000, "BTC/USDT:USDT", 0, 96)
    assert not ok and "consecutive" in why
    later = 1000 + c.risk.pause_after_losses_minutes * 60_000 + 1
    assert rm.can_open(later, "BTC/USDT:USDT", 0, 96)[0]


def test_min_equity_kill_switch():
    rm = RiskManager(cfg())
    rm.update_equity(0, 49)
    assert not rm.can_open(1, "BTC/USDT:USDT", 0, 49)[0]


def test_no_lookahead_in_signals():
    c = cfg()
    df = synthetic(3000, seed=3)
    full = prepare(df, resample(df, c.trend_timeframe), c)
    cut = df.iloc[:2000]
    part = prepare(cut, resample(cut, c.trend_timeframe), c)
    cols = ["atr", "hh", "ll", "vol_avg", "trend"]
    pd.testing.assert_frame_equal(full.iloc[:1990][cols].reset_index(drop=True),
                                  part.iloc[:1990][cols].reset_index(drop=True))


def test_signal_has_stop_on_correct_side():
    c = cfg()
    df = synthetic(6000, seed=1)
    p = prepare(df, resample(df, c.trend_timeframe), c)
    sigs = [s for s in (signal_at(r, c) for _, r in p.iterrows()) if s]
    assert sigs
    for s in sigs:
        if s.side == "long":
            assert s.stop < s.entry < s.take_profit
        else:
            assert s.take_profit < s.entry < s.stop
        assert 0 <= s.strength <= 1


def test_manage_moves_stop_to_breakeven_then_trails():
    c = cfg()
    d = manage_position("long", 100, 99.7, 99.7, price=100.31, best_price=100.31,
                        atr_now=0.1, bars_held=3, cfg=c)
    assert not d.close and d.new_stop > 100            # locked in above entry
    d2 = manage_position("long", 100, d.new_stop, 99.7, price=d.new_stop - 0.01,
                         best_price=100.31, atr_now=0.1, bars_held=4, cfg=c)
    assert d2.close
    d3 = manage_position("short", 100, 100.3, 100.3, price=100.1, best_price=99.9,
                         atr_now=0.1, bars_held=48, cfg=c)
    assert d3.close and d3.reason == "max hold time"


def test_backtest_losses_bounded_and_no_liquidation():
    c = cfg()
    frames = {"BTC/USDT:USDT": synthetic(8000, seed=5),
              "ETH/USDT:USDT": synthetic(8000, seed=6, start_price=3000)}
    stats, trades, curve = run_backtest(c, frames, 100)
    assert stats["trades"] > 0
    assert stats.get("liquidations", 0) == 0
    # every loss stays within the per-trade cap (small allowance for gaps through the stop)
    assert trades.pnl.min() > -(c.risk.max_loss_per_trade_usd * 1.5)
    assert trades.margin.between(5 - 1e-6, 15 + 1e-6).all()


# ---------- engine with a fake exchange ----------
class FakeBroker:
    def __init__(self, df):
        self.df, self.now = df, int(df.ts.iloc[-1]) + 300_000 + 1
        self.px = float(df.close.iloc[-1])
        self.equity = 100.0
        self.pos: dict[str, Position] = {}
        self.opened, self.closed = [], []

    def setup_symbol(self, s): pass
    def now_ms(self): return self.now
    def price(self, s): return self.px
    def balance(self): return self.equity, self.equity - sum(p.margin for p in self.pos.values())
    def positions(self): return dict(self.pos)
    def round_qty(self, s, q): return round(q, 4)
    def cancel_triggers(self, s): pass

    def ohlcv(self, s, tf, limit):
        return self.df if tf == "5m" else resample(self.df, tf)

    def open(self, s, side, qty, stop, tp):
        self.opened.append((s, side, qty, stop, tp))
        self.pos[s] = Position(s, side, qty, self.px, liquidation_price(self.px, side, 100, 0.005),
                               0, self.px * qty / 100)
        return self.px

    def close(self, s, p):
        self.closed.append(s)
        self.pos.pop(s)
        return self.px


def _df_ending_in_signal(c):
    df = synthetic(6000, seed=1)
    p = prepare(df, resample(df, c.trend_timeframe), c)
    for i in range(len(p) - 1, 500, -1):
        if signal_at(p.iloc[i], c):
            return df.iloc[: i + 1].reset_index(drop=True)
    raise AssertionError("no signal found")


def test_engine_opens_with_exchange_stop_and_tp(tmp_path):
    c = cfg()
    c.symbols, c.state_file = ["BTC/USDT:USDT"], str(tmp_path / "s.json")
    b = FakeBroker(_df_ending_in_signal(c))
    e = Engine(c, b)
    e.step()
    assert len(b.opened) == 1
    sym, side, qty, stop, tp = b.opened[0]
    assert (stop < b.px < tp) if side == "long" else (tp < b.px < stop)
    assert 5 - 1e-6 <= b.pos[sym].margin <= 15 + 1e-6
    assert Path(c.state_file).exists()


def test_engine_flattens_everything_on_daily_loss(tmp_path):
    c = cfg()
    c.symbols, c.state_file = ["BTC/USDT:USDT"], str(tmp_path / "s.json")
    b = FakeBroker(_df_ending_in_signal(c))
    e = Engine(c, b)
    e.step()
    assert b.pos
    b.equity = 69.0
    b.now += 1000
    e.step()
    assert b.closed == ["BTC/USDT:USDT"] and not b.pos
    b.equity, b.now = 69.0, b.now + 600_000
    n = len(b.opened)
    e.step()
    assert len(b.opened) == n                          # no new trades for the rest of the day


# ---------- high-frequency mode ----------
def test_hf_config_loads():
    c = load_config(ROOT / "config_hf.yaml")
    assert c.timeframe == "1m" and c.entry_order == "maker" and len(c.symbols) == 5
    assert c.risk.max_trades_per_day == 40 and c.risk.daily_max_loss_pct == 0.30
    assert (c.leverage, c.min_margin_usd, c.max_margin_usd) == (100, 5, 15)


def test_max_trades_per_day():
    c = cfg()
    c.risk.max_trades_per_day = 2
    rm = RiskManager(c)
    t0 = 1_700_000_000_000
    rm.update_equity(t0, 100)
    rm.record_open()
    rm.record_open()
    ok, why = rm.can_open(t0 + 1, "BTC/USDT:USDT", 0, 100)
    assert not ok and "per day" in why
    assert rm.can_open(t0 + DAY, "BTC/USDT:USDT", 0, 100)[0]


def test_maker_sizing_cheaper_than_market():
    c = cfg()
    rm = RiskManager(c)
    rm.update_equity(0, 100)
    market, _ = rm.size(100.0, 99.7, "long", 1, 100, 100)
    c.entry_order = "maker"
    maker, _ = rm.size(100.0, 99.7, "long", 1, 100, 100)
    assert maker.expected_loss / maker.notional < market.expected_loss / market.notional


def test_backtest_maker_mode_fees_and_bounds():
    c = load_config(ROOT / "config_hf.yaml")
    c.symbols = c.symbols[:2]
    frames = {s: synthetic(8000, seed=7 + i, start_price=[60_000, 3_000][i], step_ms=60_000)
              for i, s in enumerate(c.symbols)}
    stats, trades, _ = run_backtest(c, frames, 100)
    assert stats["trades"] > 0 and stats.get("liquidations", 0) == 0
    entry_fees = trades.fees - trades.exit * trades.qty * c.fees.taker
    assert (entry_fees / (trades.entry * trades.qty)).round(6).eq(c.fees.maker).all()
    assert trades.pnl.min() > -(c.risk.max_loss_per_trade_usd * 1.5)
    # never more than the daily cap of entries on any UTC+8 day
    day = (trades.open_time + 8 * 3_600_000) // DAY
    assert day.value_counts().max() <= c.risk.max_trades_per_day


class FakeMakerBroker(FakeBroker):
    def __init__(self, df):
        super().__init__(df)
        self.limits = {}

    def open_maker(self, s, side, qty, price, stop, tp):
        oid = str(len(self.limits) + 1)
        self.limits[oid] = {"s": s, "side": side, "qty": qty, "price": price, "status": "open"}
        return oid

    def order_status(self, s, oid):
        o = self.limits[oid]
        return o["status"], o["qty"] if o["status"] == "closed" else 0.0, o["price"]

    def cancel_order(self, s, oid):
        if self.limits[oid]["status"] == "open":
            self.limits[oid]["status"] = "canceled"

    def fill(self, oid):
        o = self.limits[oid]
        o["status"] = "closed"
        self.pos[o["s"]] = Position(o["s"], o["side"], o["qty"], o["price"],
                                    liquidation_price(o["price"], o["side"], 100, 0.005), 0,
                                    o["price"] * o["qty"] / 100)


def _maker_engine(tmp_path):
    c = cfg()
    c.symbols, c.state_file, c.entry_order = ["BTC/USDT:USDT"], str(tmp_path / "s.json"), "maker"
    b = FakeMakerBroker(_df_ending_in_signal(c))
    return c, b, Engine(c, b)


def test_engine_maker_entry_fills_and_is_tracked(tmp_path):
    c, b, e = _maker_engine(tmp_path)
    e.step()
    assert len(b.limits) == 1 and not b.opened and "BTC/USDT:USDT" in e.pending
    b.fill("1")
    b.now += 1000
    e.step()
    assert "BTC/USDT:USDT" in e.trades and not e.pending
    assert e.risk.state.opens_today == 1


def test_engine_maker_entry_cancelled_after_timeout(tmp_path):
    c, b, e = _maker_engine(tmp_path)
    e.step()
    b.now += c.maker_timeout_seconds * 1000 + 1
    e.step()
    assert b.limits["1"]["status"] == "canceled" and not e.pending and not e.trades


def test_capital_usd_makes_big_demo_account_act_like_100(tmp_path):
    c = cfg()
    c.symbols, c.state_file, c.capital_usd = ["BTC/USDT:USDT"], str(tmp_path / "s.json"), 100
    b = FakeBroker(_df_ending_in_signal(c))
    b.equity = 50_000.0
    e = Engine(c, b)
    e.step()
    assert b.opened and e.risk.state.day_start_equity == pytest.approx(100)
    b.equity = 50_000.0 - 31                            # bot lost 31 of its 100
    b.now += 1000
    e.step()
    assert not b.pos                                    # 30% daily limit fired


def test_bitget_leverage_falls_back_to_symbol_max():
    import ccxt
    from bot.exchange import BitgetBroker

    class FakeEx:
        def __init__(self):
            self.calls = []
        def market(self, s):
            return {"limits": {"leverage": {"max": 50}}}
        def set_position_mode(self, *a): pass
        def set_margin_mode(self, *a): pass
        def set_leverage(self, lev, s, params=None):
            self.calls.append(lev)
            if lev > 50:
                raise ccxt.ExchangeError('bitget {"code":"25223","msg":"Exceeds Max. leverage"}')

    b = BitgetBroker.__new__(BitgetBroker)
    b.cfg, b.ex, b.uta, b.lev = cfg(), FakeEx(), True, {}
    b.setup_symbol("DOGE/USDT:USDT")
    assert b.leverage_for("DOGE/USDT:USDT") == 50 and b.ex.calls[0] == 50
    b.ex.market = lambda s: {"limits": {"leverage": {"max": None}}}
    b.ex.calls.clear()
    b.setup_symbol("XRP/USDT:USDT")
    assert b.ex.calls[:2] == [100, 75] and b.leverage_for("XRP/USDT:USDT") == 50


def _broker_with_balance(raw, uta):
    from bot.exchange import BitgetBroker

    class FakeEx:
        def fetch_balance(self, params=None):
            return raw
        def fetch_positions(self, symbols=None):
            return []
    b = BitgetBroker.__new__(BitgetBroker)
    b.cfg, b.ex, b.uta, b.lev = cfg(), FakeEx(), uta, {}
    return b


def test_balance_reads_unified_account():
    raw = {"info": {"data": {"usdtEquity": "50012.5", "accountEquity": "50012.5",
                             "unionAvailable": "49000", "assets": []}},
           "USDT": {"free": 49000.0, "total": 50012.5}}
    assert _broker_with_balance(raw, True).balance() == (50012.5, 49000.0)


def test_balance_reads_demo_susdt_coin():
    raw = {"info": [{"marginCoin": "SUSDT", "accountEquity": "3000"}],
           "SUSDT": {"free": 2900.0, "total": 3000.0}, "USDT": {"free": 0, "total": 0}}
    assert _broker_with_balance(raw, False).balance() == (3000.0, 2900.0)


def test_zero_balance_never_opens_and_keeps_no_baseline(tmp_path):
    c = cfg()
    c.symbols, c.state_file, c.capital_usd = ["BTC/USDT:USDT"], str(tmp_path / "s.json"), 100
    b = FakeBroker(_df_ending_in_signal(c))
    b.equity = 0.0
    e = Engine(c, b)
    e.step()
    assert not b.opened and e.risk.state.equity_baseline == 0
    assert not e.risk.state.killed
