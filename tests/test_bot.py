import math
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
from bot.strategy import TF_MS, manage_position, prepare, signal_at  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAY = 86_400_000


def cfg() -> Config:
    """config.yaml with the 5m/1h market-entry timing these mechanics tests were built on."""
    c = load_config(ROOT / "config.yaml")
    c.timeframe, c.trend_timeframe = "5m", "1h"
    c.entry_order, c.maker_timeout_seconds, c.tp_order = "market", 60, "market"
    return c


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
    assert rm.size(100.0, 98.5, "long", 1, 100, 100)[0] is None    # 1.5%: wider than max
    assert rm.size(100.0, 99.8, "long", 1, 100, 100)[0] is None    # 0.2%: noise
    assert rm.size(100.0, 100.3, "short", 1, 100, 100)[0] is not None


def test_cross_margin_allows_wide_stop_that_isolated_would_liquidate():
    c = cfg()
    assert c.margin_mode == "cross" and c.max_open_positions == 1
    rm = RiskManager(c)
    rm.update_equity(0, 100)
    s, _ = rm.size(100.0, 99.0, "long", 1, 100, 100)              # 1% stop at 100x
    assert s and s.expected_loss <= 10 + 1e-9 and s.liq_price < 99.0
    c.margin_mode = "isolated"                                    # 逐仓 liquidates at ~0.5%
    assert RiskManager(c, rm.state).size(100.0, 99.0, "long", 1, 100, 100)[0] is None


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
    d = manage_position("long", 100, 99.7, 99.7, price=100.40, best_price=100.40,
                        atr_now=0.1, bars_held=3, cfg=c)
    assert not d.close and d.new_stop > 100            # locked in above entry
    d2 = manage_position("long", 100, d.new_stop, 99.7, price=d.new_stop - 0.01,
                         best_price=100.40, atr_now=0.1, bars_held=4, cfg=c)
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
    assert trades.margin.between(5 - 1e-6, 15 + 1e-6).all()   # both batches together


# ---------- engine with a fake exchange ----------
class FakeBroker:
    def __init__(self, df):
        self.df, self.now = df, int(df.ts.iloc[-1]) + 300_000 + 1
        self.px = float(df.close.iloc[-1])
        self.equity = 100.0
        self.pos: dict[str, Position] = {}
        self.opened, self.closed, self.reduced = [], [], []

    def setup_symbol(self, s): pass
    def now_ms(self): return self.now
    def price(self, s): return self.px
    def balance(self): return self.equity, self.equity - sum(p.margin for p in self.pos.values())
    def positions(self): return dict(self.pos)
    def round_qty(self, s, q): return math.floor(q * 1e4) / 1e4  # ccxt truncates
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

    def reduce(self, s, side, qty):
        self.reduced.append((s, qty))
        p = self.pos[s]
        self.pos[s] = Position(s, side, round(p.qty - qty, 8), p.entry, p.liq_price, 0,
                               p.margin * (p.qty - qty) / p.qty)
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
    half = c.strategy.scale_in_frac                     # 分批进场: first batch only
    assert 5 * half - 1e-6 <= b.pos[sym].margin <= 15 * half + 1e-6
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
    assert (c.timeframe, c.trend_timeframe, c.entry_order, c.tp_order) == ("5m", "1h", "market", "maker")
    assert c.replace_on_new_signal and c.strategy.take_profit_r == [1.0]
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
    c.symbols, c.entry_order, c.tp_order = c.symbols[:2], "maker", "market"
    c.timeframe, c.trend_timeframe = "1m", "15m"
    frames = {s: synthetic(8000, seed=7 + i, start_price=[60_000, 3_000][i], step_ms=60_000)
              for i, s in enumerate(c.symbols)}
    stats, trades, _ = run_backtest(c, frames, 100)
    assert stats["trades"] > 0 and stats.get("liquidations", 0) == 0
    first = trades[trades.adds == 0]  # adds (平加仓) are taker orders
    entry_fees = first.fees - first.exit * first.qty * c.fees.taker
    assert (entry_fees / (first.entry * first.qty)).round(6).eq(c.fees.maker).all()
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


def test_engine_fast_maker_fill_is_not_adopted_as_unknown(tmp_path):
    c, b, e = _maker_engine(tmp_path)
    e.step()
    pe = e.pending["BTC/USDT:USDT"]
    # position appears on the exchange while the order still reports "open"
    o = b.limits["1"]
    b.pos[o["s"]] = Position(o["s"], o["side"], o["qty"], o["price"],
                             liquidation_price(o["price"], o["side"], 100, 0.005), 0,
                             o["price"] * o["qty"] / 100)
    b.now += 1000
    e.step()
    t = e.trades["BTC/USDT:USDT"]
    assert t.initial_stop == pe.stop and not e.pending


# ---------- the video's way: 分批进场, 补仓 -> 平加仓, 分批止盈, off round numbers ----------
def video_cfg() -> Config:
    """The 大镖客 video settings (switched off in the shipped configs for now)."""
    c = cfg()
    st = c.strategy
    st.scale_in_frac, st.scale_in_at_r, st.avoid_round_numbers = 0.5, 0.5, True
    st.take_profit_r, st.take_profit_split = [1.2, 2.0, 3.0], [0.34, 0.33]
    return c


def test_stop_and_take_profits_avoid_round_numbers():
    from bot.strategy import off_round, round_step
    assert [round_step(x) for x in (83000, 2455, 150, 1.33, 0.15)] == \
        pytest.approx([100, 10, 0.1, 0.001, 0.0001])
    # the video's BTC long: stop 60700 -> 60677, take-profits 61800/62200/62700 -> 617xx...
    assert off_round(60700, "long", "stop") == 60675
    assert [off_round(x, "long", "tp") for x in (61800, 62200, 62700)] == [61765, 62165, 62665]
    assert off_round(60780, "long", "stop") == 60780       # not near a round level: unchanged
    assert off_round(2450, "short", "stop") == 2452.5
    assert off_round(2430, "short", "tp") == 2433.5


def test_trade_plan_ladder():
    from bot.strategy import trade_plan
    c = video_cfg()
    pl = trade_plan("long", 61200, 500, c)
    assert pl.stop < 60700 and pl.add_price == 60950
    assert len(pl.take_profits) == 3 and pl.take_profits == sorted(pl.take_profits)
    assert all(61200 < x for x in pl.take_profits)


def _scaling_engine(tmp_path):
    c = video_cfg()
    c.symbols, c.state_file = ["BTC/USDT:USDT"], str(tmp_path / "s.json")
    b = FakeBroker(_df_ending_in_signal(c))

    def open_(s, side, qty, stop, tp):  # adds merge into one position like on the exchange
        b.opened.append((s, side, qty, stop, tp))
        old = b.pos.get(s)
        q, e0 = (old.qty + qty, (old.entry * old.qty + b.px * qty) / (old.qty + qty)) if old \
            else (qty, b.px)
        b.pos[s] = Position(s, side, q, e0, liquidation_price(e0, side, 100, 0.005), 0, e0 * q / 100)
        return b.px
    b.open = open_
    return c, b, Engine(c, b)


def test_engine_scale_in_then_reduce_at_cost_then_partial_take_profits(tmp_path):
    c, b, e = _scaling_engine(tmp_path)
    sym = "BTC/USDT:USDT"
    e.step()
    t = e.trades[sym]
    sign = 1 if t.side == "long" else -1
    first = t.qty
    assert t.add_plan > 0 and len(t.tps) == 2

    def at(px):
        b.px, b.now = px, b.now + 1000
        e.step()
        return e.trades[sym]

    t = at(t.add_price)                                   # 补仓 with the same stop
    assert t.add_state == "open" and len(b.opened) == 2 and b.opened[1][3] == t.initial_stop
    assert t.qty == pytest.approx(first + t.add_plan)
    t = at(t.avg_entry * (1 + sign * 0.003))                # back above the new cost: 平加仓
    assert t.add_state == "done" and b.reduced[-1][1] == t.add_plan
    assert t.qty == pytest.approx(first) and t.realized > 0
    t = at(t.tps[0])                                      # first take-profit
    assert len(b.reduced) == 2 and sign * (t.stop - t.entry) > 0   # 移动止损 to cost
    assert t.add_state == "done"
    t = at(t.tps[0])                                      # second take-profit
    assert len(b.reduced) == 3 and not t.tps and t.qty > 0


def test_backtest_scale_in_happens_and_losses_stay_bounded():
    c = video_cfg()
    frames = {"BTC/USDT:USDT": synthetic(8000, seed=5),
              "ETH/USDT:USDT": synthetic(8000, seed=6, start_price=3000)}
    stats, trades, _ = run_backtest(c, frames, 100)
    assert (trades.adds > 0).any()
    assert stats.get("liquidations", 0) == 0
    assert trades.pnl.min() > -(c.risk.max_loss_per_trade_usd * 1.5)


def test_cross_margin_param_per_account_type():
    # unified (UTA) order endpoint rejects "cross" (Bitget 40034) and wants "crossed"
    assert _broker_with_balance({}, uta=True).order_margin_mode == "crossed"
    assert _broker_with_balance({}, uta=False).order_margin_mode == "cross"   # ccxt maps it
    b = _broker_with_balance({}, uta=True)
    b.cfg.margin_mode = "isolated"
    assert b.order_margin_mode == "isolated"


def test_take_profit_ladder_keeps_distinct_levels_on_eth():
    from bot.strategy import trade_plan
    # ETH 2486.82 with an ~8 point stop: 1.2R and 2R both sit next to 2500 (seen on demo)
    tps = trade_plan("long", 2486.82, 8, video_cfg()).take_profits
    assert tps == sorted(set(tps)) and len(tps) == 3


def test_only_one_position_even_with_two_signals_in_one_loop(tmp_path):
    c = cfg()
    c.symbols, c.state_file = ["BTC/USDT:USDT", "ETH/USDT:USDT"], str(tmp_path / "s.json")
    b = FakeBroker(_df_ending_in_signal(c))   # same candles -> both coins signal together
    Engine(c, b).step()
    assert len(b.opened) == 1


def test_scale_in_off_opens_first_batch_only(tmp_path):
    c, b, e = _scaling_engine(tmp_path)
    c.strategy.scale_in_at_r = 0
    e.step()
    t = e.trades["BTC/USDT:USDT"]
    assert t.add_plan == 0 and t.add_price == 0
    b.px, b.now = t.initial_stop + (t.entry - t.initial_stop) * 0.1, b.now + 1000
    e.step()
    assert len(b.opened) == 1                    # no 补仓 even deep in the loss


def test_shipped_configs_trade_full_size_with_one_take_profit():
    for path in ("config.yaml", "config_hf.yaml"):
        st = load_config(ROOT / path).strategy
        assert st.scale_in_frac == 1 and st.scale_in_at_r == 0 and len(st.take_profit_r) == 1


def test_trend_min_gap_filters_weak_trends():
    c = cfg()
    df = _df_ending_in_signal(c)
    row = prepare(df, resample(df, c.trend_timeframe), c).iloc[-1]
    assert signal_at(row, c)
    gap = abs(row["trend_close"] - row["trend_ema"]) / row["trend_ema"]
    c.strategy.trend_min_gap = gap * 1.01
    assert signal_at(row, c) is None


@pytest.mark.parametrize("kind", ["pullback", "reversion"])
def test_new_signal_types_fire_with_stop_on_correct_side(kind):
    c = cfg()
    c.strategy.signal_type = kind
    c.strategy.volume_factor = 0
    df = synthetic(6000, seed=1)
    p = prepare(df, resample(df, c.trend_timeframe), c)
    sigs = [s for s in (signal_at(r, c) for _, r in p.iterrows()) if s]
    assert sigs
    for s in sigs:
        assert kind in s.reason
        if s.side == "long":
            assert s.stop < s.entry < s.take_profit
        else:
            assert s.take_profit < s.entry < s.stop


def test_tune_variants_cover_every_signal_and_config_is_valid():
    import main
    from bot.config import validate
    variants = main.tune_variants()
    assert len(variants) == 4
    assert {v[1]["timeframe"] for v in variants} == {"30m", "1h"}
    assert {v[2]["signal_type"] for v in variants} == {"breakout"}
    for _, top, change, _ in variants:
        c = load_config(ROOT / "config.yaml")
        for k, v in top.items():
            setattr(c, k, v)
        for k, v in change.items():
            setattr(c.strategy, k, v)
        validate(c)


def test_candle_download_retries_after_too_many_requests(monkeypatch):
    import ccxt
    from bot import data
    monkeypatch.setattr(data.time, "sleep", lambda s: None)

    class Ex:
        calls = 0

        def fetch_ohlcv(self, *a):
            Ex.calls += 1
            if Ex.calls < 3:
                raise ccxt.DDoSProtection('bitget {"code":"429","msg":"Too Many Requests"}')
            return [[1, 1, 1, 1, 1, 1]]

    assert data._fetch(Ex(), "BTC/USDT:USDT", "1m", 0) == [[1, 1, 1, 1, 1, 1]]
    assert Ex.calls == 3


def test_normal_mode_is_1h_breakout_with_limit_entry_waiting_one_bar():
    c = load_config(ROOT / "config.yaml")
    assert (c.timeframe, c.trend_timeframe, c.entry_order, c.tp_order) == ("1h", "4h", "maker", "maker")
    assert c.strategy.signal_type == "breakout"
    assert c.maker_timeout_seconds * 1000 == TF_MS["1h"]


def test_score_signal_counts_votes_and_explains_itself():
    from bot.config import VOTES
    c = cfg()
    c.strategy.signal_type, c.strategy.min_score, c.strategy.volume_factor = "score", 5, 0
    df = synthetic(6000, seed=2)
    p = prepare(df, resample(df, c.trend_timeframe), c)
    cols = [f"v_{v}" for v in VOTES]
    assert p[cols].isin([-1, 0, 1]).all().all()
    assert (p["score"] == p[cols].sum(axis=1)).all()
    sigs = [(r, s) for r, s in ((r, signal_at(r, c)) for _, r in p.iterrows()) if s]
    assert sigs
    for r, s in sigs:
        want = 1 if s.side == "long" else -1
        assert r["score"] * want >= 5 > r["prev_score"] * want
        assert "综合" in s.reason
        if s.side == "long":
            assert s.stop < s.entry < s.take_profit
        else:
            assert s.take_profit < s.entry < s.stop


def test_score_has_no_lookahead():
    c = cfg()
    c.strategy.signal_type = "score"
    df = synthetic(3000, seed=3)
    full = prepare(df, resample(df, c.trend_timeframe), c)
    cut = df.iloc[:2000]
    part = prepare(cut, resample(cut, c.trend_timeframe), c)
    pd.testing.assert_series_equal(full["score"].iloc[:1990].reset_index(drop=True),
                                   part["score"].iloc[:1990].reset_index(drop=True))


def _minute_walk(n, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    p = 60000 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    o = np.r_[p[0], p[:-1]]
    return pd.DataFrame(dict(ts=1_700_000_000_000 + np.arange(n) * 60_000, open=o,
                             high=np.maximum(o, p) * (1 + rng.uniform(0, 5e-4, n)),
                             low=np.minimum(o, p) * (1 - rng.uniform(0, 5e-4, n)),
                             close=p, volume=rng.uniform(1, 10, n)))


def test_minute_bars_group_by_parent_bar():
    from bot.backtest import minute_bars
    df = _minute_walk(30, 1)
    m = minute_bars(df, "5m")
    assert sum(len(v) for v in m.values()) == 30
    for k, v in m.items():
        assert k % 300_000 == 0 and len(v) <= 5


def test_minute_exits_remove_same_bar_pessimism_for_small_take_profits():
    df = _minute_walk(60 * 24 * 20, 7)
    c = cfg()
    c.entry_order = "maker"
    c.strategy.signal_type, c.strategy.min_score, c.strategy.volume_factor = "score", 4, 0
    c.strategy.take_profit_r = [0.5]
    c.risk.min_equity_stop_usd, c.capital_usd = 0, 0
    sym = "BTC/USDT:USDT"
    _, coarse, _ = run_backtest(c, {sym: resample(df, "5m")}, 100_000)
    _, fine, _ = run_backtest(c, {sym: resample(df, "5m")}, 100_000, {sym: df})
    assert len(fine) > 50
    assert (fine.pnl > 0).mean() > (coarse.pnl > 0).mean()
    assert fine.pnl.min() >= -c.risk.max_loss_per_trade_usd - 1e-6


class FakeTpBroker(FakeMakerBroker):
    def __init__(self, df, refuse=False):
        super().__init__(df)
        self.tps, self.cancelled, self.refuse = [], [], refuse

    def place_tp(self, s, side, qty, price):
        if self.refuse:
            raise RuntimeError("post only would take")
        self.tps.append((s, side, qty, price))
        return f"tp{len(self.tps)}"

    def cancel_order(self, s, oid):
        self.cancelled.append(oid)
        if oid in self.limits:
            super().cancel_order(s, oid)


def _limit_tp_engine(tmp_path, refuse=False):
    c = cfg()
    c.symbols, c.state_file, c.tp_order = ["BTC/USDT:USDT"], str(tmp_path / "s.json"), "maker"
    b = FakeTpBroker(_df_ending_in_signal(c), refuse)
    return c, b, Engine(c, b)


def test_limit_take_profit_rests_after_entry_and_is_cancelled_when_stopped(tmp_path):
    c, b, e = _limit_tp_engine(tmp_path)
    e.step()
    (s, side, qty, stop, tp), = b.opened
    assert tp == 0  # no trigger take-profit on the exchange
    t = e.trades[s]
    assert b.tps == [(s, side, qty, t.take_profit)] and t.tp_order_id == "tp1"
    b.pos.clear()  # exchange stop closed the position
    e.step()
    assert "tp1" in b.cancelled and s not in e.trades


def test_limit_take_profit_refused_bot_closes_at_market(tmp_path):
    c, b, e = _limit_tp_engine(tmp_path, refuse=True)
    e.step()
    (s, side, *_), = b.opened
    t = e.trades[s]
    assert t.tp_order_id == "" and t.tp_tries >= 1
    b.px = t.take_profit + (1 if side == "long" else -1)
    e.step()
    assert s in b.closed


def test_limit_take_profit_needs_single_entry_and_target():
    from bot.config import validate
    c = cfg()
    c.tp_order = "maker"
    validate(c)
    c.strategy.take_profit_r, c.strategy.take_profit_split = [1.0, 2.0], [0.5]
    with pytest.raises(ValueError):
        validate(c)


class FakeBookEx:
    def __init__(self):
        self.now, self.trades = 1_700_000_000_000, []

    def milliseconds(self):
        return self.now

    def fetch_order_book(self, s, limit):
        bids = [[100 - i, 1.0] for i in range(limit)]
        asks = [[101 + i, 2.0] for i in range(limit)]
        return {"bids": bids, "asks": asks, "timestamp": self.now}

    def fetch_trades(self, s, since, limit):
        return self.trades[-limit:]


def test_order_book_recorder_counts_only_new_trades(tmp_path):
    from bot.orderflow import COLUMNS, Recorder
    ex = FakeBookEx()
    ex.trades = [{"id": "1", "side": "buy", "amount": 5.0, "price": 100.5}]
    rec = Recorder(ex, ["BTC/USDT:USDT"], tmp_path)
    rec.poll()  # first poll: trade 1 happened before we started
    ex.now += 1000
    ex.trades += [{"id": "2", "side": "buy", "amount": 1.5, "price": 101},
                  {"id": "3", "side": "sell", "amount": 0.5, "price": 100}]
    rec.poll()
    rows = pd.read_csv(rec.path("BTC/USDT:USDT", ex.now))
    assert list(rows.columns) == COLUMNS and len(rows) == 2
    first, second = rows.iloc[0], rows.iloc[1]
    assert first.trades == 0 and second.trades == 2
    assert second.buy_qty == 1.5 and second.sell_qty == 0.5
    assert second.bid == 100 and second.ask == 101
    assert second.bid_qty5 == 5 and second.ask_qty15 == 30
    assert second.maybe_missed == 0


class FakeSelftestBroker:
    def __init__(self, refuse_tp=False):
        self.px, self.pos, self.orders, self.refuse_tp, self.calls = 60000.0, {}, {}, refuse_tp, []

    def setup_symbol(self, s): self.calls.append("setup")
    def price(self, s): return self.px
    def leverage_for(self, s): return 100
    def round_qty(self, s, q): return math.floor(q * 1e4) / 1e4
    def positions(self): return dict(self.pos)

    def open_maker(self, s, side, qty, price, stop, tp):
        self.orders["e"] = "open"
        return "e"

    def order_status(self, s, oid):
        return self.orders[oid], 0.0, 0.0

    def cancel_order(self, s, oid):
        if oid in self.orders:
            self.orders[oid] = "canceled"

    def open(self, s, side, qty, stop, tp):
        assert tp == 0
        self.pos[s] = Position(s, side, qty, self.px, 0, 0, 10)
        return self.px

    def place_tp(self, s, side, qty, price):
        if self.refuse_tp:
            raise RuntimeError("40000 reduceOnly postOnly not supported")
        self.orders["tp"] = "open"
        return "tp"

    def close(self, s, p):
        self.pos.pop(s)
        return self.px


def test_selftest_runs_every_order_type_and_leaves_nothing_open():
    from bot.selftest import report, run_selftest
    b = FakeSelftestBroker()
    res = run_selftest(b, "BTC/USDT:USDT", 10, pause=0)
    assert [ok for _, ok, _ in res] == [True] * 6
    assert not b.pos and b.orders == {"e": "canceled", "tp": "canceled"}
    assert "全部通过" in report("BTC/USDT:USDT", res)


def test_selftest_reports_refused_take_profit_and_still_closes():
    from bot.selftest import report, run_selftest
    b = FakeSelftestBroker(refuse_tp=True)
    res = run_selftest(b, "BTC/USDT:USDT", 10, pause=0)
    assert any(not ok and "postOnly" in d for _, ok, d in res)
    assert not b.pos
    assert "没通过" in report("BTC/USDT:USDT", res)


def test_selftest_refuses_when_coin_already_has_a_position():
    from bot.selftest import run_selftest
    b = FakeSelftestBroker()
    b.pos["BTC/USDT:USDT"] = Position("BTC/USDT:USDT", "long", 1, 1, 0, 0, 1)
    res = run_selftest(b, "BTC/USDT:USDT", 10, pause=0)
    assert len(res) == 1 and not res[0][1] and "setup" not in b.calls


def _replace_engine(tmp_path, protected):
    from bot.engine import Trade
    c = cfg()
    c.symbols, c.state_file = ["BTC/USDT:USDT"], str(tmp_path / "s.json")
    c.max_open_positions, c.replace_on_new_signal = 1, True
    b = FakeBroker(_df_ending_in_signal(c))
    e = Engine(c, b)
    old = "ETH/USDT:USDT"
    b.pos[old] = Position(old, "long", 1, 3000, 2970, 0, 30)
    stop = 3001 if protected else 2990
    e.trades[old] = Trade("long", 1, 3000, 2990, stop, 3020, 3000, b.now - 2 * 300_000)
    return c, b, e, old


def test_new_signal_replaces_trade_that_is_not_yet_protected(tmp_path):
    c, b, e, old = _replace_engine(tmp_path, protected=False)
    e.step()
    assert old in b.closed and old not in e.trades
    assert [o[0] for o in b.opened] == ["BTC/USDT:USDT"]
    assert e.risk.state.cooldown_until_ms.get(old, 0) == 0  # a swap is not a stop-out


def test_new_signal_keeps_trade_already_at_break_even(tmp_path):
    c, b, e, old = _replace_engine(tmp_path, protected=True)
    e.step()
    assert old not in b.closed and not b.opened


def test_zero_consecutive_losses_means_never_pause():
    c = cfg()
    c.risk.max_consecutive_losses = 0
    rm = RiskManager(c)
    for i in range(10):
        rm.record_close(i, "BTC/USDT:USDT", -1.0, False, 60_000)
    assert rm.state.paused_until_ms == 0


def test_high_frequency_mode_never_waits_after_a_stop():
    c = load_config(ROOT / "config_hf.yaml")
    rm = RiskManager(c)
    rm.record_close(1_000, "BTC/USDT:USDT", -2.0, True, 900_000)
    assert rm.can_open(1_000, "BTC/USDT:USDT", 0, 100)[0]


def test_trade_opened_this_bar_is_not_swapped_out(tmp_path):
    c = cfg()
    c.symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    c.state_file, c.max_open_positions, c.replace_on_new_signal = str(tmp_path / "s.json"), 1, True
    b = FakeBroker(_df_ending_in_signal(c))   # same candles: all three coins signal together
    e = Engine(c, b)
    e.step()
    assert len(b.pos) == 1 and set(e.trades) == set(b.pos)
    assert not b.closed  # first signal wins; no ping-pong within one bar


def test_fresh_trade_from_earlier_loop_is_not_swapped_out(tmp_path):
    c, b, e, old = _replace_engine(tmp_path, protected=False)
    e.trades[old].opened_ms = b.now
    e.step()
    assert old not in b.closed and old in e.trades and not b.opened


def test_cancel_of_already_gone_order_is_not_an_error():
    import ccxt
    from bot.exchange import BitgetBroker
    br = BitgetBroker.__new__(BitgetBroker)

    class Ex:
        def cancel_order(self, oid, sym):
            raise ccxt.ExchangeError('bitget {"code":"25204","msg":"Order does not exist"}')
    br.ex = Ex()
    br.cancel_order("BTC/USDT:USDT", "123")  # must not raise


def test_old_trade_is_kept_when_the_new_signal_is_refused_by_sizing(tmp_path):
    c, b, e, old = _replace_engine(tmp_path, protected=False)
    c.risk.max_stop_pct = c.risk.min_stop_pct * 1.0001  # every new stop is "too wide"
    e.step()
    assert old not in b.closed and old in e.trades and not b.opened


def test_coin_without_futures_market_is_skipped_not_fatal(tmp_path):
    c = cfg()
    c.symbols, c.state_file = ["BTC/USDT:USDT", "BGB/USDT:USDT"], str(tmp_path / "s.json")
    b = FakeBroker(_df_ending_in_signal(c))
    b.has_symbol = lambda s: s != "BGB/USDT:USDT"
    b.setup_symbol = lambda s: None if s != "BGB/USDT:USDT" else 1 / 0
    e = Engine(c, b)
    e.step = lambda: (_ for _ in ()).throw(KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        e.run()
    assert c.symbols == ["BTC/USDT:USDT"]
