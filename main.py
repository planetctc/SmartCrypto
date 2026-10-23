"""SmartCrypto - Bitget USDT-M futures bot.

用法:
  python main.py download --days 180            下载历史 K 线
  python main.py backtest                        用下载的数据回测
  python main.py backtest --synthetic            用随机生成的数据跑一遍（只测试程序，不代表真实表现）
  python main.py compare                         普通模式和高频模式回测对比
  （高频模式：所有命令加 --config config_hf.yaml）
  python main.py run                             按 config.yaml 的 mode 实时运行（paper / demo / live）
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

BANNER = r"""
  ┌──────────────────────────────────────┐
  │   SmartCrypto  ·  Bitget Futures     │
  └──────────────────────────────────────┘
"""


def setup_logging(log_file: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handlers = [RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")]
    if sys.stdout is not None:  # no console when started in the background (pyw)
        handlers.append(logging.StreamHandler(sys.stdout))
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


def cmd_download(cfg, args):
    from bot.data import download
    for sym in cfg.symbols:
        download(sym, cfg.timeframe, args.days, args.exchange, args.data_dir)


def load_frames(cfg, args, config_path):
    from bot.backtest import synthetic
    from bot.data import csv_path, load_csv
    if args.synthetic:
        print("注意：这是随机生成的数据，只用来检查程序，不代表真实行情表现。")
        step = {"1m": 60_000, "5m": 300_000}.get(cfg.timeframe, 300_000)
        return {s: synthetic(seed=i, start_price=[60_000, 3_000][i % 2], step_ms=step)
                for i, s in enumerate(cfg.symbols)}
    frames = {}
    for s in cfg.symbols:
        p = csv_path(args.data_dir, args.exchange, s, cfg.timeframe)
        if not p.exists():
            sys.exit(f"找不到 {p}，先运行: python main.py download --config {config_path} "
                     f"--exchange {args.exchange}")
        frames[s] = load_csv(p)
    return frames


def cmd_backtest(cfg, args):
    from bot.backtest import run_backtest
    stats, trades, curve = run_backtest(cfg, load_frames(cfg, args, args.config), args.balance)
    out = Path(args.data_dir)
    out.mkdir(exist_ok=True)
    trades.to_csv(out / "backtest_trades.csv", index=False)
    curve.to_csv(out / "backtest_equity.csv", index=False)
    print("\n===== 回测结果 =====")
    for k, v in stats.items():
        print(f"{k:>24}: {v}")
    print(f"\n每笔交易: {out / 'backtest_trades.csv'}\n资金曲线: {out / 'backtest_equity.csv'}")


def cmd_compare(_, args):
    from bot.backtest import run_backtest
    from bot.config import load_config
    results = {}
    for path in args.configs:
        c = load_config(path)
        stats, _, _ = run_backtest(c, load_frames(c, args, path), args.balance)
        results[path] = stats
    keys = ["days", "trades", "return_pct", "end_equity", "win_rate_pct", "profit_factor",
            "max_drawdown_pct", "days_daily_limit_hit", "liquidations", "fees_paid"]
    w = max(len(p) for p in results) + 2
    print("\n===== 模式对比 =====")
    print(f"{'':>22}" + "".join(f"{p:>{w}}" for p in results))
    for k in keys:
        vals = [results[p].get(k, "-") for p in results]
        vals = [round(v, 2) if isinstance(v, float) else v for v in vals]
        print(f"{k:>22}" + "".join(f"{str(v):>{w}}" for v in vals))
    days = {p: max(r.get("days", 0) or 0, 1e-9) for p, r in results.items()}
    print(f"{'trades_per_day':>22}" + "".join(
        f"{round(results[p]['trades'] / days[p], 1):>{w}}" for p in results))


SIGNALS = {"breakout": "突破", "pullback": "顺势回踩", "reversion": "超买超卖反转"}


MIN_TRADES = 20  # fewer trades than this over the whole period proves nothing
SCALP_TIMEFRAMES = (("5 分钟", "5m", "1h"), ("15 分钟", "15m", "4h"))


def tune_variants():
    """(name, top-level config changes, strategy changes, show indicators) for every
    combination the panel compares.

    90 days of real data: all three 1-minute setups lost even before fees (win rate ~30%),
    and fees are about 0.7 USDT a trade at 10 margin x 100. So this round searches wider:
    the same setups faded (opposite side), bigger targets so a win covers its fees, and
    3/5/15-minute bars that move enough per trade.
    """
    out = []
    base = {"breakout_lookback": 15, "trend_ema": 50, "volume_factor": 1.0, "trend_min_gap": 0}
    for tf_name, tf, trend, hold in (("1 分钟", "1m", "15m", 10), ("5 分钟", "5m", "1h", 24),
                                     ("15 分钟", "15m", "4h", 24)):
        top = {"timeframe": tf, "trend_timeframe": trend, "entry_order": "maker",
               "tp_order": "maker", "maker_timeout_seconds": 60 if tf == "1m" else 300}
        b = {**base, "max_hold_bars": hold}
        out += [
            (f"{tf_name}·先看行情·止盈 2 倍", top, {**b, "signal_type": "adaptive",
                                                  "take_profit_r": [2.0]}, False),
            (f"{tf_name}·只做横盘·止盈 1 倍", top, {**b, "signal_type": "adaptive",
                                                  "range_trend_gap": 1.0, "take_profit_r": [1.0]}, False),
            (f"{tf_name}·反着做（顺势信号反过来）·止盈 1 倍", top,
             {**b, "signal_type": "combo", "reverse": True, "take_profit_r": [1.0]}, False),
            (f"{tf_name}·顺势突破+反弹·止盈 3 倍", top, {**b, "signal_type": "combo",
                                                      "take_profit_r": [3.0]}, False),
        ]
    return out


def indicator_lines(trades, cut):
    """For the score signal: how trades did when each indicator agreed, in both parts."""
    from bot.config import VOTES
    avg = lambda t: t.pnl.mean() if len(t) else 0.0  # noqa: E731
    first, last = trades[trades.open_time < cut], trades[trades.open_time >= cut]
    lines = ["   每个指标同意时，平均每单赚多少（已扣手续费；比全部单平均好的指标才有用）：",
             f"     全部单：挑选段 {avg(first):+.3f}，验证段 {avg(last):+.3f}"]
    for key, name in VOTES.items():
        has = trades.votes.str.split(",").apply(lambda v, k=key: k in v)
        tr, te = trades[has & (trades.open_time < cut)], trades[has & (trades.open_time >= cut)]
        lines.append(f"     {name}：挑选段 {avg(tr):+.3f}（{len(tr)} 单），验证段 {avg(te):+.3f}（{len(te)} 单）")
    return lines


def _halves(trades, cut):
    """(net, count, win %, net before fees) for the trades opened before and after cut."""
    part = {}
    for key, sel in (("train", trades.open_time < cut if len(trades) else None),
                     ("test", trades.open_time >= cut if len(trades) else None)):
        tr = trades[sel] if len(trades) else trades
        part[key] = (float(tr.pnl.sum()) if len(tr) else 0.0, len(tr),
                     float((tr.pnl > 0).mean() * 100) if len(tr) else 0.0,
                     float((tr.pnl + tr.fees).sum()) if len(tr) else 0.0)
    return part


def cmd_tune(cfg, args):
    """Download recent real 1m candles and compare timeframes, entries and which coins to trade.

    Choosing the best of many combinations on the same data flatters it, so everything is
    chosen on the first two thirds of the period (including which coins to keep) and the last
    third is reported separately as the test the choice did not see. This is what the control
    panel's 回测比较 button runs.
    """
    import ccxt
    from bot.backtest import resample, run_backtest
    from bot.config import load_config
    from bot.data import download, load_csv
    out = Path(args.data_dir)
    out.mkdir(exist_ok=True)
    progress = out / "tune_progress.txt"

    def say(msg):
        print(msg)
        progress.write_text(msg, encoding="utf-8")

    base = {}
    for i, sym in enumerate(cfg.symbols, 1):
        say(f"下载 {sym.split('/')[0]} 最近 {args.days} 天的 K 线（{i}/{len(cfg.symbols)}）…")
        try:
            base[sym] = load_csv(download(sym, "1m", args.days, args.exchange, args.data_dir))
        except ccxt.BadSymbol:
            say(f"{sym.split('/')[0]} 在 Bitget 没有合约，跳过")
    cfg.symbols = list(base)
    if not base:
        sys.exit("选的币 Bitget 都没有合约")
    t0 = min(f["ts"].iloc[0] for f in base.values())
    t1 = max(f["ts"].iloc[-1] for f in base.values())
    cut = t0 + (t1 - t0) * 2 / 3
    day = 86_400_000
    days = (t1 - t0) / day
    short = lambda syms: "、".join(x.split("/")[0] for x in syms)  # noqa: E731

    def backtest(top, change, syms):
        c = load_config("config.yaml")
        for k, v in top.items():
            setattr(c, k, v)
        apply_overrides(c, ",".join(x.split("/")[0] for x in syms), cfg.max_margin_usd)
        # a large balance keeps a losing variant trading to the end, so every one is measured
        # on the whole period; results are summed per trade with the per-trade loss cap in place
        c.risk.min_equity_stop_usd = 0
        c.capital_usd = 0
        for k, v in change.items():
            setattr(c.strategy, k, v)
        frames = {x: resample(base[x], c.timeframe) for x in syms}
        # stops and take-profits checked minute by minute inside each bar
        return run_backtest(c, frames, 100_000.0, {x: base[x] for x in syms})[1]

    variants = tune_variants()
    rows = []
    for i, (name, top, change, show) in enumerate(variants, 1):
        say(f"回测 {i}/{len(variants)}：{name}…")
        trades = backtest(top, change, cfg.symbols)
        extra = indicator_lines(trades, cut) if show and len(trades) else []
        # keep only the coins that made money in the first part, then rerun with just those
        # (one position at a time means the coins compete for the slot)
        keep = [x for x in cfg.symbols if len(trades) and
                trades[(trades.symbol == x) & (trades.open_time < cut)].pnl.sum() > 0]
        if keep == cfg.symbols:
            picked = None  # every coin made money: the all-coin line already says it
        else:
            picked = _halves(backtest(top, change, keep), cut) if keep else None
        rows.append((name, _halves(trades, cut), keep, picked, extra))
    score = lambda r: max(r[1]["train"][0], r[3]["train"][0] if r[3] else -1e9)  # noqa: E731
    rows.sort(key=lambda r: -score(r))

    def describe(label, part):
        (a, na, wa, ga), (b, nb, wb, gb) = part["train"], part["test"]
        return [f"   {label}：约 {(na + nb) / days:.1f} 单/天",
                f"     挑选段 {a:+.2f} USDT，{na} 单，胜率 {wa:.0f}%（不算手续费 {ga:+.2f}，"
                f"每单 {ga / max(na, 1):+.2f}）",
                f"     验证段 {b:+.2f} USDT，{nb} 单，胜率 {wb:.0f}%（不算手续费 {gb:+.2f}，"
                f"每单 {gb / max(nb, 1):+.2f}）"]

    lines = [f"{short(cfg.symbols)}，真实 K 线 {days:.0f} 天，每单保证金 {cfg.max_margin_usd:g}，已扣手续费",
             f"前 {(cut - t0) / day:.0f} 天用来挑（排名、挑币），后 {(t1 - cut) / day:.0f} 天用来验证（挑的时候没看过）。",
             "两段都赚的才算靠得住。回测只是参考，不保证以后一样。", ""]
    good = []
    def passes(part):
        n = part["train"][1] + part["test"][1]
        return part["train"][0] > 0 and part["test"][0] > 0 and n >= MIN_TRADES

    for rank, (name, every, keep, picked, extra) in enumerate(rows, 1):
        lines.append(f"{rank}. {name}")
        if passes(every):
            good.append((name, cfg.symbols, every))
        lines += describe("全部币" + ("  ✓ 两段都赚" if passes(every) else ""), every)
        if picked:
            n = picked["train"][1] + picked["test"][1]
            ok = picked["train"][0] > 0 and picked["test"][0] > 0 and n >= MIN_TRADES
            if ok:
                good.append((name, keep, picked))
            note = "  ✓ 两段都赚" if ok else ("  （单数太少，不算数）" if n < MIN_TRADES else "")
            lines += describe(f"只做 {short(keep)}" + note, picked)
        elif not keep:
            lines.append("   前段没有一个币赚钱")
        lines += extra
        lines.append("")
    if good:
        lines.append("结论：两段都赚的：")
        for name, keep, part in good:
            n = part["train"][1] + part["test"][1]
            lines.append(f"  {name}，币：{short(keep)}，约 {n / days:.1f} 单/天")
    else:
        lines.append("结论：没有一种在两段都赚。")
    report = "\n".join(lines)
    (out / "tune_report.txt").write_text(report, encoding="utf-8")
    say("完成")
    print("\n" + report)


def cmd_selftest(cfg, args):
    """测试下单: run every order type once on the Bitget demo account and write a report."""
    from bot.exchange import BitgetBroker
    from bot.selftest import report, run_selftest
    if cfg.mode != "demo":
        sys.exit("测试下单只在模拟盘（mode: demo）上跑")
    sym = cfg.symbols[0]
    text = report(sym, run_selftest(BitgetBroker(cfg, demo=True), sym, cfg.max_margin_usd))
    out = Path(args.data_dir)
    out.mkdir(exist_ok=True)
    (out / "selftest_report.txt").write_text(text, encoding="utf-8")
    print(text)


def cmd_flowtest(cfg, args):
    """分析盘口: test order-flow signals on the order book recorded by 记录盘口."""
    from bot.flowtest import analyse, load
    out = Path(args.data_dir)
    out.mkdir(exist_ok=True)
    (out / "tune_progress.txt").write_text("分析盘口数据…", encoding="utf-8")
    from bot.fullstudy import study
    report = analyse(load(out))
    (out / "tune_progress.txt").write_text("全面分析中…", encoding="utf-8")
    report += "\n\n" + "=" * 30 + "\n\n" + study(out)
    (out / "flowtest_report.txt").write_text(report, encoding="utf-8")
    print(report)


def cmd_record(cfg, args):
    """Record order book + trades (盘口) for the chosen coins until stopped."""
    import atexit
    from bot.orderflow import record
    pid = Path(args.data_dir) / "record.pid"
    pid.parent.mkdir(parents=True, exist_ok=True)
    pid.write_text(str(os.getpid()))
    atexit.register(lambda: pid.unlink(missing_ok=True))
    record(cfg.symbols, args.data_dir, args.interval)


def write_pid_file(cfg):
    """Lets stop_bot.bat find and stop a bot running in the background."""
    import atexit
    pid_file = Path(cfg.state_file).with_suffix(".pid")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    atexit.register(lambda: pid_file.unlink(missing_ok=True))


def cmd_run(cfg, args):
    from bot.engine import Engine
    from bot.exchange import BitgetBroker, PaperBroker
    if cfg.mode == "live":
        if os.environ.get("I_UNDERSTAND_REAL_MONEY") != "yes":
            sys.exit("实盘模式需要先设置环境变量 I_UNDERSTAND_REAL_MONEY=yes（确认你知道这是真钱）")
        broker = BitgetBroker(cfg, demo=False)
        logging.warning("LIVE MODE: REAL MONEY")
    elif cfg.mode == "demo":
        broker = BitgetBroker(cfg, demo=True)
    else:
        broker = PaperBroker(cfg, args.balance)
    write_pid_file(cfg)
    Engine(cfg, broker).run()


def force_utf8_output():
    """Windows consoles and redirected output default to cp1252/gbk, which cannot print the
    banner or Chinese messages; never let printing crash the bot."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def apply_overrides(cfg, coins: str | None, margin: float | None) -> None:
    """Control-panel choices: which coins to trade and one fixed margin for every trade."""
    from bot.config import validate
    if coins:
        cfg.symbols = [f"{c.strip().upper()}/USDT:USDT" for c in coins.split(",") if c.strip()]
        if not cfg.symbols:
            raise SystemExit("--coins 至少要有一个币")
    if margin:
        cfg.fixed_margin_usd = cfg.max_margin_usd = margin
        cfg.min_margin_usd = min(cfg.min_margin_usd, margin)
    validate(cfg)


def main():
    force_utf8_output()
    from bot.config import load_config
    ap = argparse.ArgumentParser(description="SmartCrypto Bitget futures bot")
    ap.add_argument("command", choices=["download", "backtest", "compare", "tune", "run", "record",
                                        "selftest", "flowtest"])
    ap.add_argument("--configs", nargs="+", default=["config.yaml", "config_hf.yaml"],
                    help="compare 用的配置文件")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--exchange", default="bitget", help="历史数据来源，例如 bitget 或 binanceusdm")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--balance", type=float, default=100.0, help="回测/本地模拟的起始资金")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--coins", help="只开这些币，覆盖配置文件，例如 BTC,ETH,SOL")
    ap.add_argument("--margin", type=float, help="每单固定保证金 USDT，覆盖配置文件的 5–15 区间")
    ap.add_argument("--interval", type=float, default=1.0, help="record: 每隔几秒记录一次盘口")
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.coins, args.margin)
    # only the trading bot writes the bot log the panel shows
    setup_logging(cfg.log_file if args.command == "run" else str(Path(args.data_dir) / "tools.log"))
    print(BANNER)
    {"download": cmd_download, "backtest": cmd_backtest, "compare": cmd_compare,
     "tune": cmd_tune, "run": cmd_run, "record": cmd_record,
     "selftest": cmd_selftest, "flowtest": cmd_flowtest}[args.command](cfg, args)


if __name__ == "__main__":
    main()
