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


TUNE_VARIANTS = [
    ("现在的设置", {}),
    ("关掉补仓", {"scale_in_at_r": 0}),
    ("补仓拉远到 0.8 倍", {"scale_in_at_r": 0.8}),
    ("一次开满，不分批", {"scale_in_frac": 1.0}),
    ("关掉补仓 + 持仓不限时", {"scale_in_at_r": 0, "max_hold_bars": 10**9}),
]


def cmd_tune(cfg, args):
    """Download recent real candles and backtest a few strategy variants side by side.
    This is what the control panel's 回测比较 button runs."""
    import copy

    from bot.backtest import run_backtest
    from bot.data import download, load_csv
    out = Path(args.data_dir)
    out.mkdir(exist_ok=True)
    progress = out / "tune_progress.txt"

    def say(msg):
        print(msg)
        progress.write_text(msg, encoding="utf-8")

    frames = {}
    for i, sym in enumerate(cfg.symbols, 1):
        say(f"下载 {sym.split('/')[0]} 最近 {args.days} 天的 K 线（{i}/{len(cfg.symbols)}）…")
        frames[sym] = load_csv(download(sym, cfg.timeframe, args.days, args.exchange, args.data_dir))
    rows = []
    for i, (name, change) in enumerate(TUNE_VARIANTS, 1):
        say(f"回测 {i}/{len(TUNE_VARIANTS)}：{name}…")
        c = copy.deepcopy(cfg)
        c.risk.min_equity_stop_usd = 0  # run the whole period so every variant is compared fully
        for k, v in change.items():
            setattr(c.strategy, k, v)
        stats, _, _ = run_backtest(c, frames, args.balance)
        rows.append((name, stats))
    span = max(f["ts"].iloc[-1] - f["ts"].iloc[0] for f in frames.values()) / 86_400_000
    coins = "、".join(s.split("/")[0] for s in cfg.symbols)
    lines = [f"{coins}，{'高频' if cfg.timeframe == '1m' else '普通'}模式，最近 {span:.0f} 天真实 K 线，"
             f"本金 {args.balance:.0f} USDT，每单保证金 {cfg.max_margin_usd:g}",
             "已扣手续费。回测只是参考，不保证以后一样。", ""]
    rows.sort(key=lambda r: -r[1]["end_equity"])
    for rank, (name, st) in enumerate(rows, 1):
        net, n = st["end_equity"] - args.balance, st["trades"]
        lines += [f"{rank}. {name}" + ("  ← 最好" if rank == 1 else ""),
                  f"   净赚 {net:+.2f} USDT（{st['return_pct']:+.1f}%），最大回撤 "
                  f"{st.get('max_drawdown_pct', 0):.1f}%",
                  f"   {n} 单，每天 {n / max(span, 1e-9):.1f} 单，胜率 {st.get('win_rate_pct', 0):.0f}%，"
                  f"平均每单 {net / n if n else 0:+.2f} USDT", ""]
    report = "\n".join(lines)
    (out / "tune_report.txt").write_text(report, encoding="utf-8")
    say("完成")
    print("\n" + report)


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
    ap.add_argument("command", choices=["download", "backtest", "compare", "tune", "run"])
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
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.coins, args.margin)
    # only the trading bot writes the bot log the panel shows
    setup_logging(cfg.log_file if args.command == "run" else str(Path(args.data_dir) / "tools.log"))
    print(BANNER)
    {"download": cmd_download, "backtest": cmd_backtest, "compare": cmd_compare,
     "tune": cmd_tune, "run": cmd_run}[args.command](cfg, args)


if __name__ == "__main__":
    main()
