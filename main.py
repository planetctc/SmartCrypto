"""smartMonitoring - Bitget USDT-M futures bot.

用法:
  python main.py download --days 180            下载历史 K 线
  python main.py backtest                        用下载的数据回测
  python main.py backtest --synthetic            用随机生成的数据跑一遍（只测试程序，不代表真实表现）
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
  │   smartMonitoring  ·  Bitget Futures │
  └──────────────────────────────────────┘
"""


def setup_logging(log_file: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout),
              RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


def cmd_download(cfg, args):
    from bot.data import download
    for sym in cfg.symbols:
        download(sym, cfg.timeframe, args.days, args.exchange, args.data_dir)


def cmd_backtest(cfg, args):
    from bot.backtest import run_backtest, synthetic
    from bot.data import csv_path, load_csv
    if args.synthetic:
        frames = {s: synthetic(seed=i, start_price=[60_000, 3_000][i % 2])
                  for i, s in enumerate(cfg.symbols)}
        print("注意：这是随机生成的数据，只用来检查程序，不代表真实行情表现。")
    else:
        frames = {}
        for s in cfg.symbols:
            p = csv_path(args.data_dir, args.exchange, s, cfg.timeframe)
            if not p.exists():
                sys.exit(f"找不到 {p}，先运行: python main.py download --exchange {args.exchange}")
            frames[s] = load_csv(p)
    stats, trades, curve = run_backtest(cfg, frames, args.balance)
    out = Path(args.data_dir)
    out.mkdir(exist_ok=True)
    trades.to_csv(out / "backtest_trades.csv", index=False)
    curve.to_csv(out / "backtest_equity.csv", index=False)
    print("\n===== 回测结果 =====")
    for k, v in stats.items():
        print(f"{k:>24}: {v}")
    print(f"\n每笔交易: {out / 'backtest_trades.csv'}\n资金曲线: {out / 'backtest_equity.csv'}")


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
    Engine(cfg, broker).run()


def main():
    from bot.config import load_config
    ap = argparse.ArgumentParser(description="smartMonitoring Bitget futures bot")
    ap.add_argument("command", choices=["download", "backtest", "run"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--exchange", default="bitget", help="历史数据来源，例如 bitget 或 binanceusdm")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--balance", type=float, default=100.0, help="回测/本地模拟的起始资金")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.log_file)
    print(BANNER)
    {"download": cmd_download, "backtest": cmd_backtest, "run": cmd_run}[args.command](cfg, args)


if __name__ == "__main__":
    main()
